# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from threading import Lock

import pandas as pd
import pytest
from pydantic import BaseModel

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    EntityType,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.llm_client import TransientInferenceError
from nemo_safe_synthesizer.pii_replacer.planning import (
    LLMPlanEnhancer,
    PlanDiscoverer,
    PlanDiscoveryInput,
    resolve_plan,
)
from nemo_safe_synthesizer.pii_replacer.planning import llm as llm_module
from nemo_safe_synthesizer.pii_replacer.planning.llm import (
    MAX_BATCH_BYTES,
    MAX_BATCH_ENTRIES,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    _bounded_batches,
    _json_bytes,
    _profile_batches,
)


class ScriptedTransport:
    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self._responses = list(responses)
        self._lock = Lock()
        self.calls: list[tuple[list[dict[str, str]], type[BaseModel]]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_model: type[BaseModel],
    ) -> str:
        with self._lock:
            self.calls.append(([dict(message) for message in messages], response_model))
            response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FixedDiscoverer(PlanDiscoverer):
    def __init__(self, plan: PiiReplacementPlan) -> None:
        self.plan = plan

    def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
        return self.plan


def _local_config(*, max_workers: int = 8) -> LLMConfig:
    return LLMConfig(model_id="local-model", max_workers=max_workers)


def _classifications(
    entities: Mapping[str, str | None],
    *,
    patterns: Mapping[str, str] | None = None,
) -> str:
    pattern_by_column = patterns or {}
    return json.dumps(
        {
            "classifications": [
                {
                    "column_name": column,
                    "entity_type": entity_type,
                    "pattern": pattern_by_column.get(column),
                }
                for column, entity_type in entities.items()
            ]
        }
    )


def _dependency_selection(choices: Mapping[str, Mapping[str, str | None]] | None = None) -> str:
    """Return a dependency answer: target column -> source entity type -> source column or null."""
    return json.dumps(dict(choices or {}))


def _enhancer(
    responses: Sequence[str | Exception],
    *,
    max_workers: int = 8,
    sleeps: list[float] | None = None,
) -> tuple[LLMPlanEnhancer, ScriptedTransport]:
    transport = ScriptedTransport(responses)
    recorded_sleeps = sleeps if sleeps is not None else []
    enhancer = LLMPlanEnhancer(
        _local_config(max_workers=max_workers),
        transport=transport,
        environ={"NSS_INFERENCE_ENDPOINT": "http://localhost:8000/v1"},
        sleep=recorded_sleeps.append,
    )
    return enhancer, transport


@pytest.mark.unit
class TestClassificationBatching:
    def test_batches_obey_count_and_byte_limits(self) -> None:
        dataframe = pd.DataFrame({f"column_{index}": [f"{index}-" + "x" * 128] for index in range(100)})
        captured: list[PlanDiscoveryInput] = []

        class CapturingDiscoverer(PlanDiscoverer):
            def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
                captured.append(discovery_input)
                return PiiReplacementPlan()

        resolve_plan(dataframe, ReplacePiiConfig(), DataParameters(), discoverer=CapturingDiscoverer())
        batches = _profile_batches(captured[0].column_profiles)

        assert len(batches) > 1
        assert all(len(batch) <= MAX_BATCH_ENTRIES for batch in batches)
        assert all(_json_bytes(batch) <= MAX_BATCH_BYTES for batch in batches)


@pytest.mark.unit
class TestDependencyBatching:
    def test_dependency_targets_are_batched_and_merged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(llm_module, "MAX_BATCH_ENTRIES", 1)
        dataframe = pd.DataFrame(
            {
                "first_name": ["Ada"],
                "gender": ["F"],
                "spouse_first_name": ["William"],
                "spouse_gender": ["M"],
            }
        )
        classifications = {
            "first_name": "first_name",
            "gender": "gender",
            "spouse_first_name": "first_name",
            "spouse_gender": "gender",
        }
        # One worker keeps batch order deterministic for the scripted responses.
        enhancer, transport = _enhancer(
            [
                *[_classifications({column: entity}) for column, entity in classifications.items()],
                _dependency_selection({"first_name": {"gender": "gender"}}),
                _dependency_selection({"spouse_first_name": {"gender": "spouse_gender"}}),
            ],
            max_workers=1,
        )

        plan = resolve_plan(dataframe, ReplacePiiConfig(llm=_local_config()), DataParameters(), enhancer=enhancer)

        depends_on = {
            spec.column_name: [item.column_name for item in spec.depends_on] for spec in plan.columns_to_replace
        }
        assert depends_on == {"first_name": ["gender"], "spouse_first_name": ["spouse_gender"]}
        dependency_targets = [
            [target["target_column"] for target in json.loads(messages[1]["content"])["dependency_targets"]]
            for messages, _ in transport.calls[len(classifications) :]
        ]
        assert dependency_targets == [["first_name"], ["spouse_first_name"]]

    def test_oversized_dependency_target_is_rejected(self) -> None:
        with pytest.raises(ParameterError, match="Dependency options for 'email' exceeds the 48 KiB"):
            _bounded_batches(
                [{"target_column": "email", "source_options": "x" * MAX_BATCH_BYTES}],
                kind="Dependency options",
                name_key="target_column",
            )


@pytest.mark.unit
class TestLLMPlanEnhancer:
    def test_two_pass_enhancement_classifies_then_selects_candidate_ids(self) -> None:
        dataframe = pd.DataFrame(
            {
                "company": ["Analytical Engines", "US Navy"],
                "email": ["ada@example.com", "grace@example.com"],
            }
        )
        baseline = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="company", entity_type=EntityType.FULL_NAME)]
        )
        enhancer, transport = _enhancer(
            [
                _classifications({"company": "organization", "email": "email"}),
                _dependency_selection(),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            discoverer=FixedDiscoverer(baseline),
            enhancer=enhancer,
        )

        assert [spec.column_name for spec in plan.columns_to_replace] == ["email"]
        assert len(transport.calls) == 2

        classification_messages, classification_model = transport.calls[0]
        classification_payload = json.loads(classification_messages[1]["content"])
        assert classification_payload["discovery_context"] == {"group_column": None}
        assert '"classifications":[{"column_name":' in classification_messages[0]["content"]
        assert set(classification_payload["column_profiles"][0]) == {
            "column_name",
            "dtype",
            "non_null_count",
            "unique_count",
            "unique_ratio",
            "samples",
        }
        assert classification_payload["heuristic_classifications"] == [
            {"column_name": "company", "entity_type": "full_name", "pattern": None}
        ]
        assert "heuristic_baseline" not in classification_payload
        assert "disposition" not in json.dumps(classification_model.model_json_schema())

        dependency_messages, dependency_model = transport.calls[1]
        dependency_payload = json.loads(dependency_messages[1]["content"])
        assert dependency_payload == {
            "dependency_targets": [
                {
                    "target_column": "email",
                    "target_entity_type": "email",
                    "target_pattern": None,
                    "target_pattern_syntax": None,
                    "source_options": {"organization": {"columns": ["company"], "heuristic_choice": None}},
                }
            ],
            "exclusive_dependency_groups": [
                [["first_name", "last_name", "middle_name"], ["full_name"]],
                [["full_name"], ["ethnic_background", "gender"]],
                [["zipcode"], ["city", "country", "state"]],
            ],
            "pattern_grammars": {},
        }
        assert (
            "the source entity types chosen for one target may come from at most one inner group"
            in dependency_messages[0]["content"]
        )
        assert "describes the same person or record as the target" in dependency_messages[0]["content"]
        assert "Return a JSON object with one key per target column" in dependency_messages[0]["content"]
        dependency_schema = dependency_model.model_json_schema()
        assert set(dependency_schema["properties"]) == {"email"}
        [email_choices] = dependency_schema["$defs"].values()
        allowed = email_choices["properties"]["organization"]["anyOf"][0]
        assert allowed.get("enum", [allowed.get("const")]) == ["company"]

    def test_dependency_candidates_preserve_heuristic_selections_as_prior_evidence(self) -> None:
        dataframe = pd.DataFrame(
            {
                "company": ["Analytical Engines", "US Navy"],
                "email": ["ada@example.com", "grace@example.com"],
            }
        )
        baseline = PiiReplacementPlan(
            columns_to_replace=[
                PiiColumnPlan(
                    column_name="email",
                    entity_type=EntityType.EMAIL,
                    depends_on=[
                        ConditioningColumn(
                            column_name="company",
                            entity_type=EntityType.ORGANIZATION,
                        )
                    ],
                )
            ]
        )
        enhancer, transport = _enhancer(
            [
                _classifications({"company": "organization", "email": "email"}),
                _dependency_selection(),
            ]
        )

        resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            discoverer=FixedDiscoverer(baseline),
            enhancer=enhancer,
        )

        dependency_messages, _ = transport.calls[1]
        dependency_payload = json.loads(dependency_messages[1]["content"])
        [target] = dependency_payload["dependency_targets"]
        assert target["source_options"]["organization"]["heuristic_choice"] == "company"
        assert "treat it as a hint, not a requirement" in dependency_messages[0]["content"]

    def test_classification_prompt_includes_exact_supported_pattern_grammars(self) -> None:
        dataframe = pd.DataFrame(
            {
                "phone": ["A7-a8"],
                "email": ["ada1@example.com"],
            }
        )
        enhancer, transport = _enhancer([_classifications({"phone": "phone_number", "email": "email"})])

        resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        messages, _ = transport.calls[0]
        assert "NSS decides on its own which columns to replace" in messages[0]["content"]
        assert "do not require name columns in the table" in messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        assert payload["pattern_grammars"]["character_mask"]["tokens"]["&"] == "digit or uppercase letter"
        assert payload["pattern_grammars"]["character_mask"]["tokens"]["%"] == "digit or lowercase letter"
        assert "In email patterns, # emits one digit." in payload["pattern_grammars"]["name_parts"]["rules"]
        assert all(set(entity) == {"entity_type", "pattern_syntax"} for entity in payload["entity_catalog"])

    def test_dependency_candidates_include_proposed_target_pattern(self) -> None:
        dataframe = pd.DataFrame(
            {
                "first_name": ["Ada", "Grace"],
                "email": ["ada@example.com", "grace@example.com"],
            }
        )
        enhancer, transport = _enhancer(
            [
                _classifications(
                    {"first_name": "first_name", "email": "email"},
                    patterns={"email": "{first}@{domain}"},
                ),
                _dependency_selection(),
            ]
        )

        resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        dependency_payload = json.loads(transport.calls[1][0][1]["content"])
        [target] = dependency_payload["dependency_targets"]
        assert target["target_column"] == "email"
        assert target["source_options"] == {"first_name": {"columns": ["first_name"], "heuristic_choice": None}}
        assert target["target_pattern"] == "{first}@{domain}"
        assert target["target_pattern_syntax"] == "name_parts"
        assert set(dependency_payload["pattern_grammars"]) == {"name_parts"}
        assert "{first}" in dependency_payload["pattern_grammars"]["name_parts"]["placeholders"]

    def test_code_excludes_identify_only_and_ordering_but_not_group_classifications(self) -> None:
        dataframe = pd.DataFrame(
            {
                "patient_id": [1, 2],
                "event_index": [0, 0],
                "first_name": ["Ada", "Grace"],
                "sex": ["F", "F"],
                "weight": [50, 60],
            }
        )
        enhancer, transport = _enhancer(
            [
                _classifications(
                    {
                        "patient_id": "unique_identifier",
                        "event_index": "unique_identifier",
                        "first_name": "first_name",
                        "sex": "gender",
                        "weight": None,
                    }
                ),
                _dependency_selection(),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(
                group_training_examples_by="patient_id",
                order_training_examples_by="event_index",
            ),
            enhancer=enhancer,
        )

        assert [spec.column_name for spec in plan.columns_to_replace] == ["patient_id", "first_name"]
        classification_payload = json.loads(transport.calls[0][0][1]["content"])
        # Protected columns are excluded in code, so the prompt never mentions them.
        assert classification_payload["discovery_context"] == {"group_column": "patient_id"}
        assert "protected" not in transport.calls[0][0][0]["content"]
        dependency_payload = json.loads(transport.calls[1][0][1]["content"])
        assert dependency_payload["dependency_targets"] == [
            {
                "target_column": "first_name",
                "target_entity_type": "first_name",
                "target_pattern": None,
                "target_pattern_syntax": None,
                "source_options": {"gender": {"columns": ["sex"], "heuristic_choice": None}},
            }
        ]

    def test_selected_dependencies_are_applied_to_the_plan(self) -> None:
        dataframe = pd.DataFrame(
            {
                "first_name": ["Ada", "Grace"],
                "sex": ["F", "F"],
                "race": ["White", "White"],
            }
        )
        enhancer, _ = _enhancer(
            [
                _classifications(
                    {
                        "first_name": "first_name",
                        "sex": "gender",
                        "race": "ethnic_background",
                    }
                ),
                _dependency_selection({"first_name": {"gender": "sex", "ethnic_background": "race"}}),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert [(item.column_name, item.entity_type) for item in plan.columns_to_replace[0].depends_on] == [
            ("sex", EntityType.GENDER),
            ("race", EntityType.ETHNIC_BACKGROUND),
        ]

    def test_each_target_chooses_one_source_per_entity_type(self) -> None:
        dataframe = pd.DataFrame(
            {
                "first_name": ["Ada"],
                "gender": ["F"],
                "spouse_first_name": ["William"],
                "spouse_gender": ["M"],
            }
        )
        enhancer, transport = _enhancer(
            [
                _classifications(
                    {
                        "first_name": "first_name",
                        "gender": "gender",
                        "spouse_first_name": "first_name",
                        "spouse_gender": "gender",
                    }
                ),
                _dependency_selection(
                    {
                        "first_name": {"gender": "gender"},
                        "spouse_first_name": {"gender": "spouse_gender"},
                    }
                ),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        depends_on = {
            spec.column_name: [item.column_name for item in spec.depends_on] for spec in plan.columns_to_replace
        }
        assert depends_on == {"first_name": ["gender"], "spouse_first_name": ["spouse_gender"]}
        dependency_payload = json.loads(transport.calls[1][0][1]["content"])
        assert [
            target["source_options"]["gender"]["columns"] for target in dependency_payload["dependency_targets"]
        ] == [
            ["gender", "spouse_gender"],
            ["gender", "spouse_gender"],
        ]

    def test_dependency_answer_cannot_choose_two_sources_of_one_entity_type(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "gender": ["F"], "spouse_gender": ["M"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"first_name": "first_name", "gender": "gender", "spouse_gender": "gender"}),
                json.dumps({"first_name": {"gender": ["gender", "spouse_gender"]}}),
                _dependency_selection({"first_name": {"gender": "gender"}}),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert [item.column_name for item in plan.columns_to_replace[0].depends_on] == ["gender"]
        assert len(transport.calls) == 3
        _, dependency_model = transport.calls[1]
        [choices] = dependency_model.model_json_schema()["$defs"].values()
        assert choices["properties"]["gender"]["anyOf"][0]["enum"] == ["gender", "spouse_gender"]

    def test_dependency_pass_is_skipped_when_code_derives_no_candidates(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        enhancer, transport = _enhancer([_classifications({"email": "email"})])

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert [spec.column_name for spec in plan.columns_to_replace] == ["email"]
        assert len(transport.calls) == 1

    def test_missing_classification_retries_with_validation_feedback(self) -> None:
        dataframe = pd.DataFrame({"name": ["Ada"], "email": ["ada@example.com"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"name": None}),
                _classifications({"name": None, "email": None}),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace == []
        assert len(transport.calls) == 2
        feedback = transport.calls[1][0][-1]["content"]
        assert "previous structured response was invalid" in feedback
        assert "missing: 'email'" in feedback

    def test_duplicate_classification_retries(self) -> None:
        dataframe = pd.DataFrame({"name": ["Ada"], "email": ["ada@example.com"]})
        duplicate = json.dumps(
            {
                "classifications": [
                    {"column_name": "name", "entity_type": None, "pattern": None},
                    {"column_name": "name", "entity_type": None, "pattern": None},
                ]
            }
        )
        enhancer, transport = _enhancer([duplicate, _classifications({"name": None, "email": None})])

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace == []
        assert len(transport.calls) == 2
        feedback = transport.calls[1][0][-1]["content"]
        assert "missing: 'email'" in feedback
        assert "duplicated: 'name'" in feedback

    @pytest.mark.parametrize(
        ("invalid_selection", "expected_feedback"),
        [
            (_dependency_selection({"first_name": {"gender": "invented"}}), "first_name.gender"),
            (_dependency_selection({"first_name": {"full_name": "sex"}}), "first_name.full_name"),
        ],
        ids=["unknown-source-column", "entity-type-not-offered"],
    )
    def test_invalid_dependency_selection_is_repaired_on_retry(
        self, invalid_selection: str, expected_feedback: str
    ) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["F"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"first_name": "first_name", "sex": "gender"}),
                invalid_selection,
                _dependency_selection(),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace[0].depends_on == []
        assert len(transport.calls) == 3
        feedback = transport.calls[2][0][-1]["content"]
        assert "previous structured response was invalid" in feedback
        assert expected_feedback in feedback

    def test_dependency_selection_response_is_strict(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["F"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"first_name": "first_name", "sex": "gender"}),
                json.dumps({"first_name": {"gender": None}, "unexpected": {}}),
                _dependency_selection(),
            ]
        )

        resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert len(transport.calls) == 3

    def test_conflicting_dependency_selection_is_repaired_on_retry(self) -> None:
        dataframe = pd.DataFrame(
            {
                "first_name": ["Ada"],
                "full_name": ["Ada Lovelace"],
                "sex": ["F"],
            }
        )
        enhancer, transport = _enhancer(
            [
                _classifications(
                    {
                        "first_name": "first_name",
                        "full_name": "full_name",
                        "sex": "gender",
                    }
                ),
                _dependency_selection({"first_name": {"gender": "sex", "full_name": "full_name"}}),
                _dependency_selection(),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert all(not spec.depends_on for spec in plan.columns_to_replace)
        assert len(transport.calls) == 3
        feedback = transport.calls[2][0][-1]["content"]
        assert "selected dependencies do not form a valid replacement plan: " in feedback
        assert "column 'first_name': depends_on mixes mutually exclusive conditioner groups" in feedback

    @pytest.mark.parametrize(
        ("entity_type", "pattern"),
        [
            pytest.param("street_address", "### Main St", id="entity-without-pattern-syntax"),
            pytest.param(None, "### Main St", id="unclassified-column"),
            pytest.param("street_address", "  ", id="blank-pattern"),
        ],
    )
    def test_pattern_the_column_cannot_use_is_ignored(self, entity_type: str | None, pattern: str) -> None:
        dataframe = pd.DataFrame({"address": ["123 Main St"]})
        enhancer, transport = _enhancer([_classifications({"address": entity_type}, patterns={"address": pattern})])

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert all(spec.pattern is None for spec in plan.columns_to_replace)
        assert len(transport.calls) == 1

    def test_classification_prompt_names_entity_types_without_patterns(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        enhancer, transport = _enhancer([_classifications({"email": "email"})])

        resolve_plan(dataframe, ReplacePiiConfig(llm=_local_config()), DataParameters(), enhancer=enhancer)

        system_prompt = transport.calls[0][0][0]["content"]
        assert "These entity types never take a pattern: street_address, ssn, national_id" in system_prompt
        assert "If the values mix formats, set pattern to null." in system_prompt

    def test_pattern_for_protected_ordering_column_is_ignored(self) -> None:
        dataframe = pd.DataFrame(
            {
                "patient_id": [1, 2],
                "event_index": [0, 0],
                "email": ["a@example.com", "b@example.com"],
            }
        )
        enhancer, transport = _enhancer(
            [
                _classifications(
                    {
                        "patient_id": "unique_identifier",
                        "event_index": "unique_identifier",
                        "email": "email",
                    },
                    patterns={"event_index": "#"},
                ),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(
                group_training_examples_by="patient_id",
                order_training_examples_by="event_index",
            ),
            enhancer=enhancer,
        )

        assert [spec.column_name for spec in plan.columns_to_replace] == ["patient_id", "email"]
        assert len(transport.calls) == 1

    def test_malformed_responses_fail_without_exposing_samples(self) -> None:
        dataframe = pd.DataFrame({"secret": ["raw-private-value"]})
        enhancer, _ = _enhancer(['{"classifications":', '{"classifications":', '{"classifications":'])

        with pytest.raises(GenerationError) as exc_info:
            resolve_plan(
                dataframe,
                ReplacePiiConfig(llm=_local_config()),
                DataParameters(),
                enhancer=enhancer,
            )

        assert "after 3 attempts" in str(exc_info.value)
        assert "raw-private-value" not in str(exc_info.value)

    def test_transient_transport_failure_is_retried(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        enhancer, transport = _enhancer(
            [
                TransientInferenceError("temporary"),
                _classifications({"email": "email"}),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert [spec.column_name for spec in plan.columns_to_replace] == ["email"]
        assert len(transport.calls) == 2

    def test_transient_failures_back_off_exponentially_with_jitter(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        sleeps: list[float] = []
        enhancer, transport = _enhancer(
            [
                TransientInferenceError("temporary"),
                TransientInferenceError("temporary"),
                _classifications({"email": "email"}),
            ],
            sleeps=sleeps,
        )

        resolve_plan(dataframe, ReplacePiiConfig(llm=_local_config()), DataParameters(), enhancer=enhancer)

        assert len(transport.calls) == 3
        assert len(sleeps) == 2
        assert RETRY_BASE_DELAY_SECONDS / 2 <= sleeps[0] <= RETRY_BASE_DELAY_SECONDS
        assert RETRY_BASE_DELAY_SECONDS <= sleeps[1] <= 2 * RETRY_BASE_DELAY_SECONDS

    @pytest.mark.parametrize(
        ("retry_after", "expected_sleep"),
        [(7.0, 7.0), (10 * RETRY_MAX_DELAY_SECONDS, RETRY_MAX_DELAY_SECONDS)],
    )
    def test_transient_failure_honors_capped_retry_after(self, retry_after: float, expected_sleep: float) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        sleeps: list[float] = []
        enhancer, _ = _enhancer(
            [
                TransientInferenceError("rate limited", retry_after=retry_after),
                _classifications({"email": "email"}),
            ],
            sleeps=sleeps,
        )

        resolve_plan(dataframe, ReplacePiiConfig(llm=_local_config()), DataParameters(), enhancer=enhancer)

        assert sleeps == [expected_sleep]

    def test_transient_failure_keeps_previous_validation_feedback(self) -> None:
        dataframe = pd.DataFrame({"name": ["Ada"], "email": ["ada@example.com"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"name": None}),
                TransientInferenceError("temporary"),
                _classifications({"name": None, "email": None}),
            ]
        )

        resolve_plan(dataframe, ReplacePiiConfig(llm=_local_config()), DataParameters(), enhancer=enhancer)

        assert len(transport.calls) == 3
        for messages, _ in transport.calls[1:]:
            assert "missing: 'email'" in messages[-1]["content"]

    def test_exhausted_transient_failures_fail_planning(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        sleeps: list[float] = []
        enhancer, transport = _enhancer([TransientInferenceError("temporary")] * 3, sleeps=sleeps)

        with pytest.raises(GenerationError, match="PII column classification failed after 3 attempts") as exc_info:
            resolve_plan(dataframe, ReplacePiiConfig(llm=_local_config()), DataParameters(), enhancer=enhancer)

        assert isinstance(exc_info.value.__cause__, TransientInferenceError)
        assert len(transport.calls) == 3
        assert len(sleeps) == 2

    def test_exhausted_transient_failures_during_pattern_repair_fail_planning(self) -> None:
        dataframe = pd.DataFrame({"phone": ["+1-415-555-0100", "+1-212-555-0199"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"phone": "phone_number"}, patterns={"phone": "literal"}),
                *[TransientInferenceError("temporary")] * 3,
            ]
        )

        with pytest.raises(GenerationError, match="PII pattern repair failed after 3 attempts"):
            resolve_plan(dataframe, ReplacePiiConfig(llm=_local_config()), DataParameters(), enhancer=enhancer)

        assert len(transport.calls) == 4

    def test_logs_exclude_samples_prompts_and_responses(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.DEBUG)
        dataframe = pd.DataFrame({"phone": ["+1-415-555-0100", "+1-212-555-0199"], "secret": ["raw-private-value"] * 2})
        enhancer, transport = _enhancer(
            [
                '{"classifications": "raw-response-text"}',
                _classifications({"phone": "phone_number", "secret": None}, patterns={"phone": "literal"}),
                json.dumps({"pattern": "raw-repair-1"}),
                json.dumps({"pattern": "raw-repair-2"}),
                json.dumps({"pattern": "raw-repair-3"}),
            ]
        )

        plan = resolve_plan(dataframe, ReplacePiiConfig(llm=_local_config()), DataParameters(), enhancer=enhancer)

        assert plan.columns_to_replace[0].pattern is None
        assert len(transport.calls) == 5
        assert "Dropping an invalid LLM-proposed pattern" in caplog.text
        logged = caplog.text + "".join(repr(vars(record)) for record in caplog.records)
        for private_text in ("raw-private-value", "+1-415-555-0100", "raw-response-text", "raw-repair-", "Classify"):
            assert private_text not in logged

    def test_authentication_failure_is_not_retried(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        enhancer, transport = _enhancer([ParameterError("authentication failed")])

        with pytest.raises(ParameterError, match="authentication failed"):
            resolve_plan(
                dataframe,
                ReplacePiiConfig(llm=_local_config()),
                DataParameters(),
                enhancer=enhancer,
            )

        assert len(transport.calls) == 1

    def test_invalid_pattern_is_repaired(self) -> None:
        dataframe = pd.DataFrame({"phone": ["+1-415-555-0100", "+1-212-555-0199"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"phone": "phone_number"}, patterns={"phone": "literal"}),
                json.dumps({"pattern": "+1-###-###-####"}),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace[0].pattern == "+1-###-###-####"
        assert len(transport.calls) == 2
        repair_payload = json.loads(transport.calls[1][0][1]["content"])
        assert repair_payload["pattern_syntax"] == "character_mask"
        assert repair_payload["pattern_grammar"]["tokens"]["#"] == "digit 0-9"
        assert '{"pattern":' in transport.calls[1][0][0]["content"]

    def test_pattern_covering_too_few_values_is_dropped_without_repair(self, caplog: pytest.LogCaptureFixture) -> None:
        dataframe = pd.DataFrame({"phone": ["+1-415-555-0100", "(212) 555-0199"]})
        enhancer, transport = _enhancer(
            [_classifications({"phone": "phone_number"}, patterns={"phone": "+1-###-###-####"})]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace[0].pattern is None
        assert len(transport.calls) == 1
        assert "Dropping an LLM-proposed pattern that does not cover the column's values" in caplog.text

    @pytest.mark.parametrize(
        ("repair_responses", "final_pattern", "final_syntax"),
        [
            pytest.param(["{first}@{domain}"], "{first}@{domain}", "name_parts", id="repaired"),
            pytest.param(["no-at-sign-1", "no-at-sign-2", "no-at-sign-3"], None, None, id="dropped"),
        ],
    )
    def test_dependency_selection_sees_repaired_target_pattern(
        self,
        repair_responses: list[str],
        final_pattern: str | None,
        final_syntax: str | None,
    ) -> None:
        dataframe = pd.DataFrame(
            {
                "first_name": ["Ada", "Grace"],
                "email": ["ada@example.com", "grace@example.com"],
            }
        )
        enhancer, transport = _enhancer(
            [
                _classifications(
                    {"first_name": "first_name", "email": "email"},
                    # Missing "@" is a grammar error, which still gets a repair request.
                    patterns={"email": "{first}.{last}"},
                ),
                *[json.dumps({"pattern": pattern}) for pattern in repair_responses],
                _dependency_selection({"email": {"first_name": "first_name"}}),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        dependency_payload = json.loads(transport.calls[-1][0][1]["content"])
        assert set(transport.calls[-1][1].model_json_schema()["properties"]) == {"email"}
        [target] = dependency_payload["dependency_targets"]
        assert target["target_pattern"] == final_pattern
        assert target["target_pattern_syntax"] == final_syntax
        email = next(spec for spec in plan.columns_to_replace if spec.column_name == "email")
        assert email.pattern == final_pattern
        assert [dependency.column_name for dependency in email.depends_on] == ["first_name"]

    def test_exhausted_invalid_pattern_repairs_drop_only_the_pattern(self) -> None:
        dataframe = pd.DataFrame({"phone": ["+1-415-555-0100", "+1-212-555-0199"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"phone": "phone_number"}, patterns={"phone": "literal"}),
                json.dumps({"pattern": "still-literal"}),
                json.dumps({"pattern": "also-literal"}),
                json.dumps({"pattern": "not-a-template"}),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace[0].column_name == "phone"
        assert plan.columns_to_replace[0].entity_type is EntityType.PHONE_NUMBER
        assert plan.columns_to_replace[0].pattern is None
        assert len(transport.calls) == 4

    def test_strftime_pattern_with_repeated_directive_is_repaired_instead_of_crashing(self) -> None:
        dataframe = pd.DataFrame({"dob": ["12/10/1815", "09/12/1906"]})
        repeated = "%m/%d/%Y|%d/%m/%Y"
        enhancer, transport = _enhancer(
            [
                _classifications({"dob": "date_of_birth"}, patterns={"dob": repeated}),
                *[json.dumps({"pattern": repeated})] * 3,
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace[0].column_name == "dob"
        assert plan.columns_to_replace[0].pattern is None
        assert len(transport.calls) == 4
        assert "is not valid strftime" in transport.calls[1][0][1]["content"]
