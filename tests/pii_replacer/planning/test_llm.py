# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from threading import Lock
from typing import cast

import httpx
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
    ConditioningColumn,
    EntityType,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.defaults import DEFAULT_NSS_INFERENCE_ENDPOINT, DEFAULT_NSS_INFERENCE_MODEL
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.planning import (
    LLMPlanEnhancer,
    OpenAICompatibleTransport,
    PlanDiscoverer,
    PlanDiscoveryInput,
    resolve_inference_settings,
    resolve_plan,
)
from nemo_safe_synthesizer.pii_replacer.planning.llm import (
    MAX_CLASSIFICATION_PROFILE_BYTES,
    MAX_CLASSIFICATION_PROFILES,
    _json_bytes,
    _profile_batches,
    _StructuredResponse,
    _TransientInferenceError,
)


class ScriptedTransport:
    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self._responses = list(responses)
        self._lock = Lock()
        self.calls: list[tuple[list[dict[str, str]], type[_StructuredResponse]]] = []

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_model: type[_StructuredResponse],
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
    return LLMConfig(endpoint_url="http://localhost:8000/v1", model_id="local-model", max_workers=max_workers)


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


def _dependency_selection(*candidate_ids: str) -> str:
    return json.dumps({"selected_dependency_ids": list(candidate_ids)})


def _enhancer(
    responses: Sequence[str | Exception],
    *,
    max_workers: int = 8,
) -> tuple[LLMPlanEnhancer, ScriptedTransport]:
    transport = ScriptedTransport(responses)
    enhancer = LLMPlanEnhancer(
        _local_config(max_workers=max_workers),
        transport=transport,
        environ={},
    )
    return enhancer, transport


@pytest.mark.unit
class TestInferenceSettings:
    def test_explicit_overrides_precede_config_and_environment(self) -> None:
        settings = resolve_inference_settings(
            LLMConfig(endpoint_url="https://config.example/v1", model_id="config-model"),
            endpoint_url="http://localhost:9000/v1",
            model_id="cli-model",
            api_key="runtime-key",  # pragma: allowlist secret
            environ={
                "NSS_INFERENCE_ENDPOINT": "https://env.example/v1",
                "NSS_INFERENCE_MODEL": "env-model",
                "NSS_INFERENCE_KEY": "env-key",  # pragma: allowlist secret
            },
        )

        assert settings.endpoint_url == "http://localhost:9000/v1"
        assert settings.model_id == "cli-model"
        assert settings.api_key == "runtime-key"  # pragma: allowlist secret

    def test_config_precedes_environment(self) -> None:
        settings = resolve_inference_settings(
            LLMConfig(endpoint_url="http://config.example/v1", model_id="config-model"),
            environ={
                "NSS_INFERENCE_ENDPOINT": "https://env.example/v1",
                "NSS_INFERENCE_MODEL": "env-model",
            },
        )

        assert settings.endpoint_url == "http://config.example/v1"
        assert settings.model_id == "config-model"

    def test_environment_precedes_defaults(self) -> None:
        settings = resolve_inference_settings(
            LLMConfig(),
            environ={
                "NSS_INFERENCE_ENDPOINT": "http://localhost:8080/v1",
                "NSS_INFERENCE_MODEL": "env-model",
            },
        )

        assert settings.endpoint_url == "http://localhost:8080/v1"
        assert settings.model_id == "env-model"

    def test_defaults_use_hosted_nvidia_service(self) -> None:
        settings = resolve_inference_settings(
            LLMConfig(),
            environ={"NSS_INFERENCE_KEY": "hosted-key"},  # pragma: allowlist secret
        )

        assert settings.endpoint_url == DEFAULT_NSS_INFERENCE_ENDPOINT
        assert settings.model_id == DEFAULT_NSS_INFERENCE_MODEL

    def test_default_hosted_endpoint_requires_runtime_key(self) -> None:
        with pytest.raises(ParameterError, match="NSS_INFERENCE_KEY"):
            resolve_inference_settings(LLMConfig(), environ={})

    def test_local_openai_compatible_endpoint_can_be_keyless(self) -> None:
        settings = resolve_inference_settings(_local_config(), environ={})

        assert settings.api_key is None

    def test_api_key_is_redacted_from_repr(self) -> None:
        settings = resolve_inference_settings(
            _local_config(),
            api_key="do-not-render",  # pragma: allowlist secret
            environ={},
        )

        assert "do-not-render" not in repr(settings)

    def test_invalid_endpoint_fails_without_transport(self) -> None:
        with pytest.raises(ParameterError, match="absolute HTTP"):
            resolve_inference_settings(LLMConfig(endpoint_url="localhost:8000"), environ={})

    def test_endpoint_rejects_embedded_credentials(self) -> None:
        with pytest.raises(ParameterError, match="must not contain credentials"):
            resolve_inference_settings(
                LLMConfig(endpoint_url="https://user:password@example.com/v1"),  # pragma: allowlist secret
                environ={},
            )


@pytest.mark.unit
class TestClassificationBatching:
    def test_batches_obey_count_and_byte_limits(self) -> None:
        dataframe = pd.DataFrame({f"column_{index}": [f"{index}-" + "x" * 128] for index in range(100)})
        captured: list[PlanDiscoveryInput] = []

        class CapturingDiscoverer(PlanDiscoverer):
            def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
                captured.append(discovery_input)
                return PiiReplacementPlan(scope=discovery_input.scope)

        resolve_plan(dataframe, ReplacePiiConfig(), DataParameters(), discoverer=CapturingDiscoverer())
        batches = _profile_batches(captured[0].column_profiles)

        assert len(batches) > 1
        assert all(len(batch) <= MAX_CLASSIFICATION_PROFILES for batch in batches)
        assert all(_json_bytes(batch) <= MAX_CLASSIFICATION_PROFILE_BYTES for batch in batches)


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
        assert classification_payload["heuristic_classifications"] == [
            {"column_name": "company", "entity_type": "full_name", "pattern": None}
        ]
        assert "heuristic_baseline" not in classification_payload
        assert "disposition" not in json.dumps(classification_model.model_json_schema())

        dependency_messages, dependency_model = transport.calls[1]
        dependency_payload = json.loads(dependency_messages[1]["content"])
        assert dependency_payload == {
            "dependency_candidates": [
                {
                    "id": "dependency_0",
                    "source_column": "company",
                    "source_entity_type": "organization",
                    "target_column": "email",
                    "target_entity_type": "email",
                    "selected_by_heuristic": False,
                }
            ]
        }
        assert set(dependency_model.model_json_schema()["properties"]) == {"selected_dependency_ids"}

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
        assert dependency_payload["dependency_candidates"][0]["selected_by_heuristic"] is True
        assert "fallible prior evidence" in dependency_messages[0]["content"]

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
        assert "Do not decide whether a column should be replaced" in messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        assert payload["pattern_grammars"]["character_mask"]["tokens"]["&"] == "digit or uppercase letter"
        assert payload["pattern_grammars"]["character_mask"]["tokens"]["%"] == "digit or lowercase letter"
        assert "In email patterns, # emits one digit." in payload["pattern_grammars"]["name_parts"]["rules"]

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
        dependency_payload = json.loads(transport.calls[1][0][1]["content"])
        assert dependency_payload["dependency_candidates"] == [
            {
                "id": "dependency_0",
                "source_column": "sex",
                "source_entity_type": "gender",
                "target_column": "first_name",
                "target_entity_type": "first_name",
                "selected_by_heuristic": False,
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
                _dependency_selection("dependency_0", "dependency_1"),
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
        assert "previous structured response was invalid" in transport.calls[1][0][-1]["content"]

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

    @pytest.mark.parametrize(
        "invalid_selection",
        [
            _dependency_selection("invented"),
            _dependency_selection("dependency_0", "dependency_0"),
        ],
        ids=["unknown-id", "duplicate-id"],
    )
    def test_invalid_dependency_selection_is_repaired_on_retry(self, invalid_selection: str) -> None:
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
        assert "previous structured response was invalid" in transport.calls[2][0][-1]["content"]

    def test_dependency_selection_response_is_strict(self) -> None:
        dataframe = pd.DataFrame({"first_name": ["Ada"], "sex": ["F"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"first_name": "first_name", "sex": "gender"}),
                json.dumps({"selected_dependency_ids": [], "unexpected": "field"}),
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
                _dependency_selection("dependency_0", "dependency_1"),
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

    def test_pattern_for_unsupported_entity_type_is_retried(self) -> None:
        dataframe = pd.DataFrame({"address": ["123 Main St"]})
        enhancer, transport = _enhancer(
            [
                _classifications({"address": "street_address"}, patterns={"address": "### Main St"}),
                _classifications({"address": "street_address"}),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace[0].pattern is None
        assert len(transport.calls) == 2

    def test_pattern_for_protected_ordering_column_is_retried(self) -> None:
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
                _classifications(
                    {
                        "patient_id": "unique_identifier",
                        "event_index": "unique_identifier",
                        "email": "email",
                    }
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
        assert len(transport.calls) == 2

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
                _TransientInferenceError("temporary"),
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


@pytest.mark.unit
class TestOpenAICompatibleTransport:
    def test_sends_chat_completions_with_strict_json_schema(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = resolve_inference_settings(
            _local_config(),
            api_key="runtime-key",  # pragma: allowlist secret
            environ={},
        )
        transport = OpenAICompatibleTransport(settings)
        captured: dict[str, object] = {}

        def post(url: str, **kwargs: object) -> httpx.Response:
            captured["url"] = url
            captured.update(kwargs)
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

        monkeypatch.setattr(httpx, "post", post)

        content = transport.complete(
            messages=[{"role": "user", "content": "request"}],
            response_model=_StructuredResponse,
        )

        assert content == "{}"
        assert captured["url"] == "http://localhost:8000/v1/chat/completions"
        assert captured["headers"] == {
            "Content-Type": "application/json",
            "Authorization": "Bearer runtime-key",  # pragma: allowlist secret
        }
        payload = cast(dict[str, object], captured["json"])
        assert payload["model"] == "local-model"
        response_format = cast(dict[str, object], payload["response_format"])
        assert response_format["type"] == "json_schema"
        json_schema = cast(dict[str, object], response_format["json_schema"])
        assert json_schema["strict"] is True

    def test_auth_error_does_not_include_response_body_or_key(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = resolve_inference_settings(
            _local_config(),
            api_key="private-key",  # pragma: allowlist secret
            environ={},
        )
        transport = OpenAICompatibleTransport(settings)
        monkeypatch.setattr(
            httpx,
            "post",
            lambda *args, **kwargs: httpx.Response(401, text="raw-private-response"),
        )

        with pytest.raises(ParameterError) as exc_info:
            transport.complete(
                messages=[{"role": "user", "content": "raw-private-prompt"}],
                response_model=_StructuredResponse,
            )

        rendered = str(exc_info.value)
        assert "raw-private-response" not in rendered
        assert "private-key" not in rendered
