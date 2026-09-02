# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from threading import Lock
from typing import cast

import httpx
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import (
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
    MAX_ASSESSMENT_PROFILE_BYTES,
    MAX_ASSESSMENT_PROFILES,
    _StructuredResponse,
    _TransientInferenceError,
    _json_bytes,
    _profile_batches,
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


def _assessments(*columns: str) -> str:
    return json.dumps(
        {
            "assessments": [
                {
                    "column_name": column,
                    "disposition": "ignore",
                    "entity_type": None,
                    "pattern": None,
                }
                for column in columns
            ]
        }
    )


def _synthesis(columns_to_replace: list[dict[str, object]]) -> str:
    return json.dumps({"columns_to_replace": columns_to_replace})


def _enhancer(responses: Sequence[str | Exception], *, max_workers: int = 8) -> tuple[LLMPlanEnhancer, ScriptedTransport]:
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
class TestAssessmentBatching:
    def test_batches_obey_count_and_byte_limits(self) -> None:
        dataframe = pd.DataFrame(
            {
                f"column_{index}": [f"{index}-" + "x" * 128]
                for index in range(100)
            }
        )
        captured: list[PlanDiscoveryInput] = []

        class CapturingDiscoverer(PlanDiscoverer):
            def discover(self, discovery_input: PlanDiscoveryInput) -> PiiReplacementPlan:
                captured.append(discovery_input)
                return PiiReplacementPlan(scope=discovery_input.scope)

        resolve_plan(dataframe, ReplacePiiConfig(), DataParameters(), discoverer=CapturingDiscoverer())
        batches = _profile_batches(captured[0].column_profiles)

        assert len(batches) > 1
        assert all(len(batch) <= MAX_ASSESSMENT_PROFILES for batch in batches)
        assert all(_json_bytes(batch) <= MAX_ASSESSMENT_PROFILE_BYTES for batch in batches)


@pytest.mark.unit
class TestLLMPlanEnhancer:
    def test_two_pass_enhancement_can_replace_the_heuristic_baseline(self) -> None:
        dataframe = pd.DataFrame(
            {
                "name": ["Ada Lovelace", "Grace Hopper"],
                "email": ["ada@example.com", "grace@example.com"],
            }
        )
        baseline = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="name", entity_type=EntityType.FULL_NAME)]
        )
        enhancer, transport = _enhancer(
            [
                _assessments("name", "email"),
                _synthesis([{"column_name": "email", "entity_type": "email"}]),
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
        synthesis_payload = transport.calls[1][0][1]["content"]
        assert '"heuristic_baseline"' in synthesis_payload
        assert '"column_name":"name"' in synthesis_payload

    def test_missing_assessment_retries_with_validation_feedback(self) -> None:
        dataframe = pd.DataFrame({"name": ["Ada"], "email": ["ada@example.com"]})
        enhancer, transport = _enhancer(
            [
                _assessments("name"),
                _assessments("name", "email"),
                _synthesis([]),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace == []
        assert len(transport.calls) == 3
        assert "previous structured response was invalid" in transport.calls[1][0][-1]["content"]

    def test_duplicate_assessment_retries(self) -> None:
        dataframe = pd.DataFrame({"name": ["Ada"], "email": ["ada@example.com"]})
        enhancer, transport = _enhancer(
            [
                _assessments("name", "name"),
                _assessments("name", "email"),
                _synthesis([]),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace == []
        assert len(transport.calls) == 3

    @pytest.mark.parametrize(
        "invalid_synthesis",
        [
            _synthesis([{"column_name": "invented", "entity_type": "email"}]),
            _synthesis(
                [
                    {"column_name": "email", "entity_type": "email"},
                    {"column_name": "email", "entity_type": "email"},
                ]
            ),
        ],
        ids=["invented-column", "duplicate-column"],
    )
    def test_invalid_synthesis_is_repaired_on_retry(self, invalid_synthesis: str) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        enhancer, transport = _enhancer(
            [
                _assessments("email"),
                invalid_synthesis,
                _synthesis([]),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace == []
        assert len(transport.calls) == 3

    def test_nested_synthesis_fields_are_strict(self) -> None:
        dataframe = pd.DataFrame({"email": ["ada@example.com"]})
        enhancer, transport = _enhancer(
            [
                _assessments("email"),
                _synthesis(
                    [
                        {
                            "column_name": "email",
                            "entity_type": "email",
                            "unexpected": "must not be ignored",
                        }
                    ]
                ),
                _synthesis([]),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace == []
        assert len(transport.calls) == 3

    def test_malformed_responses_fail_without_exposing_samples(self) -> None:
        dataframe = pd.DataFrame({"secret": ["raw-private-value"]})
        enhancer, _ = _enhancer(['{"assessments":', '{"assessments":', '{"assessments":'])

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
                _assessments("email"),
                _synthesis([]),
            ]
        )

        plan = resolve_plan(
            dataframe,
            ReplacePiiConfig(llm=_local_config()),
            DataParameters(),
            enhancer=enhancer,
        )

        assert plan.columns_to_replace == []
        assert len(transport.calls) == 3

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

    def test_protected_column_response_is_retried_then_fails_without_fallback(self) -> None:
        dataframe = pd.DataFrame({"patient_id": [1, 2], "email": ["a@example.com", "b@example.com"]})
        invalid = _synthesis([{"column_name": "patient_id", "entity_type": "unique_identifier"}])
        baseline = PiiReplacementPlan(
            columns_to_replace=[PiiColumnPlan(column_name="email", entity_type=EntityType.EMAIL)]
        )
        enhancer, transport = _enhancer(
            [
                _assessments("patient_id", "email"),
                invalid,
                invalid,
                invalid,
            ]
        )

        with pytest.raises(GenerationError, match="invalid structured output after 3 attempts"):
            resolve_plan(
                dataframe,
                ReplacePiiConfig(llm=_local_config()),
                DataParameters(group_training_examples_by="patient_id"),
                discoverer=FixedDiscoverer(baseline),
                enhancer=enhancer,
            )

        assert len(transport.calls) == 4

    def test_invalid_pattern_is_repaired(self) -> None:
        dataframe = pd.DataFrame({"phone": ["+1-415-555-0100", "+1-212-555-0199"]})
        enhancer, transport = _enhancer(
            [
                _assessments("phone"),
                _synthesis(
                    [
                        {
                            "column_name": "phone",
                            "entity_type": "phone_number",
                            "pattern": "literal",
                        }
                    ]
                ),
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
        assert len(transport.calls) == 3

    def test_exhausted_invalid_pattern_repairs_drop_only_the_pattern(self) -> None:
        dataframe = pd.DataFrame({"phone": ["+1-415-555-0100", "+1-212-555-0199"]})
        enhancer, transport = _enhancer(
            [
                _assessments("phone"),
                _synthesis(
                    [
                        {
                            "column_name": "phone",
                            "entity_type": "phone_number",
                            "pattern": "literal",
                        }
                    ]
                ),
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
        assert len(transport.calls) == 5


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
