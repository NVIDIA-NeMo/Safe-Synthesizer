# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from nemo_safe_synthesizer.config.replace_pii import LLMConfig
from nemo_safe_synthesizer.defaults import DEFAULT_NSS_INFERENCE_ENDPOINT, DEFAULT_NSS_INFERENCE_MODEL
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.llm_client import (
    OpenAICompatibleTransport,
    resolve_inference_settings,
)


class _StructuredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _local_config() -> LLMConfig:
    return LLMConfig(model_id="local-model")


@pytest.mark.unit
class TestInferenceSettings:
    def test_explicit_overrides_precede_persisted_model_and_environment(self) -> None:
        settings = resolve_inference_settings(
            LLMConfig(model_id="config-model"),
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

    def test_environment_model_precedes_persisted_model(self) -> None:
        settings = resolve_inference_settings(
            LLMConfig(model_id="config-model"),
            environ={
                "NSS_INFERENCE_ENDPOINT": "https://env.example/v1",
                "NSS_INFERENCE_MODEL": "env-model",
            },
        )

        assert settings.endpoint_url == "https://env.example/v1"
        assert settings.model_id == "env-model"

    def test_persisted_model_precedes_default(self) -> None:
        settings = resolve_inference_settings(
            LLMConfig(model_id="config-model"),
            environ={"NSS_INFERENCE_ENDPOINT": "http://localhost:8080/v1"},
        )

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
        settings = resolve_inference_settings(
            _local_config(),
            endpoint_url="http://localhost:8000/v1",
            environ={},
        )

        assert settings.api_key is None

    def test_api_key_is_redacted_from_repr(self) -> None:
        settings = resolve_inference_settings(
            _local_config(),
            endpoint_url="http://localhost:8000/v1",
            api_key="do-not-render",  # pragma: allowlist secret
            environ={},
        )

        assert "do-not-render" not in repr(settings)

    def test_invalid_endpoint_fails_without_transport(self) -> None:
        with pytest.raises(ParameterError, match="absolute HTTP"):
            resolve_inference_settings(LLMConfig(), endpoint_url="localhost:8000", environ={})

    def test_endpoint_rejects_embedded_credentials(self) -> None:
        with pytest.raises(ParameterError, match="must not contain credentials"):
            resolve_inference_settings(
                LLMConfig(),
                endpoint_url="https://user:password@example.com/v1",  # pragma: allowlist secret
                environ={},
            )


@pytest.mark.unit
class TestOpenAICompatibleTransport:
    def test_sends_chat_completions_with_strict_json_schema(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = resolve_inference_settings(
            _local_config(),
            endpoint_url="http://localhost:8000/v1",
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
            endpoint_url="http://localhost:8000/v1",
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
