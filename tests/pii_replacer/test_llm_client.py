# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from nemo_safe_synthesizer.config.replace_pii import LLMConfig
from nemo_safe_synthesizer.defaults import DEFAULT_NSS_INFERENCE_ENDPOINT, DEFAULT_NSS_INFERENCE_MODEL
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.llm_client import (
    InferenceSettings,
    InvalidInferenceResponse,
    MissingInferenceKeyError,
    OpenAICompatibleTransport,
    TransientInferenceError,
    resolve_inference_settings,
)

LOCAL_ENDPOINT = "http://localhost:8000/v1"


class _StructuredResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _OptionalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    pattern: str | None = None


class _OptionalFieldsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[_OptionalItem]
    note: str | None = Field(default=None)


def _local_settings(*, api_key: str | None = None) -> InferenceSettings:
    environ = {"NSS_INFERENCE_ENDPOINT": LOCAL_ENDPOINT}
    if api_key is not None:
        environ["NSS_INFERENCE_KEY"] = api_key
    return resolve_inference_settings(LLMConfig(model_id="local-model"), environ=environ)


def _as_dict(value: object) -> dict[str, object]:
    return cast(dict[str, object], value)


def _capture_request(monkeypatch: pytest.MonkeyPatch, response_model: type[BaseModel]) -> dict[str, object]:
    captured: dict[str, object] = {}

    def post(url: str, **kwargs: object) -> httpx.Response:
        captured["url"] = url
        captured.update(kwargs)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    monkeypatch.setattr(httpx, "post", post)
    content = OpenAICompatibleTransport(_local_settings(api_key="runtime-key")).complete(  # pragma: allowlist secret
        messages=[{"role": "user", "content": "request"}],
        response_model=response_model,
    )
    assert content == "{}"
    return captured


@pytest.mark.unit
class TestInferenceSettings:
    @pytest.mark.parametrize(
        ("config_model", "environ", "expected_model"),
        [
            pytest.param("config-model", {"NSS_INFERENCE_MODEL": "env-model"}, "config-model", id="yaml-before-env"),
            pytest.param(None, {"NSS_INFERENCE_MODEL": "env-model"}, "env-model", id="env-before-default"),
            pytest.param(None, {"NSS_INFERENCE_MODEL": "  "}, DEFAULT_NSS_INFERENCE_MODEL, id="blank-env-ignored"),
            pytest.param(None, {}, DEFAULT_NSS_INFERENCE_MODEL, id="default"),
        ],
    )
    def test_model_precedence(
        self,
        config_model: str | None,
        environ: dict[str, str],
        expected_model: str,
    ) -> None:
        settings = resolve_inference_settings(
            LLMConfig(model_id=config_model),
            environ={"NSS_INFERENCE_ENDPOINT": LOCAL_ENDPOINT, **environ},
        )

        assert settings.endpoint_url == LOCAL_ENDPOINT
        assert settings.model_id == expected_model

    def test_defaults_use_hosted_nvidia_service(self) -> None:
        settings = resolve_inference_settings(
            LLMConfig(),
            environ={"NSS_INFERENCE_KEY": "hosted-key"},  # pragma: allowlist secret
        )

        assert settings.endpoint_url == DEFAULT_NSS_INFERENCE_ENDPOINT
        assert settings.model_id == DEFAULT_NSS_INFERENCE_MODEL

    def test_default_hosted_endpoint_requires_runtime_key(self) -> None:
        with pytest.raises(MissingInferenceKeyError, match="NSS_INFERENCE_KEY"):
            resolve_inference_settings(LLMConfig(), environ={})

    def test_local_openai_compatible_endpoint_can_be_keyless(self) -> None:
        assert _local_settings().api_key is None

    def test_api_key_is_redacted_from_repr(self) -> None:
        settings = _local_settings(api_key="do-not-render")  # pragma: allowlist secret

        assert "do-not-render" not in repr(settings)

    @pytest.mark.parametrize(
        "endpoint",
        [
            "http://localhost:8000/v1",
            "http://LOCALHOST:8000/v1",
            "http://127.0.0.1:8000/v1",
            "http://127.1.2.3/v1",
            "http://[::1]:8000/v1",
            "https://inference.example.com/v1",
        ],
    )
    def test_accepts_https_or_loopback_http(self, endpoint: str) -> None:
        settings = resolve_inference_settings(LLMConfig(), environ={"NSS_INFERENCE_ENDPOINT": endpoint})

        assert settings.endpoint_url == endpoint

    @pytest.mark.parametrize(
        ("endpoint", "message"),
        [
            pytest.param("localhost:8000", "absolute HTTP", id="missing-scheme"),
            pytest.param("ftp://localhost/v1", "absolute HTTP", id="unsupported-scheme"),
            pytest.param(
                "https://user:password@example.com/v1",  # pragma: allowlist secret
                "must not contain credentials",
                id="embedded-credentials",
            ),
            pytest.param("http://inference.example.com/v1", "must use HTTPS", id="remote-http"),
            pytest.param("http://10.0.0.5:8000/v1", "must use HTTPS", id="private-network-http"),
            pytest.param("http://localhost.example.com/v1", "must use HTTPS", id="localhost-lookalike"),
        ],
    )
    def test_rejects_invalid_or_plaintext_remote_endpoints(self, endpoint: str, message: str) -> None:
        with pytest.raises(ParameterError, match=message) as exc_info:
            resolve_inference_settings(LLMConfig(), environ={"NSS_INFERENCE_ENDPOINT": endpoint})

        assert not isinstance(exc_info.value, MissingInferenceKeyError)

    def test_retryable_errors_belong_to_the_generation_error_hierarchy(self) -> None:
        assert issubclass(TransientInferenceError, GenerationError)
        assert issubclass(InvalidInferenceResponse, GenerationError)
        assert issubclass(MissingInferenceKeyError, ParameterError)


@pytest.mark.unit
class TestOpenAICompatibleTransport:
    def test_rejects_plaintext_remote_endpoint_from_direct_settings(self) -> None:
        settings = InferenceSettings(
            endpoint_url="http://inference.example.com/v1",
            model_id="remote-model",
            max_workers=1,
            api_key="private-key",  # pragma: allowlist secret
        )

        with pytest.raises(ParameterError, match="must use HTTPS"):
            OpenAICompatibleTransport(settings)

    def test_sends_chat_completions_with_strict_json_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture_request(monkeypatch, _StructuredResponse)

        assert captured["url"] == "http://localhost:8000/v1/chat/completions"
        assert captured["headers"] == {
            "Content-Type": "application/json",
            "Authorization": "Bearer runtime-key",  # pragma: allowlist secret
        }
        payload = _as_dict(captured["json"])
        assert payload["model"] == "local-model"
        response_format = _as_dict(payload["response_format"])
        assert response_format["type"] == "json_schema"
        assert _as_dict(response_format["json_schema"])["strict"] is True

    def test_strict_schema_requires_every_property_without_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = _capture_request(monkeypatch, _OptionalFieldsResponse)

        response_format = _as_dict(_as_dict(captured["json"])["response_format"])
        schema = _as_dict(_as_dict(response_format["json_schema"])["schema"])
        item_schema = _as_dict(_as_dict(schema["$defs"])["_OptionalItem"])
        note_schema = _as_dict(_as_dict(schema["properties"])["note"])
        pattern_schema = _as_dict(_as_dict(item_schema["properties"])["pattern"])
        assert schema["required"] == ["items", "note"]
        assert item_schema["required"] == ["name", "pattern"]
        assert "default" not in note_schema
        assert "default" not in pattern_schema
        assert {"type": "null"} in cast(list[object], pattern_schema["anyOf"])

    def test_auth_error_does_not_include_response_body_or_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = OpenAICompatibleTransport(_local_settings(api_key="private-key"))  # pragma: allowlist secret
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

    @pytest.mark.parametrize(
        ("headers", "expected_retry_after"),
        [({"Retry-After": "5"}, 5.0), ({"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, None), ({}, None)],
    )
    def test_transient_status_reports_numeric_retry_after(
        self,
        monkeypatch: pytest.MonkeyPatch,
        headers: dict[str, str],
        expected_retry_after: float | None,
    ) -> None:
        monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: httpx.Response(429, headers=headers))

        with pytest.raises(TransientInferenceError) as exc_info:
            OpenAICompatibleTransport(_local_settings()).complete(
                messages=[{"role": "user", "content": "request"}],
                response_model=_StructuredResponse,
            )

        assert exc_info.value.retry_after == expected_retry_after
