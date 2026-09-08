# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OpenAI-compatible client for PII LLM operations."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from ..config.replace_pii import LLMConfig
from ..defaults import DEFAULT_NSS_INFERENCE_ENDPOINT, DEFAULT_NSS_INFERENCE_MODEL
from ..errors import ParameterError

__all__ = [
    "InferenceSettings",
    "InvalidInferenceResponse",
    "LLMTransport",
    "OpenAICompatibleTransport",
    "TransientInferenceError",
    "resolve_inference_settings",
]

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429})


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    """Resolved runtime settings for the OpenAI-compatible adapter."""

    endpoint_url: str
    model_id: str
    max_workers: int
    api_key: str | None = field(default=None, repr=False)


class TransientInferenceError(RuntimeError):
    """Retryable inference transport failure without response content."""


class InvalidInferenceResponse(RuntimeError):
    """Retryable malformed inference envelope without response content."""


class LLMTransport(Protocol):
    """Transport capable of requesting a structured LLM response."""

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_model: type[BaseModel],
    ) -> str:
        """Return the assistant response text without logging it."""


def _nonblank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _validate_endpoint(endpoint_url: str) -> None:
    parsed = urlparse(endpoint_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ParameterError("The PII inference endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ParameterError("The PII inference endpoint URL must not contain credentials")


def _is_default_hosted_endpoint(endpoint_url: str) -> bool:
    return endpoint_url.rstrip("/") == DEFAULT_NSS_INFERENCE_ENDPOINT.rstrip("/")


def resolve_inference_settings(
    config: LLMConfig,
    *,
    endpoint_url: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> InferenceSettings:
    """Resolve runtime overrides, environment, persisted model settings, and defaults.

    The endpoint and API key are deliberately absent from persisted
    configuration. They are accepted only as runtime overrides or through the
    corresponding ``NSS_INFERENCE_*`` environment variables.
    """
    runtime_env = os.environ if environ is None else environ
    resolved_endpoint = (
        _nonblank(endpoint_url)
        or _nonblank(runtime_env.get("NSS_INFERENCE_ENDPOINT"))
        or DEFAULT_NSS_INFERENCE_ENDPOINT
    )
    resolved_model = (
        _nonblank(model_id)
        or _nonblank(runtime_env.get("NSS_INFERENCE_MODEL"))
        or _nonblank(config.model_id)
        or DEFAULT_NSS_INFERENCE_MODEL
    )
    resolved_key = _nonblank(api_key) or _nonblank(runtime_env.get("NSS_INFERENCE_KEY"))

    _validate_endpoint(resolved_endpoint)
    if _is_default_hosted_endpoint(resolved_endpoint) and resolved_key is None:
        raise ParameterError(
            "NSS_INFERENCE_KEY or --inference-api-key is required for the default hosted NVIDIA inference endpoint"
        )

    return InferenceSettings(
        endpoint_url=resolved_endpoint.rstrip("/"),
        model_id=resolved_model,
        api_key=resolved_key,
        max_workers=config.max_workers,
    )


class OpenAICompatibleTransport:
    """Minimal privacy-preserving OpenAI chat-completions transport."""

    def __init__(self, settings: InferenceSettings, *, timeout: float = 60.0) -> None:
        self._settings = settings
        self._timeout = timeout

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_model: type[BaseModel],
    ) -> str:
        """Return one structured assistant response.

        Errors intentionally exclude response bodies because they may contain
        prompts, samples, or model-authored sensitive text.
        """
        headers = {"Content-Type": "application/json"}
        if self._settings.api_key is not None:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        payload = {
            "model": self._settings.model_id,
            "messages": list(messages),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
            "temperature": 0,
        }
        try:
            response = httpx.post(
                self._settings.endpoint_url + _CHAT_COMPLETIONS_PATH,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            raise TransientInferenceError("PII inference transport failed") from exc
        except httpx.HTTPError as exc:
            raise TransientInferenceError("PII inference transport failed") from exc

        if response.status_code in {401, 403}:
            raise ParameterError(f"PII inference authentication or authorization failed (HTTP {response.status_code})")
        if response.status_code in _TRANSIENT_STATUS_CODES or response.status_code >= 500:
            raise TransientInferenceError(f"PII inference service returned HTTP {response.status_code}")
        if response.status_code >= 400:
            raise ParameterError(f"PII inference request was rejected (HTTP {response.status_code})")

        try:
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise InvalidInferenceResponse("PII inference service returned an invalid response envelope") from exc
        if not isinstance(content, str):
            raise InvalidInferenceResponse("PII inference service returned non-text response content")
        return content
