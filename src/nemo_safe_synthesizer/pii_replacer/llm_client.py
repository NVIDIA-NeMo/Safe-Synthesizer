# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OpenAI-compatible client for PII LLM operations."""

from __future__ import annotations

import ipaddress
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from ..config.replace_pii import LLMConfig
from ..defaults import DEFAULT_NSS_INFERENCE_ENDPOINT, DEFAULT_NSS_INFERENCE_MODEL
from ..errors import GenerationError, ParameterError

__all__ = [
    "InferenceSettings",
    "InvalidInferenceResponse",
    "LLMTransport",
    "MissingInferenceKeyError",
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
    """Chat-completions base URL without a trailing slash."""

    model_id: str
    """Model identifier sent with every request."""

    max_workers: int
    """Maximum concurrent requests."""

    api_key: str | None = field(default=None, repr=False)
    """Bearer credential, or ``None`` for keyless local endpoints; excluded from ``repr``."""


class TransientInferenceError(GenerationError):
    """Retryable inference transport failure without response content.

    Args:
        message: Description of the failure; must not contain response bodies.
        retry_after: Server-requested delay in seconds from a numeric
            ``Retry-After`` header, or ``None`` when absent.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class InvalidInferenceResponse(GenerationError):
    """Retryable malformed inference envelope without response content."""


class MissingInferenceKeyError(ParameterError):
    """The resolved endpoint requires an API key, but none was supplied."""


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


def _is_loopback_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _validate_endpoint(endpoint_url: str) -> None:
    """Reject endpoints that are malformed, embed credentials, or send data unencrypted.

    Plan enhancement sends raw cell samples and possibly a bearer key, so
    plaintext HTTP is accepted only for loopback hosts.
    """
    parsed = urlparse(endpoint_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ParameterError("The PII inference endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ParameterError("The PII inference endpoint URL must not contain credentials")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise ParameterError(
            "The PII inference endpoint must use HTTPS unless it is a loopback address (localhost, 127.0.0.0/8, or ::1)"
        )


def _is_default_hosted_endpoint(endpoint_url: str) -> bool:
    return endpoint_url.rstrip("/") == DEFAULT_NSS_INFERENCE_ENDPOINT.rstrip("/")


def resolve_inference_settings(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> InferenceSettings:
    """Resolve inference settings from persisted configuration, environment, and defaults.

    The model resolves from ``config.model_id``, then ``NSS_INFERENCE_MODEL``,
    then the NSS default. As with other overlapping YAML and environment
    settings, an explicit YAML value wins and the environment only supplies its
    default. The CLI applies an explicit ``--inference-model-id`` to
    ``config.model_id`` before this runs.

    The endpoint and API key are deliberately absent from persisted
    configuration. They come only from ``NSS_INFERENCE_ENDPOINT`` and
    ``NSS_INFERENCE_KEY``, which the CLI populates from its runtime options.

    Args:
        config: Persisted LLM behavior from ``replace_pii.llm``.
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        Settings with a validated endpoint and resolved model.

    Raises:
        ParameterError: If the endpoint is not an absolute HTTP(S) URL, embeds
            credentials, or uses plaintext HTTP for a non-loopback host.
        MissingInferenceKeyError: If the default hosted endpoint is selected
            without an API key.
    """
    runtime_env = os.environ if environ is None else environ
    resolved_endpoint = _nonblank(runtime_env.get("NSS_INFERENCE_ENDPOINT")) or DEFAULT_NSS_INFERENCE_ENDPOINT
    resolved_model = (
        _nonblank(config.model_id) or _nonblank(runtime_env.get("NSS_INFERENCE_MODEL")) or DEFAULT_NSS_INFERENCE_MODEL
    )
    resolved_key = _nonblank(runtime_env.get("NSS_INFERENCE_KEY"))

    _validate_endpoint(resolved_endpoint)
    if _is_default_hosted_endpoint(resolved_endpoint) and resolved_key is None:
        raise MissingInferenceKeyError(
            "NSS_INFERENCE_KEY or --inference-api-key is required for the default hosted NVIDIA inference endpoint"
        )

    return InferenceSettings(
        endpoint_url=resolved_endpoint.rstrip("/"),
        model_id=resolved_model,
        api_key=resolved_key,
        max_workers=config.max_workers,
    )


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Return a numeric ``Retry-After`` delay; HTTP-date values are ignored."""
    try:
        seconds = float(response.headers.get("Retry-After", ""))
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def _without_default(property_schema: object) -> object:
    if isinstance(property_schema, Mapping):
        return {key: value for key, value in property_schema.items() if key != "default"}
    return property_schema


def _strict_schema_value(value: object) -> object:
    if isinstance(value, Mapping):
        # JSON schema keys are always strings.
        return _strict_schema_object({str(key): item for key, item in value.items()})
    if isinstance(value, list):
        return [_strict_schema_value(item) for item in value]
    return value


def _strict_schema_object(node: Mapping[str, object]) -> dict[str, object]:
    """Return a copy of one schema mapping, recursively made strict-mode compatible."""
    strict = {key: _strict_schema_value(value) for key, value in node.items()}
    properties = strict.get("properties")
    if strict.get("type") == "object" and isinstance(properties, Mapping):
        strict["required"] = [str(name) for name in properties]
        strict["properties"] = {str(name): _without_default(schema) for name, schema in properties.items()}
    return strict


def _strict_json_schema(response_model: type[BaseModel]) -> dict[str, object]:
    """Return a JSON schema accepted by OpenAI strict structured outputs.

    Strict mode requires every object property to be listed in ``required``.
    Pydantic omits fields that have defaults, so this marks every property
    required and drops property defaults. Nullable fields keep their ``null``
    branch, so the model can still express an absent value explicitly. The
    model's own schema is left unchanged.
    """
    return _strict_schema_object(response_model.model_json_schema())


class OpenAICompatibleTransport:
    """Minimal privacy-preserving OpenAI chat-completions transport.

    Args:
        settings: Resolved endpoint, model, and credential.
        timeout: Per-request timeout in seconds.

    Raises:
        ParameterError: If the endpoint is malformed, embeds credentials, or
            uses plaintext HTTP for a non-loopback host.
    """

    def __init__(self, settings: InferenceSettings, *, timeout: float = 60.0) -> None:
        # Settings built directly, not through ``resolve_inference_settings``,
        # must not send samples or the bearer key over an unsafe endpoint.
        _validate_endpoint(settings.endpoint_url)
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

        Args:
            messages: Chat messages to send.
            response_model: Pydantic model whose strict JSON schema constrains the response.

        Returns:
            The assistant message content.

        Raises:
            ParameterError: On authentication, authorization, or other permanent request rejection.
            TransientInferenceError: On network failures, timeouts, or retryable HTTP statuses.
            InvalidInferenceResponse: If the response envelope is malformed.
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
                    "schema": _strict_json_schema(response_model),
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
        except httpx.HTTPError as exc:
            raise TransientInferenceError("PII inference transport failed") from exc

        if response.status_code in {401, 403}:
            raise ParameterError(f"PII inference authentication or authorization failed (HTTP {response.status_code})")
        if response.status_code in _TRANSIENT_STATUS_CODES or response.status_code >= 500:
            raise TransientInferenceError(
                f"PII inference service returned HTTP {response.status_code}",
                retry_after=_retry_after_seconds(response),
            )
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
