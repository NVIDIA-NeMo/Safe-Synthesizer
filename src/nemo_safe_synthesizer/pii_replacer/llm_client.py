# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OpenAI-compatible client for PII LLM operations."""

from __future__ import annotations

import ipaddress
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from ..config.replace_pii import LLMConfig
from ..defaults import DEFAULT_NSS_INFERENCE_TIMEOUT_SECONDS
from ..errors import GenerationError, ParameterError

__all__ = [
    "ENDPOINT_ENV",
    "KEY_ENV",
    "MODEL_ENV",
    "REQUEST_OPTIONS_ENV",
    "RESERVED_REQUEST_FIELDS",
    "TIMEOUT_ENV",
    "InferenceSettings",
    "InvalidInferenceResponse",
    "LLMTransport",
    "MissingInferenceModelError",
    "OpenAICompatibleTransport",
    "TransientInferenceError",
    "configured_model_id",
    "is_loopback_host",
    "merge_request_options",
    "nonblank",
    "parse_request_options",
    "reserved_request_fields",
    "resolve_inference_settings",
    "resolve_inference_timeout",
    "resolve_request_options",
]

ENDPOINT_ENV = "NSS_INFERENCE_ENDPOINT"
"""Runtime setting for the OpenAI-compatible inference endpoint."""
KEY_ENV = "NSS_INFERENCE_KEY"
"""Runtime setting for the endpoint's API key."""
MODEL_ENV = "NSS_INFERENCE_MODEL"
"""Runtime setting for the model ID when ``replace_pii.llm.model_id`` is unset."""
TIMEOUT_ENV = "NSS_INFERENCE_TIMEOUT"
"""Runtime setting for the per-request timeout in seconds."""
REQUEST_OPTIONS_ENV = "NSS_INFERENCE_REQUEST_OPTIONS"
"""Runtime setting for extra chat-completions fields, as a JSON object."""

_CHAT_COMPLETIONS_PATH = "/chat/completions"
_DEFAULT_REQUEST_OPTIONS: dict[str, object] = {"temperature": 0}

RESERVED_REQUEST_FIELDS = frozenset({"messages", "model", "response_format", "stream"})
"""Chat-completions fields NSS sets itself and request options must not override."""
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

    timeout_seconds: float = DEFAULT_NSS_INFERENCE_TIMEOUT_SECONDS
    """Per-request timeout; reasoning models on local GPUs can need several minutes."""

    request_options: Mapping[str, object] = field(default_factory=lambda: dict(_DEFAULT_REQUEST_OPTIONS))
    """Extra chat-completions fields, such as sampling settings, sent with every request."""

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


class MissingInferenceModelError(ParameterError):
    """An explicit endpoint is configured, but no model ID for it."""


class LLMTransport(Protocol):
    """Transport capable of requesting a structured LLM response."""

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_model: type[BaseModel],
    ) -> str:
        """Return the assistant response text without logging it."""


def nonblank(value: str | None) -> str | None:
    """Return ``value`` stripped of surrounding whitespace, or ``None`` if blank."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def is_loopback_host(hostname: str | None) -> bool:
    """Return whether ``hostname`` is ``localhost`` or a loopback IP address."""
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
    if parsed.scheme == "http" and not is_loopback_host(parsed.hostname):
        raise ParameterError(
            "The PII inference endpoint must use HTTPS unless it is a loopback address (localhost, 127.0.0.0/8, or ::1)"
        )


def resolve_inference_timeout(environ: Mapping[str, str] | None = None) -> float:
    """Return the per-request timeout from ``NSS_INFERENCE_TIMEOUT``, or the default.

    Raises:
        ParameterError: If the value is not a positive number of seconds.
    """
    runtime_env = os.environ if environ is None else environ
    raw = nonblank(runtime_env.get(TIMEOUT_ENV))
    if raw is None:
        return DEFAULT_NSS_INFERENCE_TIMEOUT_SECONDS
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 0.0
    if not timeout > 0 or timeout == float("inf"):
        raise ParameterError(f"NSS_INFERENCE_TIMEOUT must be a positive number of seconds, got {raw!r}")
    return timeout


def reserved_request_fields(fields: Iterable[str]) -> list[str]:
    """Return the sorted fields in ``fields`` that NSS sets itself."""
    return sorted(RESERVED_REQUEST_FIELDS.intersection(fields))


def parse_request_options(raw: str) -> dict[str, object]:
    """Parse a JSON object of request options from ``NSS_INFERENCE_REQUEST_OPTIONS``.

    Raises:
        ParameterError: If ``raw`` is not a JSON object or sets a field in
            ``RESERVED_REQUEST_FIELDS``.
    """
    try:
        options = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ParameterError(f"{REQUEST_OPTIONS_ENV} must be a JSON object") from exc
    if not isinstance(options, dict):
        raise ParameterError(f"{REQUEST_OPTIONS_ENV} must be a JSON object")
    if reserved := reserved_request_fields(options):
        raise ParameterError(f"{REQUEST_OPTIONS_ENV} must not set fields NSS manages: {', '.join(reserved)}")
    # JSON object keys are always strings.
    return {str(key): value for key, value in options.items()}


def merge_request_options(*layers: Mapping[str, object]) -> dict[str, object]:
    """Merge request-option layers field by field, dropping fields set to ``None``.

    Later layers win, and a ``None`` (JSON ``null``) value removes the field,
    leaving it to the server's default.
    """
    merged: dict[str, object] = {}
    for layer in layers:
        merged.update(layer)
    return {field: value for field, value in merged.items() if value is not None}


def resolve_request_options(environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Return the chat-completions fields sent with every request.

    ``NSS_INFERENCE_REQUEST_OPTIONS`` holds a JSON object, such as
    ``{"temperature": 1.0, "thinking_token_budget": 1000}``, merged field by
    field over the default ``{"temperature": 0}``. Set a field to ``null`` to
    leave it to the server's default.

    Raises:
        ParameterError: If the value is not a JSON object or sets a field in
            ``RESERVED_REQUEST_FIELDS``.
    """
    runtime_env = os.environ if environ is None else environ
    raw = nonblank(runtime_env.get(REQUEST_OPTIONS_ENV))
    overrides = {} if raw is None else parse_request_options(raw)
    return merge_request_options(_DEFAULT_REQUEST_OPTIONS, overrides)


def configured_model_id(config: LLMConfig, environ: Mapping[str, str]) -> str | None:
    """Return ``config.model_id``, else ``NSS_INFERENCE_MODEL``, or ``None`` when neither is set."""
    return nonblank(config.model_id) or nonblank(environ.get(MODEL_ENV))


def resolve_inference_settings(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> InferenceSettings:
    """Resolve inference settings from persisted configuration and environment.

    There are no defaults: an endpoint and its model must both be set. The
    model resolves from ``config.model_id``, then ``NSS_INFERENCE_MODEL``. As
    with other overlapping YAML and environment settings, an explicit YAML
    value wins and the environment only supplies its default. The CLI applies
    an explicit ``--inference-model-id`` to ``config.model_id`` before this runs.

    The endpoint and API key are deliberately absent from persisted
    configuration. They come only from ``NSS_INFERENCE_ENDPOINT`` and
    ``NSS_INFERENCE_KEY``, which the CLI populates from its runtime options.
    The per-request timeout comes from ``NSS_INFERENCE_TIMEOUT`` and extra
    request fields from ``NSS_INFERENCE_REQUEST_OPTIONS``.
    Plan discovery calls this inside ``planning_inference_environment``, which
    points these variables at the managed local server when no endpoint is set.

    Args:
        config: Persisted LLM behavior from ``replace_pii.llm``.
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        Settings with a validated endpoint and resolved model.

    Raises:
        ParameterError: If no endpoint is set, or the endpoint is not an
            absolute HTTP(S) URL, embeds credentials, or uses plaintext HTTP
            for a non-loopback host.
        MissingInferenceModelError: If the endpoint has no model ID.

    See ``resolve_inference_timeout`` and ``resolve_request_options`` for
    their validation errors.
    """
    runtime_env = os.environ if environ is None else environ
    resolved_endpoint = nonblank(runtime_env.get(ENDPOINT_ENV))
    resolved_model = configured_model_id(config, runtime_env)
    resolved_key = nonblank(runtime_env.get(KEY_ENV))

    if resolved_endpoint is None:
        raise ParameterError("No PII inference endpoint is set; set NSS_INFERENCE_ENDPOINT or --inference-endpoint-url")
    _validate_endpoint(resolved_endpoint)
    if resolved_model is None:
        raise MissingInferenceModelError(
            "NSS_INFERENCE_ENDPOINT is set, so the model it serves must be set too: use "
            "replace_pii.llm.model_id, --inference-model-id, or NSS_INFERENCE_MODEL"
        )

    return InferenceSettings(
        endpoint_url=resolved_endpoint.rstrip("/"),
        model_id=resolved_model,
        api_key=resolved_key,
        max_workers=config.max_workers,
        timeout_seconds=resolve_inference_timeout(runtime_env),
        request_options=resolve_request_options(runtime_env),
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
        timeout: Per-request timeout in seconds; defaults to ``settings.timeout_seconds``.

    Raises:
        ParameterError: If the endpoint is malformed, embeds credentials, or
            uses plaintext HTTP for a non-loopback host.
    """

    def __init__(self, settings: InferenceSettings, *, timeout: float | None = None) -> None:
        # Settings built directly, not through ``resolve_inference_settings``,
        # must not send samples or the bearer key over an unsafe endpoint.
        _validate_endpoint(settings.endpoint_url)
        self._settings = settings
        self._timeout = settings.timeout_seconds if timeout is None else timeout
        # Loopback servers never need a proxy, and an environment proxy would
        # receive the bearer key and raw cell samples. Remote endpoints keep
        # the environment's proxy settings.
        self._trust_env = not is_loopback_host(urlparse(settings.endpoint_url).hostname)

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
            **self._settings.request_options,
        }
        try:
            response = httpx.post(
                self._settings.endpoint_url + _CHAT_COMPLETIONS_PATH,
                headers=headers,
                json=payload,
                timeout=self._timeout,
                trust_env=self._trust_env,
            )
        except httpx.HTTPError as exc:
            raise TransientInferenceError("PII inference transport failed") from exc

        if response.status_code in {401, 403}:
            raise ParameterError(
                f"PII inference authentication or authorization failed (HTTP {response.status_code}); "
                "check NSS_INFERENCE_KEY or --inference-api-key"
            )
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
