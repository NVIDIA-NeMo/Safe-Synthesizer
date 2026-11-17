# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OpenAI-compatible client for PII LLM operations."""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel

from ..config.inference_environment import (
    ENDPOINT_ENV,
    KEY_ENV,
    MODEL_ENV,
    REQUEST_OPTIONS_ENV,
    RESERVED_REQUEST_FIELDS,
    TIMEOUT_ENV,
    InferenceEnvironment,
    InvalidRequestSettingsError,
    nonblank,
)
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
    "InvalidRequestSettingsError",
    "LLMTransport",
    "MissingInferenceModelError",
    "OpenAICompatibleTransport",
    "TransientInferenceError",
    "configured_model_id",
    "endpoint_inference_settings",
    "is_loopback_host",
    "merge_request_options",
    "nonblank",
    "reserved_request_fields",
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

    timeout_seconds: float = DEFAULT_NSS_INFERENCE_TIMEOUT_SECONDS
    """Per-request timeout; reasoning models on local GPUs can need several minutes."""

    request_options: Mapping[str, object] = field(default_factory=dict)
    """Extra chat-completions fields, such as sampling settings, sent with every request.

    Fields left out, including ``temperature``, use the server's defaults; vLLM
    takes them from the model's ``generation_config.json``.
    """

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
    # The request path is appended to the endpoint, so a query or fragment would swallow it:
    # ``https://host/v1?key=abc`` would be sent as ``https://host/v1?key=abc/chat/completions``.
    if parsed.query or parsed.fragment:
        raise ParameterError(
            "The PII inference endpoint URL must not contain a query string or fragment; "
            "pass an API key through NSS_INFERENCE_KEY or --inference-api-key instead"
        )
    if parsed.scheme == "http" and not is_loopback_host(parsed.hostname):
        raise ParameterError(
            "The PII inference endpoint must use HTTPS unless it is a loopback address (localhost, 127.0.0.0/8, or ::1)"
        )


def merge_request_options(*layers: Mapping[str, object]) -> dict[str, object]:
    """Merge request-option layers field by field, dropping fields set to ``None``.

    Later layers win, and a ``None`` (JSON ``null``) value removes the field,
    leaving it to the server's default.
    """
    merged: dict[str, object] = {}
    for layer in layers:
        merged.update(layer)
    return {field: value for field, value in merged.items() if value is not None}


def reserved_request_fields(fields: Iterable[str]) -> list[str]:
    """Return the sorted fields in ``fields`` that NSS sets itself."""
    return sorted(RESERVED_REQUEST_FIELDS.intersection(fields))


def configured_model_id(config: LLMConfig, environment: InferenceEnvironment) -> str | None:
    """Return ``config.model_id``, else ``NSS_INFERENCE_MODEL``, or ``None`` when neither is set."""
    return nonblank(config.model_id) or environment.model_id


def resolve_inference_settings(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> InferenceSettings:
    """Resolve settings for an explicit inference endpoint from configuration and environment.

    There are no defaults: an endpoint and its model must both be set. The
    model resolves from ``config.model_id``, then ``NSS_INFERENCE_MODEL``. As
    with other overlapping YAML and environment settings, an explicit YAML
    value wins and the environment only supplies its default. The CLI applies
    an explicit ``--inference-model-id`` to ``config.model_id`` before this runs.

    The endpoint and API key are deliberately absent from persisted
    configuration. They come only from ``NSS_INFERENCE_ENDPOINT`` and
    ``NSS_INFERENCE_KEY``, which the CLI populates from its runtime options.
    The per-request timeout comes from ``NSS_INFERENCE_TIMEOUT`` (default 60
    seconds) and extra request fields from ``NSS_INFERENCE_REQUEST_OPTIONS``;
    NSS sends no sampling settings of its own, so fields left out use the
    server's defaults. Plan discovery reaches this through
    ``resolve_planning_inference``, which selects the managed local server
    instead when no endpoint is set.

    Args:
        config: Persisted LLM behavior from ``replace_pii.llm``.
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        Settings with a validated endpoint and resolved model.

    Raises:
        InvalidRequestSettingsError: If the timeout or request options are invalid.
        ParameterError: If no endpoint is set, or the endpoint is not an
            absolute HTTP(S) URL, embeds credentials, or uses plaintext HTTP
            for a non-loopback host.
        MissingInferenceModelError: If the endpoint has no model ID.
    """
    return endpoint_inference_settings(config, InferenceEnvironment.from_environ(environ))


def endpoint_inference_settings(config: LLMConfig, environment: InferenceEnvironment) -> InferenceSettings:
    """Return settings for the explicit endpoint in ``environment``; see ``resolve_inference_settings``."""
    resolved_endpoint = environment.endpoint
    resolved_model = configured_model_id(config, environment)
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
        api_key=environment.api_key,
        max_workers=config.max_workers,
        timeout_seconds=(
            DEFAULT_NSS_INFERENCE_TIMEOUT_SECONDS
            if environment.timeout_seconds is None
            else environment.timeout_seconds
        ),
        request_options=merge_request_options(environment.request_options),
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
        settings: Resolved endpoint, model, credential, timeout, and request options.

    Raises:
        ParameterError: If the endpoint is malformed, embeds credentials, or
            uses plaintext HTTP for a non-loopback host.
    """

    def __init__(self, settings: InferenceSettings) -> None:
        # Settings built directly, not through ``resolve_inference_settings``,
        # must not send samples or the bearer key over an unsafe endpoint.
        _validate_endpoint(settings.endpoint_url)
        self._settings = settings
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
                timeout=self._settings.timeout_seconds,
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
