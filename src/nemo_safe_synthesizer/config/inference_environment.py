# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``NSS_INFERENCE_*`` runtime settings for LLM-assisted PII planning, read and validated in one place.

These settings never live in a configuration file. SDK users set the environment
variables; the CLI's ``--inference-*`` options validate their values with
``InferenceEnvironment`` and write them to the same variables, so planning reads
them the same way either way. This module imports nothing heavy, so the CLI can
use it at startup.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

from ..errors import ParameterError

__all__ = [
    "ENDPOINT_ENV",
    "INFERENCE_ENV_VARS",
    "KEY_ENV",
    "LOCAL_PROFILE_ENV",
    "MODEL_ENV",
    "REQUEST_OPTIONS_ENV",
    "RESERVED_REQUEST_FIELDS",
    "TIMEOUT_ENV",
    "InferenceEnvironment",
    "InvalidRequestSettingsError",
    "nonblank",
]

ENDPOINT_ENV = "NSS_INFERENCE_ENDPOINT"
"""OpenAI-compatible inference endpoint; when set, planning sends requests there instead of a local server."""
KEY_ENV = "NSS_INFERENCE_KEY"
"""API key for the endpoint."""
MODEL_ENV = "NSS_INFERENCE_MODEL"
"""Model ID when ``replace_pii.llm.model_id`` is unset."""
TIMEOUT_ENV = "NSS_INFERENCE_TIMEOUT"
"""Per-request timeout in seconds."""
REQUEST_OPTIONS_ENV = "NSS_INFERENCE_REQUEST_OPTIONS"
"""Extra chat-completions fields, as a JSON object."""
LOCAL_PROFILE_ENV = "NSS_INFERENCE_LOCAL_PROFILE"
"""Bundled profile name or profile YAML path for the managed local vLLM server."""

INFERENCE_ENV_VARS = (ENDPOINT_ENV, KEY_ENV, MODEL_ENV, TIMEOUT_ENV, REQUEST_OPTIONS_ENV, LOCAL_PROFILE_ENV)
"""Every environment variable ``InferenceEnvironment`` reads."""

RESERVED_REQUEST_FIELDS = frozenset({"messages", "model", "response_format", "stream"})
"""Chat-completions fields NSS sets itself and request options must not override."""


class InvalidRequestSettingsError(ParameterError):
    """``NSS_INFERENCE_TIMEOUT`` or ``NSS_INFERENCE_REQUEST_OPTIONS`` is invalid."""


def nonblank(value: str | None) -> str | None:
    """Return ``value`` stripped of surrounding whitespace, or ``None`` if blank."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True, slots=True)
class InferenceEnvironment:
    """The ``NSS_INFERENCE_*`` settings, with blank values read as unset.

    The endpoint is checked where it is used: an explicit endpoint must use HTTPS
    unless it is a loopback address, while with a local profile it must be the
    loopback address the managed server listens on.

    Example:
        ``InferenceEnvironment.from_environ({"NSS_INFERENCE_TIMEOUT": "120"}).timeout_seconds`` is ``120.0``.
    """

    endpoint: str | None = None
    api_key: str | None = field(default=None, repr=False)
    model_id: str | None = None
    timeout_seconds: float | None = None
    request_options: Mapping[str, object] = field(default_factory=dict)
    """Chat-completions fields; a ``None`` value drops the field from a lower layer, such as a profile."""
    local_profile: str | None = None

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> Self:
        """Read and validate the settings from ``environ``, which defaults to ``os.environ``.

        Raises:
            InvalidRequestSettingsError: If the timeout is not a positive, finite number of
                seconds, or the request options are not a JSON object or set a field NSS manages.
        """
        values = os.environ if environ is None else environ
        return cls(
            endpoint=nonblank(values.get(ENDPOINT_ENV)),
            api_key=nonblank(values.get(KEY_ENV)),
            model_id=nonblank(values.get(MODEL_ENV)),
            timeout_seconds=_timeout_seconds(nonblank(values.get(TIMEOUT_ENV))),
            request_options=_request_options(nonblank(values.get(REQUEST_OPTIONS_ENV))),
            local_profile=nonblank(values.get(LOCAL_PROFILE_ENV)),
        )


def _timeout_seconds(raw: str | None) -> float | None:
    if raw is None:
        return None
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 0.0
    if not timeout > 0 or not math.isfinite(timeout):
        raise InvalidRequestSettingsError(f"{TIMEOUT_ENV} must be a positive number of seconds, got {raw!r}")
    return timeout


def _request_options(raw: str | None) -> dict[str, object]:
    """Return the JSON object in ``raw``, keeping ``null`` values so they can drop a field from a lower layer."""
    if raw is None:
        return {}
    try:
        options = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise InvalidRequestSettingsError(f"{REQUEST_OPTIONS_ENV} must be a JSON object") from exc
    if not isinstance(options, dict):
        raise InvalidRequestSettingsError(f"{REQUEST_OPTIONS_ENV} must be a JSON object")
    if reserved := sorted(RESERVED_REQUEST_FIELDS.intersection(options)):
        raise InvalidRequestSettingsError(
            f"{REQUEST_OPTIONS_ENV} must not set fields NSS manages: {', '.join(reserved)}"
        )
    # JSON object keys are always strings.
    return {str(key): value for key, value in options.items()}
