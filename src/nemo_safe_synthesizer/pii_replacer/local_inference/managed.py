# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve where LLM-assisted PII planning runs, and scope a managed local vLLM server to one planning pass."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from urllib.parse import urlparse

from ...config.replace_pii import LLMConfig
from ...defaults import DEFAULT_NSS_INFERENCE_LOCAL_MODEL
from ...errors import ParameterError
from ...observability import get_logger
from ..llm_client import (
    ENDPOINT_ENV,
    KEY_ENV,
    MODEL_ENV,
    InferenceSettings,
    configured_model_id,
    explicit_inference_timeout,
    explicit_request_options,
    is_loopback_host,
    merge_request_options,
    nonblank,
    resolve_inference_settings,
)
from .profile import LocalVllmProfile, bundled_profile_for_model, bundled_profile_names, load_profile
from .server import DEFAULT_HOST, LocalVllmServer

logger = get_logger(__name__)

__all__ = [
    "LOCAL_PROFILE_ENV",
    "InvalidLocalServerSettingsError",
    "LocalServerRequest",
    "planning_inference_settings",
    "resolve_planning_inference",
]

LOCAL_PROFILE_ENV = "NSS_INFERENCE_LOCAL_PROFILE"
"""Runtime setting naming a custom profile YAML path (or a bundled profile name).

When neither this nor ``NSS_INFERENCE_ENDPOINT`` is set, the configured model
selects a bundled profile instead.
"""


class InvalidLocalServerSettingsError(ParameterError):
    """The managed local server's profile, address, or model settings are invalid."""


@dataclass(frozen=True, slots=True)
class LocalServerRequest:
    """Validated settings for one managed local server launch and the planner that calls it."""

    profile: LocalVllmProfile
    """Model and engine settings to serve."""

    host: str
    """Loopback host the server binds."""

    port: int | None
    """Port from ``NSS_INFERENCE_ENDPOINT``, or ``None`` to choose a free port."""

    timeout_seconds: float
    """Per-request planner timeout: ``NSS_INFERENCE_TIMEOUT``, else the profile's ``request_timeout_seconds``."""

    request_options: Mapping[str, object] = field(default_factory=dict)
    """Chat-completions fields: the profile's, overridden field by field by ``NSS_INFERENCE_REQUEST_OPTIONS``."""


def _managed_address(endpoint: str | None) -> tuple[str, int | None]:
    """Return the host and port the managed server listens on.

    An unset endpoint selects a free loopback port. A set endpoint must be a
    plain-HTTP loopback URL with an explicit port and the ``/v1`` path that
    vLLM serves.
    """
    if endpoint is None:
        return DEFAULT_HOST, None
    expected_form = "an http:// loopback URL with a port and /v1 path, such as http://127.0.0.1:8000/v1"
    try:
        parsed = urlparse(endpoint)
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        # Raised for malformed bracketed IPv6 hosts and non-numeric ports.
        raise InvalidLocalServerSettingsError(
            f"With {LOCAL_PROFILE_ENV}, NSS_INFERENCE_ENDPOINT must be {expected_form}"
        ) from exc
    if parsed.scheme != "http" or host is None or not is_loopback_host(host):
        raise InvalidLocalServerSettingsError(
            f"{LOCAL_PROFILE_ENV} starts the PII inference server on this machine, so "
            f"NSS_INFERENCE_ENDPOINT must be unset or {expected_form}"
        )
    has_extras = parsed.username or parsed.password or parsed.params or parsed.query or parsed.fragment
    if port is None or parsed.path.rstrip("/") != "/v1" or has_extras:
        raise InvalidLocalServerSettingsError(
            f"With {LOCAL_PROFILE_ENV}, NSS_INFERENCE_ENDPOINT must be {expected_form}"
        )
    return host, port


def _check_model_names(config: LLMConfig, environ: Mapping[str, str], profile: LocalVllmProfile) -> None:
    served = profile.served_name
    conflicts: list[str] = []
    configured = nonblank(config.model_id)
    if configured is not None and configured != served:
        conflicts.append(f"replace_pii.llm.model_id is {configured!r}")
    env_model = nonblank(environ.get(MODEL_ENV))
    if env_model is not None and env_model != served:
        conflicts.append(f"NSS_INFERENCE_MODEL is {env_model!r}")
    if conflicts:
        raise InvalidLocalServerSettingsError(
            f"The local inference profile serves {served!r}, but {' and '.join(conflicts)}. "
            "Remove the conflicting model setting or set it to the served name."
        )


def _explicit_profile(reference: str) -> LocalVllmProfile:
    try:
        return load_profile(reference)
    except ParameterError as exc:
        raise InvalidLocalServerSettingsError(str(exc)) from exc


def _bundled_profile(config: LLMConfig, environ: Mapping[str, str]) -> LocalVllmProfile:
    """Return the bundled profile for the configured model, or for the default local model."""
    model_id = configured_model_id(config, environ) or DEFAULT_NSS_INFERENCE_LOCAL_MODEL
    profile = bundled_profile_for_model(model_id)
    if profile is None:
        served = ", ".join(sorted(load_profile(name).served_name for name in bundled_profile_names()))
        raise InvalidLocalServerSettingsError(
            f"No inference endpoint is set, so NSS runs {model_id!r} in a local vLLM server, but no bundled "
            f"profile serves it (bundled: {served}). Set NSS_INFERENCE_ENDPOINT to a service that serves the "
            f"model, or set {LOCAL_PROFILE_ENV} to a profile YAML for it."
        )
    return profile


def resolve_planning_inference(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> LocalServerRequest | InferenceSettings:
    """Resolve where LLM-assisted planning runs, without launching anything.

    This is the one place that reads the ``NSS_INFERENCE_*`` settings for
    planning; preflight and plan discovery both call it. An explicit
    ``NSS_INFERENCE_LOCAL_PROFILE`` always selects a managed server, and any
    configured model must match it. Without a profile, an explicit
    ``NSS_INFERENCE_ENDPOINT`` selects that endpoint and no managed server.
    With neither, the configured model (``replace_pii.llm.model_id``, then
    ``NSS_INFERENCE_MODEL``, then ``DEFAULT_NSS_INFERENCE_LOCAL_MODEL``)
    selects the bundled profile to run.

    Args:
        config: Persisted LLM behavior from ``replace_pii.llm``.
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        A ``LocalServerRequest`` for a managed local server, or the
        ``InferenceSettings`` for an explicit endpoint.

    Raises:
        InvalidRequestSettingsError: If ``NSS_INFERENCE_TIMEOUT`` or
            ``NSS_INFERENCE_REQUEST_OPTIONS`` is invalid.
        InvalidLocalServerSettingsError: If the profile is invalid,
            ``NSS_INFERENCE_ENDPOINT`` is not a usable loopback address, a
            configured model differs from an explicit profile, or no bundled
            profile serves the model.
        MissingInferenceModelError: If an explicit endpoint has no model ID.
        ParameterError: If an explicit endpoint is invalid.
    """
    runtime_env = os.environ if environ is None else environ
    reference = nonblank(runtime_env.get(LOCAL_PROFILE_ENV))
    endpoint = nonblank(runtime_env.get(ENDPOINT_ENV))
    if reference is None and endpoint is not None:
        return resolve_inference_settings(config, environ=runtime_env)
    timeout = explicit_inference_timeout(runtime_env)
    explicit_options = explicit_request_options(runtime_env)
    if reference is not None:
        profile = _explicit_profile(reference)
        _check_model_names(config, runtime_env, profile)
    else:
        profile = _bundled_profile(config, runtime_env)
    host, port = _managed_address(endpoint)
    return LocalServerRequest(
        profile=profile,
        host=host,
        port=port,
        timeout_seconds=profile.request_timeout_seconds if timeout is None else timeout,
        request_options=merge_request_options(profile.request_options, explicit_options),
    )


def _log_external_destination(endpoint_url: str) -> None:
    """Tell the user where planning sends column samples when an endpoint is set.

    The endpoint is already validated, so it has no credentials; only its host
    and port are logged.
    """
    parsed = urlparse(endpoint_url)
    if is_loopback_host(parsed.hostname):
        logger.user.info(f"PII planning sends column samples to the local server at {parsed.netloc}")
    else:
        logger.user.info(f"PII planning sends column samples to {parsed.netloc}")


@contextmanager
def planning_inference_settings(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> Iterator[InferenceSettings]:
    """Yield the inference settings for one LLM planning pass.

    For an explicit endpoint (see ``resolve_planning_inference``), this
    yields its settings. Otherwise it starts a ``LocalVllmServer``, yields
    settings that point at that server, and stops the server when the block
    exits. The server therefore never outlives planning, and its GPU memory is
    free before replacement or training.

    Args:
        config: Persisted LLM behavior from ``replace_pii.llm``.
        environ: Environment mapping to read; defaults to ``os.environ``.

    Yields:
        Settings the planner sends its requests with.

    Raises:
        ParameterError: If the inference settings are invalid, or the managed
            server cannot run on this machine.
        GenerationError: If the managed server fails to start.
    """
    runtime_env = os.environ if environ is None else environ
    inference = resolve_planning_inference(config, environ=runtime_env)
    if isinstance(inference, InferenceSettings):
        _log_external_destination(inference.endpoint_url)
        yield inference
        return
    if nonblank(runtime_env.get(KEY_ENV)) is not None:
        logger.user.warning(
            f"{KEY_ENV} is set, but PII planning runs in a local vLLM server, which uses its own key, so it is "
            f"ignored. Set {ENDPOINT_ENV} to send planning requests to a remote service instead."
        )
    logger.user.info(
        f"PII planning runs {inference.profile.served_name!r} in a local vLLM server; "
        "column samples stay on this machine"
    )
    with LocalVllmServer(
        inference.profile,
        host=inference.host,
        port=inference.port,
        environ=runtime_env,
        request_options=inference.request_options,
    ) as server:
        yield server.inference_settings(max_workers=config.max_workers, timeout_seconds=inference.timeout_seconds)
