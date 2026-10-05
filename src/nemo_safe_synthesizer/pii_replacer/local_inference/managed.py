# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope a managed local vLLM server to one LLM-assisted planning pass."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from urllib.parse import urlparse

from ...config.replace_pii import LLMConfig
from ...defaults import DEFAULT_NSS_INFERENCE_LOCAL_MODEL
from ...errors import ParameterError
from ..llm_client import _is_loopback_host, _nonblank, resolve_inference_timeout, resolve_request_options
from .profile import LocalVllmProfile, bundled_profile_for_model, bundled_profile_names, load_profile
from .server import DEFAULT_HOST, LocalVllmServer

__all__ = [
    "LOCAL_PROFILE_ENV",
    "LocalServerRequest",
    "planning_inference_environment",
    "resolve_local_server_request",
]

LOCAL_PROFILE_ENV = "NSS_INFERENCE_LOCAL_PROFILE"
"""Runtime setting naming a custom profile YAML path (or a bundled profile name).

When neither this nor ``NSS_INFERENCE_ENDPOINT`` is set, the configured model
selects a bundled profile instead.
"""


@dataclass(frozen=True, slots=True)
class LocalServerRequest:
    """Validated settings for one managed local server launch."""

    profile: LocalVllmProfile
    host: str
    port: int | None
    """Port from ``NSS_INFERENCE_ENDPOINT``, or ``None`` to choose a free port."""


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
        raise ParameterError(f"With {LOCAL_PROFILE_ENV}, NSS_INFERENCE_ENDPOINT must be {expected_form}") from exc
    if parsed.scheme != "http" or host is None or not _is_loopback_host(host):
        raise ParameterError(
            f"{LOCAL_PROFILE_ENV} starts the PII inference server on this machine, so "
            f"NSS_INFERENCE_ENDPOINT must be unset or {expected_form}"
        )
    has_extras = parsed.username or parsed.password or parsed.params or parsed.query or parsed.fragment
    if port is None or parsed.path.rstrip("/") != "/v1" or has_extras:
        raise ParameterError(f"With {LOCAL_PROFILE_ENV}, NSS_INFERENCE_ENDPOINT must be {expected_form}")
    return host, port


def _check_model_names(config: LLMConfig, environ: Mapping[str, str], profile: LocalVllmProfile) -> None:
    served = profile.served_name
    conflicts: list[str] = []
    configured = _nonblank(config.model_id)
    if configured is not None and configured != served:
        conflicts.append(f"replace_pii.llm.model_id is {configured!r}")
    env_model = _nonblank(environ.get("NSS_INFERENCE_MODEL"))
    if env_model is not None and env_model != served:
        conflicts.append(f"NSS_INFERENCE_MODEL is {env_model!r}")
    if conflicts:
        raise ParameterError(
            f"The local inference profile serves {served!r}, but {' and '.join(conflicts)}. "
            "Remove the conflicting model setting or set it to the served name."
        )


def _bundled_profile(config: LLMConfig, environ: Mapping[str, str]) -> LocalVllmProfile:
    """Return the bundled profile for the configured model, or for the default local model."""
    model_id = (
        _nonblank(config.model_id) or _nonblank(environ.get("NSS_INFERENCE_MODEL")) or DEFAULT_NSS_INFERENCE_LOCAL_MODEL
    )
    profile = bundled_profile_for_model(model_id)
    if profile is None:
        served = ", ".join(sorted(load_profile(name).served_name for name in bundled_profile_names()))
        raise ParameterError(
            f"No inference endpoint is set, so NSS runs {model_id!r} in a local vLLM server, but no bundled "
            f"profile serves it (bundled: {served}). Set NSS_INFERENCE_ENDPOINT to a service that serves the "
            f"model, or set {LOCAL_PROFILE_ENV} to a profile YAML for it."
        )
    return profile


def resolve_local_server_request(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> LocalServerRequest | None:
    """Validate the managed-server settings without launching anything.

    An explicit ``NSS_INFERENCE_LOCAL_PROFILE`` always selects a managed
    server, and any configured model must match it. Without a profile, an
    explicit ``NSS_INFERENCE_ENDPOINT`` selects that endpoint and no managed
    server. With neither, the configured model (``replace_pii.llm.model_id``,
    then ``NSS_INFERENCE_MODEL``, then ``DEFAULT_NSS_INFERENCE_LOCAL_MODEL``)
    selects the bundled profile to run.

    Args:
        config: Persisted LLM behavior from ``replace_pii.llm``.
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        The launch request, or ``None`` when an explicit endpoint replaces the managed server.

    Raises:
        ParameterError: If the profile is invalid, ``NSS_INFERENCE_ENDPOINT``
            is not a usable loopback address, ``NSS_INFERENCE_TIMEOUT`` or
            ``NSS_INFERENCE_REQUEST_OPTIONS`` is invalid, a configured model differs from an explicit profile, or no
            bundled profile serves the model.
    """
    runtime_env = os.environ if environ is None else environ
    reference = _nonblank(runtime_env.get(LOCAL_PROFILE_ENV))
    endpoint = _nonblank(runtime_env.get("NSS_INFERENCE_ENDPOINT"))
    if reference is None and endpoint is not None:
        return None
    # Catch invalid runtime settings before an expensive server launch.
    resolve_inference_timeout(runtime_env)
    resolve_request_options(runtime_env)
    if reference is not None:
        profile = load_profile(reference)
        _check_model_names(config, runtime_env, profile)
    else:
        profile = _bundled_profile(config, runtime_env)
    host, port = _managed_address(endpoint)
    return LocalServerRequest(profile=profile, host=host, port=port)


@contextmanager
def planning_inference_environment(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> Iterator[Mapping[str, str] | None]:
    """Yield the inference environment for one LLM planning pass.

    When an explicit ``NSS_INFERENCE_ENDPOINT`` replaces the managed server
    (see :func:`resolve_local_server_request`), this yields ``environ``
    unchanged. Otherwise it starts a :class:`LocalVllmServer`, yields an environment
    whose ``NSS_INFERENCE_*`` values point at that server, and stops the
    server when the block exits. The server therefore never outlives
    planning, and its GPU memory is free before replacement or training.

    Args:
        config: Persisted LLM behavior from ``replace_pii.llm``.
        environ: Environment mapping to read; defaults to ``os.environ``.

    Yields:
        The environment the planner should resolve inference settings from.

    Raises:
        ParameterError: If the managed-server settings are invalid.
        GenerationError: If the managed server fails to start.
    """
    request = resolve_local_server_request(config, environ=environ)
    if request is None:
        yield environ
        return
    with LocalVllmServer(request.profile, host=request.host, port=request.port, environ=environ) as server:
        yield server.inference_environ()
