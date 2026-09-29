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
from ...errors import ParameterError
from ..llm_client import _is_loopback_host, _nonblank
from .profile import LocalVllmProfile, load_profile
from .server import DEFAULT_HOST, LocalVllmServer

__all__ = [
    "LOCAL_PROFILE_ENV",
    "LocalServerRequest",
    "planning_inference_environment",
    "resolve_local_server_request",
]

LOCAL_PROFILE_ENV = "NSS_INFERENCE_LOCAL_PROFILE"
"""Runtime setting naming a bundled profile or profile YAML path."""


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
    parsed = urlparse(endpoint)
    if parsed.scheme != "http" or not _is_loopback_host(parsed.hostname):
        raise ParameterError(
            f"{LOCAL_PROFILE_ENV} starts the PII inference server on this machine, so "
            f"NSS_INFERENCE_ENDPOINT must be unset or {expected_form}"
        )
    try:
        port = parsed.port
    except ValueError:
        port = None
    has_extras = parsed.username or parsed.password or parsed.params or parsed.query or parsed.fragment
    if port is None or parsed.path.rstrip("/") != "/v1" or has_extras:
        raise ParameterError(f"With {LOCAL_PROFILE_ENV}, NSS_INFERENCE_ENDPOINT must be {expected_form}")
    assert parsed.hostname is not None
    return parsed.hostname, port


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


def resolve_local_server_request(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> LocalServerRequest | None:
    """Validate the managed-server settings without launching anything.

    Args:
        config: Persisted LLM behavior from ``replace_pii.llm``.
        environ: Environment mapping to read; defaults to ``os.environ``.

    Returns:
        The launch request, or ``None`` when ``NSS_INFERENCE_LOCAL_PROFILE`` is unset.

    Raises:
        ParameterError: If the profile is invalid, ``NSS_INFERENCE_ENDPOINT``
            is not a usable loopback address, or a configured model name
            differs from the profile's served name.
    """
    runtime_env = os.environ if environ is None else environ
    reference = _nonblank(runtime_env.get(LOCAL_PROFILE_ENV))
    if reference is None:
        return None
    profile = load_profile(reference)
    host, port = _managed_address(_nonblank(runtime_env.get("NSS_INFERENCE_ENDPOINT")))
    _check_model_names(config, runtime_env, profile)
    return LocalServerRequest(profile=profile, host=host, port=port)


@contextmanager
def planning_inference_environment(
    config: LLMConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> Iterator[Mapping[str, str] | None]:
    """Yield the inference environment for one LLM planning pass.

    Without ``NSS_INFERENCE_LOCAL_PROFILE``, this yields ``environ`` unchanged.
    With it, this starts a :class:`LocalVllmServer`, yields an environment
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
