# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed local vLLM server for LLM-assisted PII planning."""

from __future__ import annotations

from .managed import (
    LOCAL_PROFILE_ENV,
    LocalServerRequest,
    planning_inference_environment,
    resolve_local_server_request,
)
from .profile import LocalVllmProfile, bundled_profile_for_model, bundled_profile_names, load_profile
from .server import LocalVllmServer, build_serve_command, is_vllm_installed, local_runtime_problem

__all__ = [
    "LOCAL_PROFILE_ENV",
    "LocalServerRequest",
    "LocalVllmProfile",
    "LocalVllmServer",
    "build_serve_command",
    "bundled_profile_for_model",
    "bundled_profile_names",
    "is_vllm_installed",
    "load_profile",
    "local_runtime_problem",
    "planning_inference_environment",
    "resolve_local_server_request",
]
