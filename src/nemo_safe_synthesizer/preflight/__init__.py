# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight validation for the training split and resolved config.

See ``docs/developer-guide/preflight-plugins.md`` for the package layout,
plugin-authoring guide, and runtime behavior (dependency gating,
``disabled_checks``, crash isolation, namespace rules).
"""

from __future__ import annotations

from ..config.preflight import PreflightParameters
from . import helpers
from .base import AdvisoryCheck, ConfigCheck, DataFrameCheck, IssueCollector, MetadataCheck, PreflightCheck
from .checks import (
    ConstantColumnCheck,
    CUDAAvailabilityCheck,
    DatasetSizeCheck,
    GroupbyColumnCheck,
    HFModelCacheCheck,
    HFTokenCheck,
    InferenceKeyCheck,
    OrderbyColumnCheck,
    OversamplingCheck,
    PseudoColumnCheck,
    SmallDatasetCheck,
    TimestampColumnCheck,
    TokenBudgetCheck,
    VRAMHeadroomCheck,
)
from .orchestrator import CRASH_CODE, run_preflight
from .registry import (
    PreflightRegistry,
    build_registry,
    get_registry,
    register_preflight_check,
    reset_preflight_plugins,
)
from .types import (
    ConfigView,
    DataFrameView,
    MetadataView,
    PreflightCheckResult,
    PreflightContext,
    PreflightIssue,
    PreflightReport,
    PreflightStage,
    PreflightStatus,
)

__all__ = [
    "AdvisoryCheck",
    "CRASH_CODE",
    "CUDAAvailabilityCheck",
    "ConfigCheck",
    "ConfigView",
    "DataFrameView",
    "ConstantColumnCheck",
    "DataFrameCheck",
    "SmallDatasetCheck",
    "DatasetSizeCheck",
    "GroupbyColumnCheck",
    "HFModelCacheCheck",
    "HFTokenCheck",
    "InferenceKeyCheck",
    "IssueCollector",
    "MetadataCheck",
    "MetadataView",
    "OrderbyColumnCheck",
    "OversamplingCheck",
    "PreflightCheck",
    "PreflightCheckResult",
    "PreflightContext",
    "PreflightIssue",
    "PreflightParameters",
    "PreflightRegistry",
    "PreflightReport",
    "PreflightStatus",
    "PreflightStage",
    "PseudoColumnCheck",
    "TimestampColumnCheck",
    "TokenBudgetCheck",
    "VRAMHeadroomCheck",
    "build_registry",
    "get_registry",
    "helpers",
    "register_preflight_check",
    "reset_preflight_plugins",
    "run_preflight",
]
