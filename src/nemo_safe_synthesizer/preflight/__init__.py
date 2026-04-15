# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight validation for the training split and resolved config.

See ``docs/developer-guide/preflight-plugins.md`` for the package layout,
plugin-authoring guide, and runtime behavior (dependency gating,
``disabled_checks``, crash isolation, namespace rules).
"""

from __future__ import annotations

from typing import Any

from ..config.preflight import PreflightParameters
from . import helpers
from . import registry as _registry
from .base import AdvisoryCheck, ConfigCheck, DataFrameCheck, MetadataCheck, PreflightCheck
from .checks import (
    ConstantColumnCheck,
    CUDAAvailabilityCheck,
    DatasetRowCountCheck,
    DatasetSizeCheck,
    GroupbyColumnCheck,
    HFTokenCheck,
    InferenceKeyCheck,
    OrderbyColumnCheck,
    OversamplingCheck,
    PseudoColumnCheck,
    TimestampColumnCheck,
    TokenBudgetCheck,
    TrainingStepsCheck,
    UndersamplingCheck,
    VRAMHeadroomCheck,
)
from .orchestrator import CRASH_CODE, run_preflight
from .orchestrator import _run_registry as _run_registry  # re-exported for tests
from .registry import _validate_registry as _validate_registry  # re-exported for tests
from .registry import (
    build_registry,
    register_preflight_check,
    reset_preflight_plugins,
)
from .types import (
    IssueCollector,
    PreflightCheckResult,
    PreflightContext,
    PreflightIssue,
    PreflightRegistry,
    PreflightReport,
    PreflightStage,
)


def __getattr__(name: str) -> Any:
    # ``PREFLIGHT_REGISTRY`` is rebound on the ``registry`` submodule each
    # time a plugin is registered or reset. A plain ``from .registry import
    # PREFLIGHT_REGISTRY`` at the top would freeze the initial binding here,
    # so we resolve it dynamically on attribute access instead.
    if name == "PREFLIGHT_REGISTRY":
        return _registry.PREFLIGHT_REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdvisoryCheck",
    "CRASH_CODE",
    "CUDAAvailabilityCheck",
    "ConfigCheck",
    "ConstantColumnCheck",
    "DataFrameCheck",
    "DatasetRowCountCheck",
    "DatasetSizeCheck",
    "GroupbyColumnCheck",
    "HFTokenCheck",
    "InferenceKeyCheck",
    "IssueCollector",
    "MetadataCheck",
    "OrderbyColumnCheck",
    "OversamplingCheck",
    "PreflightCheck",
    "PreflightCheckResult",
    "PreflightContext",
    "PreflightIssue",
    "PreflightParameters",
    "PreflightRegistry",
    "PreflightReport",
    "PreflightStage",
    "PseudoColumnCheck",
    "TimestampColumnCheck",
    "TokenBudgetCheck",
    "TrainingStepsCheck",
    "UndersamplingCheck",
    "VRAMHeadroomCheck",
    "build_registry",
    "helpers",
    "register_preflight_check",
    "reset_preflight_plugins",
    "run_preflight",
]
