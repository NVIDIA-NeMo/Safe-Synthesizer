# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preflight execution entry point.

``run_preflight`` is the single public entry point; ``_run_registry``
handles per-check gating, failure isolation, and result aggregation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..observability import LogCategory, get_logger, traced
from . import registry as _registry
from .base import PreflightCheck
from .registry import PreflightRegistry
from .types import (
    PreflightCheckResult,
    PreflightContext,
    PreflightIssue,
    PreflightReport,
    PreflightStage,
)

if TYPE_CHECKING:
    from ..config.parameters import SafeSynthesizerParameters
    from ..llm.metadata import ModelMetadata

__all__ = ["CRASH_CODE", "run_preflight"]


logger = get_logger(__name__)


CRASH_CODE = "preflight.check_crash"
"""Issue code used when a check raises from ``enabled()`` or ``run()``."""


def _report_crash(check_name: str, site: str, exc: BaseException) -> list[PreflightIssue]:
    """Log an uncaught check exception and return the synthetic crash issue."""
    logger.runtime.debug(
        "Preflight check %r raised from %s; treating as crash.",
        check_name,
        site,
        exc_info=True,
    )
    return [
        PreflightIssue(
            code=CRASH_CODE,
            severity="error",
            check=check_name,
            message=f"{type(exc).__name__}: {exc}",
        )
    ]


def _execute_check(
    check: PreflightCheck,
    ctx: PreflightContext,
) -> list[PreflightIssue] | None:
    """Invoke ``enabled()`` and, if true, ``run()``; convert crashes to issues.

    Returns ``None`` when the check opted out via ``enabled()`` returning
    false. Otherwise returns the issue list (possibly empty, possibly a
    synthetic crash issue).

    Non-``Exception`` ``BaseException`` subclasses (``KeyboardInterrupt``,
    ``SystemExit``, ``MemoryError``, ...) deliberately propagate: a user
    Ctrl-C must abort preflight immediately, and a ``SystemExit`` from
    inside a check should not be silently converted into a report entry.
    The caller (``run_preflight``) will see the same exception it would
    have seen without the orchestrator in the middle.
    """
    try:
        if not check.enabled(ctx):
            return None
    except Exception as exc:
        return _report_crash(check.name, "enabled()", exc)

    try:
        return check.run(ctx)
    except Exception as exc:
        return _report_crash(check.name, "run()", exc)


def _run_registry(
    ctx: PreflightContext,
    registry: PreflightRegistry,
) -> list[PreflightCheckResult]:
    """Execute the registry; return one ``PreflightCheckResult`` per considered check.

    Checks that opt out via ``enabled()`` returning false are absent
    from the returned list (and still gate dependents via the
    ``disabled_checks`` set, matching the prior behavior). Checks that
    ran and errored have ``status="failed"``; checks that ran clean
    have ``status="passed"``; checks gated out by a failed/disabled
    dependency have ``status="skipped"`` with no issues emitted.

    .. note::
        Although private, this function is part of the **test-accessible**
        surface.  Tests that need fine-grained control over the registry or
        context (without going through the full CLI/SDK path) should call
        ``_run_registry`` directly rather than monkey-patching
        ``run_preflight``.  The leading underscore signals "internal to this
        module" rather than "unstable API" -- the signature is stable and
        intentionally exercised by unit tests.
    """
    results: list[PreflightCheckResult] = []
    errored_checks: set[str] = set()
    disabled_checks: set[str] = set()

    for check in registry:
        if any(dep in errored_checks or dep in disabled_checks for dep in check.requires):
            results.append(PreflightCheckResult(name=check.name, status="skipped"))
            errored_checks.add(check.name)  # propagate: skipped checks block their own dependents
            continue

        issues = _execute_check(check, ctx)
        if issues is None:
            disabled_checks.add(check.name)
            continue

        has_error = any(i.severity == "error" for i in issues)
        results.append(
            PreflightCheckResult(
                name=check.name,
                status="failed" if has_error else "passed",
                issues=list(issues),
            )
        )

        # Advisory-stage errors describe data-quality concerns rather than
        # prerequisites for later checks, so they never gate dependents --
        # see ``AdvisoryCheck``.
        if has_error and check.stage is not PreflightStage.ADVISORY:
            errored_checks.add(check.name)

    return results


def _warn_unknown_disabled_checks(
    config: SafeSynthesizerParameters,
    registry: PreflightRegistry,
) -> None:
    """Emit a user-visible warning for unknown names in ``disabled_checks``."""
    unknown = [name for name in config.preflight.disabled_checks if name not in registry]
    if unknown:
        logger.user.warning(
            "Ignoring unknown preflight check name(s) in disabled_checks: %s",
            sorted(unknown),
        )


@traced("preflight", category=LogCategory.USER)
def run_preflight(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    metadata: ModelMetadata,
    *,
    registry: PreflightRegistry | None = None,
) -> PreflightReport:
    """Execute all pre-flight checks against the training split.

    Args:
        data: The training split produced by ``Holdout.train_test_split``.
            On a full run this is also post-PII replacement; on
            ``--validate`` PII replacement is skipped. Row counts, group
            sizes, and column statistics reflect this partition, not the
            original input dataset.
        config: Resolved configuration (``AutoConfigResolver`` already ran).
        metadata: Model metadata (tokenizer and context length).

    Returns:
        A structured ``PreflightReport``.
    """
    effective_registry = _registry.get_registry() if registry is None else registry
    _warn_unknown_disabled_checks(config, effective_registry)

    ctx = PreflightContext(data=data, config=config, metadata=metadata)
    report = PreflightReport(checks=_run_registry(ctx, effective_registry))
    n_checks = len(report.checks)
    n_skipped = sum(1 for c in report.checks if c.status == "skipped")
    n_errors = len(report.errors)
    n_warns = len(report.warnings)
    logger.user.info(
        "Preflight: %d check(s) ran, %d skipped — %d error(s), %d warning(s)",
        n_checks - n_skipped,
        n_skipped,
        n_errors,
        n_warns,
    )
    logger.runtime.debug(
        "Preflight complete",
        extra={
            "errors": len(report.errors),
            "warnings": len(report.warnings),
        },
    )
    return report
