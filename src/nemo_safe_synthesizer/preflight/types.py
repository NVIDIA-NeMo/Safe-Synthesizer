# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data types surfaced by the preflight layer.

This module is intentionally rendering-free: it defines the structured
value objects (``PreflightIssue``, ``PreflightCheckResult``,
``PreflightReport``) and the execution context (``PreflightContext``,
``IssueCollector``, ``PreflightStage``) produced and consumed by
``run_preflight``. All console / Rich formatting lives in
``nemo_safe_synthesizer.tooling.preflight``; use
``render_preflight_report(report, registry=..., ...)`` to display a
report.

Rendering requires the registry (not just the report) because
``PreflightCheckResult`` intentionally does not carry display metadata
(``label``, ``category``, ordering). Those live on the ``PreflightCheck``
classes; the renderer looks them up by check name at render time.
"""

from __future__ import annotations

from collections.abc import Iterator, KeysView, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from ..config.parameters import SafeSynthesizerParameters
    from ..llm.metadata import ModelMetadata
    from .base import PreflightCheck

__all__ = [
    "IssueCollector",
    "PreflightCheckResult",
    "PreflightContext",
    "PreflightIssue",
    "PreflightRegistry",
    "PreflightReport",
    "PreflightStage",
    "PreflightStatus",
]


PreflightStatus = Literal["passed", "failed", "skipped"]


class PreflightStage(Enum):
    CONFIG = "config"
    DATAFRAME = "dataframe"
    METADATA = "metadata"
    ADVISORY = "advisory"


@dataclass(frozen=True)
class PreflightIssue:
    """A single problem discovered by a preflight check.

    A check may emit zero, one, or many issues. Issues are carried on
    the emitting ``PreflightCheckResult``; ``check`` is the back-reference
    to that result's ``name`` (also usable as a join key to the registry
    for display metadata).

    Attributes:
        code: Stable, programmatic identifier (e.g. ``"no_gpu"``,
            ``"schema_exceeds_context"``). Matched by users and tests; the
            full table lives in ``docs/user-guide/troubleshooting.md``.
        severity: ``"error"`` blocks the run, ``"warning"`` is advisory.
        check: Fully-qualified name of the emitting check (e.g.
            ``"gpu.vram"`` or ``"my_plugin.my_check"``). Matches
            ``PreflightCheckResult.name`` and ``PreflightCheck.name``.
        message: Human-readable description rendered in the CLI report.
    """

    code: str
    severity: Literal["error", "warning"]
    check: str
    message: str

    @property
    def namespace(self) -> str | None:
        """Namespace prefix from ``check`` (text before the first ``.``), or ``None`` if the check name has no prefix."""
        return self.check.split(".", 1)[0] if "." in self.check else None


@dataclass(frozen=True)
class PreflightCheckResult:
    """Outcome for a single check the orchestrator considered.

    Carries only what the orchestrator produces: the check ``name`` (join
    key back to the registry), the execution ``status``, and any issues
    the check emitted. Display metadata (``label``, ``category``) lives
    on the ``PreflightCheck`` class and is looked up from the registry
    at render time -- it is not duplicated here.

    Attributes:
        name: Fully-qualified check name (matches ``PreflightCheck.name``
            and ``PreflightIssue.check``).
        status: ``"passed"`` (ran clean -- warnings allowed),
            ``"failed"`` (ran and emitted at least one error), or
            ``"skipped"`` (did not run because a ``requires`` dependency
            errored or was disabled).
        issues: Issues emitted by this check. Empty for ``"passed"``
            or ``"skipped"``.
    """

    name: str
    status: PreflightStatus
    issues: list[PreflightIssue] = field(default_factory=list)


@dataclass(frozen=True)
class PreflightReport:
    """Aggregated pre-flight results for a single ``run_preflight`` call.

    A list of per-check results (one per check the orchestrator
    considered, including clean passes and skips). Rendering requires
    the registry in addition to the report so the renderer can look up
    display metadata by check name; see
    [`render_preflight_report`][nemo_safe_synthesizer.tooling.preflight.render_preflight_report].
    """

    checks: list[PreflightCheckResult]

    @property
    def issues(self) -> list[PreflightIssue]:
        return [i for c in self.checks for i in c.issues]

    @property
    def errors(self) -> list[PreflightIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[PreflightIssue]:
        return [i for i in self.issues if i.severity == "warning"]


@dataclass(frozen=True)
class PreflightRegistry:
    """Ordered, name-keyed view of the checks the orchestrator will run.

    Constructed by [`build_registry`][nemo_safe_synthesizer.preflight.registry.build_registry]
    and treated as immutable. Iteration yields the ``PreflightCheck``
    instances themselves (not their names), so ``for check in registry``
    reads naturally; name-based access uses ``registry[name]`` /
    ``name in registry``. Extend by calling ``register_preflight_check``
    or rebuilding via ``build_registry``; never mutate in place.

    Attributes:
        checks: Insertion-ordered mapping from ``PreflightCheck.name`` to
            the check instance. Typically backed by
            ``types.MappingProxyType`` so the underlying dict is hidden.
    """

    checks: Mapping[str, PreflightCheck]

    def __iter__(self) -> Iterator[PreflightCheck]:
        return iter(self.checks.values())

    def __contains__(self, name: object) -> bool:
        return name in self.checks

    def __getitem__(self, name: str) -> PreflightCheck:
        return self.checks[name]

    def __len__(self) -> int:
        return len(self.checks)

    @property
    def names(self) -> KeysView[str]:
        """The check names, in registry order."""
        return self.checks.keys()


@dataclass(frozen=True)
class PreflightContext:
    """Inputs threaded to every check.

    ``data`` is the training split produced by ``Holdout.train_test_split``
    -- not the full input dataset. On a full run it is also post-PII
    replacement; on ``--validate`` PII replacement is skipped.
    """

    data: pd.DataFrame
    config: SafeSynthesizerParameters
    metadata: ModelMetadata


@dataclass
class IssueCollector:
    check_name: str
    _issues: list[PreflightIssue] = field(default_factory=list)

    def error(self, code: str, message: str) -> None:
        self._issues.append(
            PreflightIssue(
                code=code,
                severity="error",
                check=self.check_name,
                message=message,
            )
        )

    def warning(self, code: str, message: str) -> None:
        self._issues.append(
            PreflightIssue(
                code=code,
                severity="warning",
                check=self.check_name,
                message=message,
            )
        )

    @property
    def issues(self) -> list[PreflightIssue]:
        return self._issues
