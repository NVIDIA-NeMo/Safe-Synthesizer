# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable value objects surfaced by the preflight layer.

This module is intentionally rendering-free.  It contains two groups of
types, described below.

Value objects (produced by the orchestrator, consumed by the renderer)
----------------------------------------------------------------------
``PreflightIssue``, ``PreflightCheckResult``, ``PreflightReport`` carry
the outcome of a ``run_preflight`` call.  All are frozen dataclasses.
``PreflightStatus`` is a ``Literal`` alias used in ``PreflightCheckResult``.

``PreflightContext`` is the *full* input bundle threaded through the
orchestrator: the training DataFrame, the resolved config, and the model
metadata.  The orchestrator constructs exactly one per ``run_preflight``
call and passes it to each ``PreflightCheck.run()``.

Rendering requires the registry (not just the report) because
``PreflightCheckResult`` intentionally does not carry display metadata
(``label``, ``category``, ordering).  Those live on ``PreflightCheck``
and are looked up by name at render time; see
``nemo_safe_synthesizer.tooling.preflight.render_preflight_report``.

Stage-specific context views (consumed by check implementations)
----------------------------------------------------------------
``ConfigView``, ``DataFrameView``, and ``MetadataView`` are frozen
dataclasses that each expose only the subset of ``PreflightContext``
that a given stage is conceptually allowed to access:

    PreflightContext              all three fields (orchestrator-internal)
    ├── ConfigView                config only
    ├── DataFrameView             config + data      (DataFrameCheck, AdvisoryCheck)
    └── MetadataView              config + data + metadata

These views exist to give check implementations a precise type
annotation and to let the type-checker enforce stage boundaries.  If a
``ConfigCheck`` author writes ``ctx.data``, the type-checker flags it
because ``ConfigView`` has no ``data`` attribute.

The views are *not* subtypes of ``PreflightContext`` -- they are
independent frozen dataclasses produced by the stage ABCs' ``_narrow()``
methods.  At runtime ``_narrow()`` simply copies the relevant fields
from the full ``PreflightContext``.  There is no inheritance relationship
and no Protocol matching; the narrowing is explicit and auditable.

``IssueCollector`` (the mutable accumulator used inside ``check()``)
lives in ``base.py`` alongside the ``PreflightCheck`` ABC so plugin
authors need only one import for the full check-authoring surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    from ..config.parameters import SafeSynthesizerParameters
    from ..llm.metadata import ModelMetadata

__all__ = [
    "ConfigView",
    "DataFrameView",
    "MetadataView",
    "PreflightCheckResult",
    "PreflightContext",
    "PreflightIssue",
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
class PreflightContext:
    """Inputs threaded to every check.

    ``data`` is the training split produced by ``Holdout.train_test_split``
    -- not the full input dataset. On a full run it is also post-PII
    replacement; on ``--validate`` PII replacement is skipped.

    The orchestrator builds one ``PreflightContext`` per ``run_preflight``
    call and passes it to ``PreflightCheck.run()``, which narrows it to a
    stage-specific view (``ConfigView``, ``DataFrameView``, etc.) before
    invoking ``check()``. Check implementations should type their ``ctx``
    parameter as the appropriate view, not as ``PreflightContext``.
    """

    data: pd.DataFrame
    config: SafeSynthesizerParameters
    metadata: ModelMetadata


@dataclass(frozen=True)
class ConfigView:
    """Narrowed context passed to :class:`~nemo_safe_synthesizer.preflight.base.ConfigCheck` implementations.

    Contains only the resolved config. Accessing ``data`` or ``metadata``
    inside a ``ConfigCheck.check()`` is a type error -- use
    :class:`DataFrameView` or :class:`MetadataView` if you need those fields.
    """

    config: SafeSynthesizerParameters


@dataclass(frozen=True)
class DataFrameView:
    """Narrowed context passed to :class:`~nemo_safe_synthesizer.preflight.base.DataFrameCheck` and
    :class:`~nemo_safe_synthesizer.preflight.base.AdvisoryCheck` implementations.

    Contains the resolved config and the training DataFrame. Accessing
    ``metadata`` is a type error -- use :class:`MetadataView` if you
    need model metadata.
    """

    config: SafeSynthesizerParameters
    data: pd.DataFrame


@dataclass(frozen=True)
class MetadataView:
    """Narrowed context passed to :class:`~nemo_safe_synthesizer.preflight.base.MetadataCheck` implementations.

    Contains all three fields: resolved config, training DataFrame, and
    model metadata. This is the widest view; prefer a narrower view
    (``ConfigView`` / ``DataFrameView``) if you do not need all three.
    """

    config: SafeSynthesizerParameters
    data: pd.DataFrame
    metadata: ModelMetadata
