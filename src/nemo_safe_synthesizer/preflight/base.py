# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and contract for writing preflight checks.

Defines the ``PreflightCheck`` ABC and the four stage-specific subclasses
(``ConfigCheck``, ``DataFrameCheck``, ``MetadataCheck``, ``AdvisoryCheck``)
plus the namespace / API-version invariants enforced at class-definition
time.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import ClassVar

from .types import IssueCollector, PreflightContext, PreflightIssue, PreflightStage

__all__ = [
    "AdvisoryCheck",
    "ConfigCheck",
    "DataFrameCheck",
    "MetadataCheck",
    "PreflightCheck",
]


_SUPPORTED_PREFLIGHT_API_VERSIONS: frozenset[int] = frozenset({1})


_CORE_NAMESPACES: frozenset[str] = frozenset(
    {
        "gpu",
        "env",
        "config",
        "columns",
        "timeseries",
        "token_budget",
        "dataset",
        "training",
        "preflight",
    }
)


class PreflightCheck(ABC):
    """Base class for pre-flight validation checks.

    Lifecycle: subclass -> set class attrs (name, label, requires) ->
    instantiate -> register in PREFLIGHT_REGISTRY -> ``_run_registry``
    calls ``run(ctx)`` which delegates to the stage-specific ``check()``
    method.

    Subclasses must not override ``run()`` -- override ``check()`` instead.
    The stage subclass (``ConfigCheck``, ``DataFrameCheck``, etc.) handles
    context destructuring so ``check()`` receives only the args it needs.

    Writing a plugin check:
        - Subclass ``PreflightCheck`` (or one of the stage-specific ABCs).
        - Define ``name``, ``label``, and ``stage`` class attributes. The
          first dotted segment of ``name`` must not match a reserved core
          namespace (see ``_CORE_NAMESPACES``); this is enforced at
          registration time by ``register_preflight_check``.
        - Keep ``__preflight_api_version__`` at a value in
          ``_SUPPORTED_PREFLIGHT_API_VERSIONS`` (currently ``{1}``).
        - Register an instance with ``register_preflight_check(MyCheck())``
          before ``run_preflight`` is invoked.
        - Opt out of a run by adding the check's ``name`` to
          ``config.preflight.disabled_checks``.
        - Uncaught exceptions from ``enabled()``, ``run()``, or
          ``check()`` are reported as a synthetic ``PreflightIssue``
          with code ``preflight.check_crash`` and do not halt the
          remaining registry.
    """

    __preflight_api_version__: ClassVar[int] = 1

    name: ClassVar[str]
    label: ClassVar[str]
    stage: ClassVar[PreflightStage]
    category: ClassVar[str] = "data quality"
    requires: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        for attr in ("name", "label", "stage"):
            if not hasattr(cls, attr):
                raise TypeError(f"{cls.__qualname__} must define {attr!r}")

        api_version = cls.__preflight_api_version__
        if api_version not in _SUPPORTED_PREFLIGHT_API_VERSIONS:
            supported = sorted(_SUPPORTED_PREFLIGHT_API_VERSIONS)
            raise TypeError(
                f"{cls.__qualname__} targets preflight API v{api_version}; this runtime supports {supported}"
            )

    def run(self, ctx: PreflightContext) -> list[PreflightIssue]:
        """Execute this check and return any issues it collected.

        Subclasses implement ``check(ctx, collector)`` instead of
        overriding ``run``. The base implementation wires an
        ``IssueCollector`` for you and returns its accumulated issues.
        """
        collector = IssueCollector(check_name=self.name)
        self.check(ctx, collector)
        return collector.issues

    @abstractmethod
    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
        """Perform the check, appending any findings to ``collector``."""
        raise NotImplementedError

    def enabled(self, ctx: PreflightContext) -> bool:
        """Whether this check should execute for ``ctx``.

        The default implementation honors
        ``ctx.config.preflight.disabled_checks``. Override to add
        declarative skip logic based on config state.
        """
        return self.name not in set(ctx.config.preflight.disabled_checks)


class ConfigCheck(PreflightCheck):
    """Check that only needs the resolved config.

    Stage marker; concrete subclasses implement
    ``check(self, ctx, collector)`` and typically read ``ctx.config``.
    """

    stage = PreflightStage.CONFIG


class DataFrameCheck(PreflightCheck):
    """Check that needs the training DataFrame and config."""

    stage = PreflightStage.DATAFRAME


class MetadataCheck(PreflightCheck):
    """Check that needs data, config, and model metadata."""

    stage = PreflightStage.METADATA


class AdvisoryCheck(PreflightCheck):
    """Advisory data-quality check that needs data and config.

    Uses the ``ADVISORY`` stage. ``_run_registry`` skips the
    ``errored_checks`` bookkeeping for advisory checks, so errors they
    emit are surfaced in the report but never gate downstream checks via
    ``requires``.
    """

    stage = PreflightStage.ADVISORY
