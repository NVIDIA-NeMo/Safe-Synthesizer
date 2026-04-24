# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Base classes and contract for writing preflight checks.

Overview
--------
This module defines the ``PreflightCheck`` ABC and its four stage-specific
subclasses, plus the ``IssueCollector`` accumulator that check
implementations append findings to.  For plugin authors, this is the
single import for the complete check-authoring surface.

The Generic[C] pattern
----------------------
``PreflightCheck`` is parameterised by a *context view* type ``C``::

    class PreflightCheck(ABC, Generic[C]):
        def check(self, ctx: C, collector: IssueCollector) -> None: ...

Each of the four stage ABCs binds ``C`` to a concrete frozen dataclass
from ``types.py``:

+------------------+--------------------+-------------------------------+
| Stage ABC        | C bound to         | Fields in ctx                 |
+==================+====================+===============================+
| ConfigCheck      | ConfigView         | config                        |
+------------------+--------------------+-------------------------------+
| DataFrameCheck   | DataFrameView      | config, data                  |
+------------------+--------------------+-------------------------------+
| MetadataCheck    | MetadataView       | config, data, metadata        |
+------------------+--------------------+-------------------------------+
| AdvisoryCheck    | DataFrameView      | config, data                  |
+------------------+--------------------+-------------------------------+

The purpose is **type-safety without runtime overhead**: the orchestrator
always builds a single full ``PreflightContext`` and passes it to
``run()``.  ``run()`` calls ``_narrow(ctx)`` -- implemented once per
stage ABC -- which constructs the appropriate view by slicing only the
fields that stage is allowed to touch.  ``check()`` then receives that
narrowed view.  If a ``ConfigCheck`` author accidentally writes
``ctx.data``, the type-checker flags it immediately; at runtime the view
object simply does not have that attribute so an ``AttributeError`` would
surface too.

Why frozen dataclasses instead of Protocols
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Structural ``Protocol`` views were considered and rejected.  A ``Protocol``
would allow any object with the right attributes to satisfy the
constraint, which is useful for tests but hides the "this view was
constructed by narrowing" intent.  Frozen dataclasses enforce that views
are always *produced* by ``_narrow()``, keeping the narrowing path
explicit and auditable.

enabled() vs check()
~~~~~~~~~~~~~~~~~~~~~
``enabled(self, ctx: PreflightContext)`` always receives the full context
because it runs before the stage dispatch -- its job is to decide *whether*
the check should execute based on config, and it may need fields that the
stage's view does not expose.  Only ``check()`` receives the narrowed view.

What plugin authors need to do
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Subclass a stage ABC, set ``name`` / ``label``, implement
``check(self, ctx: <ViewType>, collector)``.  Do **not** override
``run()`` or ``_narrow()``.  The stage ABC handles both.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar, Generic, TypeVar

from .types import ConfigView, DataFrameView, MetadataView, PreflightContext, PreflightIssue, PreflightStage

C = TypeVar("C")

__all__ = [
    "AdvisoryCheck",
    "ConfigCheck",
    "DataFrameCheck",
    "IssueCollector",
    "MetadataCheck",
    "PreflightCheck",
]


@dataclass
class IssueCollector:
    """Mutable accumulator for issues emitted by a single check run.

    Created by ``PreflightCheck.run`` and passed to ``check(ctx, collector)``.
    Plugin authors call ``collector.error`` / ``collector.warning`` inside
    their ``check`` implementation; the orchestrator reads ``collector.issues``
    after the call returns.

    Attributes:
        check_name: Fully-qualified name of the owning check (stamped on
            every issue for traceability back to the registry).
    """

    check_name: str
    _issues: list[PreflightIssue] = field(default_factory=list)

    def error(self, code: str, message: str) -> None:
        """Emit an error-severity issue."""
        self._issues.append(
            PreflightIssue(
                code=code,
                severity="error",
                check=self.check_name,
                message=message,
            )
        )

    def warning(self, code: str, message: str) -> None:
        """Emit a warning-severity issue."""
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
        """All issues accumulated so far, in emission order."""
        return self._issues


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


class PreflightCheck(ABC, Generic[C]):
    """Base class for pre-flight validation checks.

    Lifecycle: subclass -> set class attrs (name, label, requires) ->
    instantiate -> register via ``register_preflight_check`` -> ``_run_registry``
    calls ``run(ctx)`` which narrows the context via ``_narrow()`` then
    delegates to the stage-specific ``check()`` method.

    Subclasses must not override ``run()`` or ``_narrow()`` -- override
    ``check()`` instead. The stage subclass (``ConfigCheck``,
    ``DataFrameCheck``, etc.) binds the generic parameter ``C`` to the
    appropriate view type and implements ``_narrow()`` so that ``check()``
    receives only the fields it is allowed to access.

    Writing a plugin check:
        - Subclass one of the stage-specific ABCs (``ConfigCheck``,
          ``DataFrameCheck``, ``MetadataCheck``, ``AdvisoryCheck``).
        - Define ``name``, ``label``, and ``stage`` class attributes. The
          first dotted segment of ``name`` must not match a reserved core
          namespace (see ``_CORE_NAMESPACES``); this is enforced at
          registration time by ``register_preflight_check``.
        - Keep ``__preflight_api_version__`` at a value in
          ``_SUPPORTED_PREFLIGHT_API_VERSIONS`` (currently ``{1}``).
        - Implement ``check(self, ctx, collector)`` where ``ctx`` is the
          view type for your stage (e.g. ``ConfigView`` for
          ``ConfigCheck``).
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
        overriding ``run``. The base implementation narrows the context
        via ``_narrow()``, wires an ``IssueCollector``, and returns its
        accumulated issues.
        """
        collector = IssueCollector(check_name=self.name)
        self.check(self._narrow(ctx), collector)
        return collector.issues

    @abstractmethod
    def _narrow(self, ctx: PreflightContext) -> C:
        """Slice ``ctx`` down to the fields meaningful for this stage.

        Implemented by each stage ABC (``ConfigCheck``, ``DataFrameCheck``,
        etc.); plugin authors never override this method.
        """
        ...

    @abstractmethod
    def check(self, ctx: C, collector: IssueCollector) -> None:
        """Perform the check, appending any findings to ``collector``.

        ``ctx`` is the stage-specific view produced by ``_narrow()``:

        - ``ConfigCheck`` → ``ConfigView`` (``ctx.config`` only)
        - ``DataFrameCheck`` / ``AdvisoryCheck`` → ``DataFrameView``
          (``ctx.config`` + ``ctx.data``)
        - ``MetadataCheck`` → ``MetadataView`` (all three fields)
        """
        ...

    def enabled(self, ctx: PreflightContext) -> bool:
        """Whether this check should execute for ``ctx``.

        The default implementation honors
        ``ctx.config.preflight.disabled_checks``. Override to add
        declarative skip logic based on config state.

        Note:
            ``enabled()`` always receives the full ``PreflightContext``
            (not a narrowed view) because it runs before the stage
            dispatch and needs access to config regardless of stage.
        """
        return self.name not in ctx.config.preflight.disabled_checks


class ConfigCheck(PreflightCheck[ConfigView]):
    """Check that only needs the resolved config.

    Concrete subclasses implement ``check(self, ctx: ConfigView, collector)``
    and may access only ``ctx.config``. Accessing ``ctx.data`` or
    ``ctx.metadata`` is a type error.
    """

    stage = PreflightStage.CONFIG

    def _narrow(self, ctx: PreflightContext) -> ConfigView:
        return ConfigView(config=ctx.config)


class DataFrameCheck(PreflightCheck[DataFrameView]):
    """Check that needs the training DataFrame and config.

    Concrete subclasses implement ``check(self, ctx: DataFrameView, collector)``
    and may access ``ctx.config`` and ``ctx.data``. Accessing
    ``ctx.metadata`` is a type error.
    """

    stage = PreflightStage.DATAFRAME

    def _narrow(self, ctx: PreflightContext) -> DataFrameView:
        return DataFrameView(config=ctx.config, data=ctx.data)


class MetadataCheck(PreflightCheck[MetadataView]):
    """Check that needs data, config, and model metadata.

    Concrete subclasses implement ``check(self, ctx: MetadataView, collector)``
    and may access all three fields: ``ctx.config``, ``ctx.data``, and
    ``ctx.metadata``.
    """

    stage = PreflightStage.METADATA

    def _narrow(self, ctx: PreflightContext) -> MetadataView:
        return MetadataView(config=ctx.config, data=ctx.data, metadata=ctx.metadata)


class AdvisoryCheck(PreflightCheck[DataFrameView]):
    """Advisory data-quality check that needs data and config.

    Uses the ``ADVISORY`` stage. Concrete subclasses implement
    ``check(self, ctx: DataFrameView, collector)`` and may access
    ``ctx.config`` and ``ctx.data``.

    ``_run_registry`` skips the ``errored_checks`` bookkeeping for
    advisory checks, so errors they emit are surfaced in the report but
    never gate downstream checks via ``requires``.
    """

    stage = PreflightStage.ADVISORY

    def _narrow(self, ctx: PreflightContext) -> DataFrameView:
        return DataFrameView(config=ctx.config, data=ctx.data)
