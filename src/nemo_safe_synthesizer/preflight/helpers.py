# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable helpers for writing ``PreflightCheck`` subclasses.

Translators between common failure modes and ``IssueCollector`` calls.
Every helper takes the issue code as an explicit parameter -- the caller
owns the code-to-failure mapping so each ``collector.error("code", ...)``
sits next to the check that owns it.

Plugin authors import the helpers they want; they don't need to inherit
a mixin or commit to an inheritance contract.

Example:
    from nemo_safe_synthesizer.preflight import helpers

    class MyCheck(DataFrameCheck):
        def check(self, ctx, collector):
            torch = helpers.require_import(
                collector, "torch",
                code="no_torch",
                message="PyTorch is not installed.",
            )
            if torch is None:
                return
            helpers.emit_on_raise(
                collector,
                lambda: some_validator(ctx.data),
                expect=ValueError,
                code="bad_data",
            )
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Literal

from .types import IssueCollector

__all__ = [
    "Severity",
    "emit_on_raise",
    "require_import",
    "resolved_record_count",
]


Severity = Literal["error", "warning"]


def emit_on_raise(
    collector: IssueCollector,
    action: Callable[[], None],
    *,
    expect: type[BaseException] | tuple[type[BaseException], ...],
    code: str,
    severity: Severity = "error",
) -> bool:
    """Run ``action``; translate an expected raise into an issue.

    If ``action`` raises one of ``expect``, emit an issue on ``collector``
    with ``code`` and the exception's message, then return ``False``.
    Unexpected exceptions propagate to the orchestrator's crash handler
    (surfacing as ``preflight.check_crash`` rather than silently swallowed).

    Args:
        collector: Target for the translated issue on expected raises.
        action: Zero-arg callable; any return value is ignored.
        expect: Exception type or non-empty tuple of exception types to
            translate. Passing an empty tuple is rejected because
            ``except ()`` catches nothing and the call degenerates to a
            bare ``action()`` -- almost certainly a programming error.
        code: Issue code to emit on the collector.
        severity: ``"error"`` or ``"warning"``.

    Returns:
        ``True`` if ``action`` completed without raising; ``False`` if it
        raised one of ``expect``.

    Raises:
        ValueError: If ``expect`` is an empty tuple.
    """
    if isinstance(expect, tuple) and len(expect) == 0:
        raise ValueError("emit_on_raise(..., expect=()) catches nothing; pass at least one exception type.")
    try:
        action()
    except expect as exc:
        getattr(collector, severity)(code, str(exc))
        return False
    return True


def require_import(
    collector: IssueCollector,
    module: str,
    *,
    code: str,
    message: str,
    severity: Severity = "error",
) -> ModuleType | None:
    """Import ``module`` or emit an issue and return ``None``.

    Useful for checks that depend on an optional runtime (e.g. ``torch``)
    and want to degrade to a deterministic issue rather than crash when
    the dependency is missing.

    Only ``ImportError`` (and its ``ModuleNotFoundError`` subclass) is
    caught: a module whose top-level code raises some other exception
    (for example, a broken install that raises ``RuntimeError`` during
    import) is a genuine environment fault, so the exception propagates
    to the orchestrator's crash handler and surfaces as
    ``preflight.check_crash`` rather than being silently downgraded to
    a "missing dependency" warning.
    """
    try:
        return importlib.import_module(module)
    except ImportError:
        getattr(collector, severity)(code, message)
        return None


# Re-exported at the bottom so ``preflight.checks`` (which pulls in check
# modules that import ``emit_on_raise``/``require_import`` from this
# module) only gets loaded after those symbols are defined.
from .checks._helpers import resolved_record_count  # noqa: E402  -- see above
