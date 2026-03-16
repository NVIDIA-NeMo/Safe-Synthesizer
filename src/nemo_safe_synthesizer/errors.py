# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Error hierarchy for Safe Synthesizer.

All public exceptions inherit from ``SafeSynthesizerError``. Errors on
the user side (bad data, bad config, generation failure) inherit from
``UserError`` and a matching built-in (``ValueError``, ``RuntimeError``)
so callers can catch either.

Classes:
    SafeSynthesizerError: Base for all known errors.
    UserError: Invalid usage (bad inputs, uninitialized state).
    InternalError: Library bug (equivalent to HTTP 5xx).
    DataError: Problems with training data (NaNs, unsupported types).
    ParameterError: Invalid config or parameter input.
    GenerationError: Sampling/generation failures.
"""

from __future__ import annotations


class SafeSynthesizerError(Exception):
    """Base class for all known Safe Synthesizer errors."""


class UserError(SafeSynthesizerError):
    """Invalid usage -- bad input parameters, uninitialized state, etc.

    If you receive this error, check the documentation for the corresponding
    class and verify your inputs.
    """


class InternalError(SafeSynthesizerError, RuntimeError):
    """Invalid internal state indicating a bug in Safe Synthesizer.

    When using documented interfaces this usually indicates a library bug.
    When using undocumented interfaces it may indicate invalid usage.
    Equivalent to HTTP 5xx status codes.
    """


class DataError(UserError, ValueError):
    """Problems with training data before work is attempted.

    Examples: data contains infinity, too many NaNs, nested structures,
    or types unsupported by the model.
    """


class ParameterError(UserError, ValueError):
    """Invalid configuration or parameter input to user-facing methods.

    Examples: config references a column not present in the data, invalid
    combination of parameters.
    """


class GenerationError(UserError, RuntimeError):
    """Errors during sampling or generation.

    Examples: rejection sampling fails, invalid record threshold met.
    """
