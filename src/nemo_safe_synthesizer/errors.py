# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom error classes"""


class SafeSynthesizerError(Exception):
    """Base class for all known errors that can be thrown from nemo_safe_synthesizer."""


class UserError(SafeSynthesizerError):
    """
    Base class for errors that are caused by invalid usage. This is usually caused
    by invalid input parameters, by calling methods on a class that is not initialized,
    etc.
    If you are receiving this error, please see documentation of the corresponding
    classes and check your inputs.
    """


class InternalError(SafeSynthesizerError, RuntimeError):
    """
    Error that indicate invalid internal state.

    If you're using safe_synthesizer through documented interfaces, this usually
    indicates a bug in safe_synthesizer itself.
    If you're using not documented interfaces, this could indicate invalid usage.

    This class of errors is equivalent to 5xx status codes in HTTP protocol.
    """


class DataError(UserError, ValueError):
    """
    Represents problems with training data before work is actually attempted.
    For example: data contains values that are not supported by the model that is
    being used: infinity, too many NaNs, nested data, etc.
    """


class ParameterError(UserError, ValueError):
    """
    Represents errors with configurations or parameter input to user-facing methods.
    For example: config referencing column that is not present in the data.
    """


class GenerationError(UserError, RuntimeError):
    """
    Represents errors happening during sampling/generation.
    For example: rejection sampling fails, invalid record threshold met.
    """
