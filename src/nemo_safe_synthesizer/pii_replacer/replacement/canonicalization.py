# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable typed identities for dataframe scalar values."""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from numbers import Integral, Real
from uuid import UUID

import numpy as np
import pandas as pd

from ...errors import ParameterError
from .types import CanonicalValue

__all__ = ["canonicalize_scalar", "is_missing_scalar"]


def is_missing_scalar(value: object) -> bool:
    """Return whether ``value`` is one scalar missing-value marker."""
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    if isinstance(value, np.datetime64 | np.timedelta64):
        return bool(np.isnat(value))
    if isinstance(value, Decimal):
        return value.is_nan()
    if isinstance(value, Real):
        return math.isnan(float(value))
    return False


def canonicalize_scalar(value: object) -> CanonicalValue:
    """Return a deterministic type-tagged identity for a non-missing scalar.

    Python, NumPy, and pandas representations of the same scalar normalize to
    the same identity. Strings remain byte-for-byte unchanged.

    Raises:
        ParameterError: If ``value`` is missing, non-scalar, or unsupported.
    """
    if is_missing_scalar(value):
        raise ParameterError("missing values cannot be canonicalized for PII replacement")
    if isinstance(value, str | np.str_):
        return CanonicalValue("string", str(value))
    if isinstance(value, bool | np.bool_):
        return CanonicalValue("boolean", "true" if bool(value) else "false")
    if isinstance(value, pd.Timestamp | np.datetime64 | datetime):
        return CanonicalValue("datetime", pd.Timestamp(value).isoformat())
    if isinstance(value, date):
        return CanonicalValue("date", value.isoformat())
    if isinstance(value, time):
        return CanonicalValue("time", value.isoformat())
    if isinstance(value, pd.Timedelta | np.timedelta64 | timedelta):
        return CanonicalValue("timedelta_nanoseconds", str(pd.Timedelta(value).value))
    if isinstance(value, Integral):
        return CanonicalValue("integer", str(int(value)))
    if isinstance(value, Decimal):
        return CanonicalValue("decimal", _canonical_decimal(value))
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ParameterError("non-finite numeric values cannot be used for PII replacement")
        return CanonicalValue("float", number.hex())
    if isinstance(value, bytes | np.bytes_):
        return CanonicalValue("bytes_hex", bytes(value).hex())
    if isinstance(value, UUID):
        return CanonicalValue("uuid", str(value))
    raise ParameterError(f"unsupported structured scalar type for PII replacement: {type(value).__name__}")


def _canonical_decimal(value: Decimal) -> str:
    normalized = value.normalize()
    if normalized == 0:
        return "0"
    return format(normalized, "f")
