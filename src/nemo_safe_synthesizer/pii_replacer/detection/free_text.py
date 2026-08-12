# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Free-text column selection via NSS field classification."""

from __future__ import annotations

import pandas as pd

from ...artifacts.analyzers.field_features import describe_field
from ...artifacts.base.fields import FieldType
from ...observability import get_logger

logger = get_logger(__name__)

__all__ = ["select_free_text_columns"]


_FREE_TEXT_ELIGIBLE_FIELD_TYPES = frozenset({FieldType.TEXT, FieldType.OTHER})
_NON_FREE_TEXT_FIELD_TYPES = frozenset(
    {
        FieldType.BINARY,
        FieldType.CATEGORICAL,
        FieldType.NUMERIC,
        FieldType.EMPTY,
    }
)


def _free_text_eligibility(col: str, series: pd.Series) -> tuple[bool, str]:
    """Return whether a column may be free text and a short reason string."""
    from pandas.api.types import is_object_dtype, is_string_dtype

    if not (is_object_dtype(series.dtype) or is_string_dtype(series.dtype)):
        return False, f"dtype={series.dtype}"
    field_type = describe_field(col, series).type
    if field_type in _NON_FREE_TEXT_FIELD_TYPES:
        return False, f"field_type={field_type.value}"
    if field_type in _FREE_TEXT_ELIGIBLE_FIELD_TYPES:
        return True, f"field_type={field_type.value}"
    return False, f"field_type={field_type.value}"


def select_free_text_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    """Return columns a run scans as free text via dtype and NSS field classification.

    Discovery uses this to choose scan targets, deferring to the classifier
    the rest of NSS agrees with rather than PII-specific thresholds.

    Example:
        A ``notes`` TEXT column is kept; a numeric or CATEGORICAL column is not.

    Args:
        df: Input dataframe.
        exclude: Column names already handled as structured PII.

    Returns:
        Column names eligible for free-text PII scanning.
    """
    text_fields: list[str] = []
    not_scanned: list[str] = []
    for col in df.columns:
        if col in exclude:
            logger.runtime.debug(
                f"[PII Replacement] Column {col!r} not scanned as free text for PII detection: "
                "already handled as a structured column"
            )
            continue
        eligible, _reason = _free_text_eligibility(col, df[col])
        if eligible:
            text_fields.append(col)
        else:
            not_scanned.append(col)
    scanned_desc = ", ".join(text_fields) if text_fields else "(none)"
    not_scanned_desc = ", ".join(not_scanned) if not_scanned else "(none)"
    logger.runtime.info(
        f"[PII Replacement] Free-text scan for PII detection: "
        f"scanned as text: {scanned_desc}; not scanned: {not_scanned_desc}"
    )
    return text_fields
