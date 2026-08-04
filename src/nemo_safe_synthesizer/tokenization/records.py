# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered adapters for Datasets columnar batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..errors import ParameterError


def columnar_batch_to_records(
    columns: Mapping[str, Sequence[object]],
    *,
    row_count: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Convert equally sized ordered columns to independent row mappings."""
    if not isinstance(columns, Mapping) or not all(isinstance(name, str) for name in columns):
        raise ParameterError("Record column names must be strings.")
    if row_count is not None and (not isinstance(row_count, int) or isinstance(row_count, bool) or row_count < 0):
        raise ParameterError("Record row_count must be a non-negative integer.")
    normalized: list[tuple[str, Sequence[object]]] = []
    lengths: set[int] = set()
    for name, values in columns.items():
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
            raise ParameterError(f"Record column {name!r} must be a concrete sequence.")
        normalized.append((name, values))
        lengths.add(len(values))
    if len(lengths) > 1:
        raise ParameterError("Record columns must have equal lengths.")
    inferred = next(iter(lengths), 0)
    if row_count is not None and normalized and row_count != inferred:
        raise ParameterError("Explicit record row_count must match all column lengths.")
    count = inferred if row_count is None else row_count
    return tuple({name: values[index] for name, values in normalized} for index in range(count))
