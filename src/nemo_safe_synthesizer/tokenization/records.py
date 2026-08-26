# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered adapters for Datasets columnar batches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self

from ..errors import ParameterError


@dataclass(frozen=True, slots=True)
class ColumnarBatch:
    """Validated ordered columns and their resolved row count."""

    columns: tuple[tuple[str, Sequence[object]], ...]
    row_count: int

    @classmethod
    def parse(
        cls,
        columns: Mapping[str, Sequence[object]],
        *,
        row_count: int | None = None,
    ) -> Self:
        """Validate and retain an ordered Datasets columnar batch."""
        if not isinstance(columns, Mapping) or not all(isinstance(name, str) for name in columns):
            raise ParameterError("Record column names must be strings.")
        match row_count:
            case None:
                pass
            case int() as explicit if not isinstance(explicit, bool) and explicit >= 0:
                pass
            case _:
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
        return cls(tuple(normalized), inferred if row_count is None else row_count)

    def to_records(self) -> tuple[dict[str, object], ...]:
        """Return independent row mappings in source column order."""
        return tuple({name: values[index] for name, values in self.columns} for index in range(self.row_count))


def columnar_batch_to_records(
    columns: Mapping[str, Sequence[object]],
    *,
    row_count: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Convert equally sized ordered columns to independent row mappings."""
    return ColumnarBatch.parse(columns, row_count=row_count).to_records()
