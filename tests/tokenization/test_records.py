# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordered columnar-batch value contracts."""

from __future__ import annotations

from typing import Any

import pytest

from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.tokenization.records import ColumnarBatch, columnar_batch_to_records


def test_columnar_batch_preserves_column_and_row_order() -> None:
    batch = ColumnarBatch.parse({"second": [2, 4], "first": [1, 3]})

    assert batch.row_count == 2
    assert batch.to_records() == (
        {"second": 2, "first": 1},
        {"second": 4, "first": 3},
    )
    assert columnar_batch_to_records({"second": [2, 4], "first": [1, 3]}) == batch.to_records()


def test_empty_columnar_batch_honors_explicit_row_count() -> None:
    batch = ColumnarBatch.parse({}, row_count=2)

    assert batch.to_records() == ({}, {})


@pytest.mark.parametrize("row_count", [True, -1, 1.5])
def test_columnar_batch_rejects_invalid_explicit_row_count(row_count: Any) -> None:
    with pytest.raises(ParameterError, match="row_count"):
        ColumnarBatch.parse({}, row_count=row_count)


def test_columnar_batch_rejects_different_column_lengths() -> None:
    with pytest.raises(ParameterError, match="equal lengths"):
        ColumnarBatch.parse({"first": [1], "second": [2, 3]})
