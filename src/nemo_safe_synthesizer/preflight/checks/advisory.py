# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory-stage data-quality checks. Errors here never gate dependents."""

from __future__ import annotations

from typing_extensions import override

from ..base import AdvisoryCheck, IssueCollector
from ..types import DataFrameView
from ._helpers import resolved_record_count

__all__ = [
    "SmallDatasetCheck",
    "OversamplingCheck",
]


class SmallDatasetCheck(AdvisoryCheck):
    """Advise when the dataset is small but above the hard minimum.

    The hard floor lives in ``DatasetSizeCheck`` (DataFrame stage);
    this check only emits an advisory *warning* between that floor and a
    comfort threshold (``min_rows_warning``).
    """

    name = "dataset.row_count"
    label = "Dataset row count"
    min_rows_warning: int = 1000
    requires = ("dataset.size",)

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        n_rows = len(ctx.data)
        if n_rows < self.min_rows_warning:
            collector.warning(
                "dataset_small",
                f"Training split has {n_rows} rows; consider using more input data (Holdout leaves a fraction aside).",
            )


class OversamplingCheck(AdvisoryCheck):
    """Flag extreme oversampling that risks overfitting."""

    name = "training.oversampling"
    label = "Oversampling"
    oversampling_ratio: float = 5.0

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        n_records = resolved_record_count(ctx)
        if n_records is None:
            return
        n_rows = len(ctx.data)
        data_fraction = n_records / n_rows if n_rows > 0 else 0
        if data_fraction > self.oversampling_ratio:
            collector.warning(
                "extreme_oversampling",
                f"num_input_records_to_sample is {data_fraction:.1f}x the training split size; risk of overfitting.",
            )
