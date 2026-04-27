# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory-stage data-quality checks. Errors here never gate dependents."""

from __future__ import annotations

from ..base import AdvisoryCheck, IssueCollector
from ..types import DataFrameView
from ._helpers import resolved_record_count

__all__ = [
    "SmallDatasetCheck",
    "OversamplingCheck",
    "TrainingStepsCheck",
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


class TrainingStepsCheck(AdvisoryCheck):
    """Warn when the configured sample size yields too few optimizer steps.

    Estimates the number of post-assembly training examples and compares
    ``n_examples / effective_batch_size`` against ``min_training_steps``.
    ``num_input_records_to_sample`` counts pre-assembly raw records, which
    has a highly variable relationship to the post-assembly example count
    -- especially when ``group_training_examples_by`` packs many rows into
    one training example. The estimate here prefers the tighter quantity:

    * With ``group_training_examples_by`` set: ``n_examples ~= n_groups``
      (the grouped assembler produces at least one example per group, and
      typically exactly one for groups that fit in ``max_seq_length``).
    * Otherwise: ``n_examples ~= min(n_records, n_rows)`` (each raw row is
      an example before any packing by the ungrouped assembler; packing
      can only reduce the count further, so this is still an upper bound
      but tighter than ``n_records`` alone when the user oversamples).

    ``requires = ("columns.groupby",)`` so the check is skipped when the
    group column is missing or has nulls -- no point estimating from a
    column we already know is broken.
    """

    name = "training.steps"
    label = "Training steps"
    category = "configuration"
    requires = ("columns.groupby",)
    min_training_steps: int = 10

    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        n_records = resolved_record_count(ctx)
        if n_records is None:
            return
        data = ctx.data
        n_rows = len(data)
        group_col = ctx.config.data.group_training_examples_by

        if group_col is not None and group_col in data.columns:
            n_examples = int(data[group_col].nunique())
            basis = f"{n_examples} groups from '{group_col}'"
        else:
            n_examples = min(n_records, n_rows) if n_rows > 0 else n_records
            basis = f"~{n_examples} training examples"

        effective_batch = ctx.config.training.effective_batch_size
        effective_steps = n_examples / effective_batch if effective_batch > 0 else 0

        if effective_steps >= self.min_training_steps:
            return

        batch_exceeds_data = (
            f"Effective batch size ({effective_batch}) exceeds the training-example "
            f"count ({n_examples}), so the optimizer will take fewer than 1 step per epoch. "
            if effective_batch > n_examples
            else ""
        )
        collector.warning(
            "few_training_steps",
            (
                f"{batch_exceeds_data}"
                f"Effective training steps (~{effective_steps:.0f}) is below the recommended "
                f"minimum of {self.min_training_steps}: {basis} at effective batch size "
                f"{effective_batch}. Increase num_input_records_to_sample or reduce "
                "batch_size / gradient_accumulation_steps."
            ),
        )
