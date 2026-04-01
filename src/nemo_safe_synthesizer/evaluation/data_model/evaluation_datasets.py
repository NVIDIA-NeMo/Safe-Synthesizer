# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import random

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...artifacts.analyzers.field_features import FieldType
from ...config.evaluate import DEFAULT_RECORD_COUNT, DEFAULT_SQS_REPORT_COLUMNS
from ...evaluation.data_model.evaluation_field import EvaluationField
from ...evaluation.statistics import stats
from ...observability import get_logger
from ...pii_replacer.transform_result import ColumnStatistics

logger = get_logger(__name__)


class EvaluationDatasets(BaseModel):
    """Training, synthetic, and optionally test dataframes prepared for evaluation.

    On construction the validator computes per-column ``EvaluationField``
    instances, counts memorized lines, and records dataset dimensions.
    Use ``from_dataframes`` to build an instance with optional column/row
    subsampling.
    """

    training: pd.DataFrame = Field(default=pd.DataFrame(), description="Training dataframe.")
    synthetic: pd.DataFrame = Field(default=pd.DataFrame(), description="Synthetic dataframe.")
    test: pd.DataFrame | None = Field(
        default=None, description="Optional holdout dataframe for text-similarity and privacy metrics."
    )

    training_rows: int = Field(default=0, ge=0, description="Row count of the training dataframe.")
    training_cols: int = Field(default=0, ge=0, description="Column count of the training dataframe.")
    synthetic_rows: int = Field(default=0, ge=0, description="Row count of the synthetic dataframe.")
    synthetic_cols: int = Field(default=0, ge=0, description="Column count of the synthetic dataframe.")
    memorized_lines: int = Field(
        default=0, ge=0, description="Number of exact row matches between training and synthetic."
    )

    column_statistics: dict[str, ColumnStatistics] | None = Field(
        default=None, description="Per-column PII entity counts and transform metadata."
    )
    evaluation_fields: list[EvaluationField] = Field(
        default=list(), description="Per-column evaluation metadata and distribution scores."
    )

    # DataFrame fields make pydantic... le sad panda
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def check_dataframe(df: pd.DataFrame, df_name: str):
        """Raise ``ValueError`` if ``df`` is ``None`` or empty."""
        if df is None:
            raise ValueError(f"{df_name} is None!")
        if df.empty:
            raise ValueError(f"{df_name} is empty!")

    def get_columns_of_type(self, types: set[FieldType], mode="training") -> list[str]:
        """Return column names whose ``FieldType`` is in ``types``.

        Args:
            types: Set of ``FieldType`` values to match.
            mode: Which dataframe's field features to inspect --
                ``"training"``, ``"synthetic"``, or ``"both"`` (intersection).

        Returns:
            List of matching column names.
        """
        if mode == "training":
            return [f.name for f in self.evaluation_fields if f.training_field_features.type in types]
        elif mode == "synthetic":
            return [f.name for f in self.evaluation_fields if f.synthetic_field_features.type in types]
        elif mode == "both":
            return [
                f.name
                for f in self.evaluation_fields
                if f.training_field_features.type in types and f.synthetic_field_features.type in types
            ]
        else:
            return []

    def get_tabular_columns(self, mode="training") -> list[str]:
        """Return columns classified as binary, categorical, or numeric."""
        return self.get_columns_of_type({FieldType.BINARY, FieldType.CATEGORICAL, FieldType.NUMERIC}, mode)

    def get_nominal_columns(self, mode="training") -> list[str]:
        """Return columns classified as binary or categorical."""
        return self.get_columns_of_type({FieldType.BINARY, FieldType.CATEGORICAL}, mode)

    def get_text_columns(self, mode="training") -> list[str]:
        """Return columns classified as free text."""
        return self.get_columns_of_type({FieldType.TEXT}, mode)

    @model_validator(mode="after")
    def validate(self):
        # Expected raw input data:
        # training
        # synthetic
        # test (optional)
        # column_statistics (optional)

        # Check that df's exist and are not empty
        EvaluationDatasets.check_dataframe(self.training, "Training")
        EvaluationDatasets.check_dataframe(self.synthetic, "Synthetic")

        # Make all the evaluation fields.
        column_statistics: dict[str, ColumnStatistics] = self.column_statistics if self.column_statistics else dict()
        evaluation_fields: list[EvaluationField] = []
        for col in self.training.columns:
            evaluation_fields.append(
                EvaluationField.from_series(
                    name=col,
                    training=self.training[col],
                    synthetic=self.synthetic[col],
                    column_statistics=column_statistics.get(col),
                )
            )

        training_rows, training_cols = self.training.shape
        synthetic_rows, synthetic_cols = self.synthetic.shape
        memorized_lines = stats.count_memorized_lines(self.training, self.synthetic)

        self.training_rows = training_rows
        self.training_cols = training_cols
        self.synthetic_rows = synthetic_rows
        self.synthetic_cols = synthetic_cols
        self.memorized_lines = memorized_lines

        self.evaluation_fields = evaluation_fields
        return self

    @staticmethod
    def subsample_columns(
        training: pd.DataFrame,
        synthetic: pd.DataFrame,
        test: pd.DataFrame | None = None,
        target_column_count: int = DEFAULT_SQS_REPORT_COLUMNS,
        mandatory_columns: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        """Reduce dataframes to shared columns, optionally subsampling columns.

        Mandatory columns are always included. A fixed random seed ensures
        reproducible column selection across evaluation components.

        Args:
            training: Training dataframe.
            synthetic: Synthetic dataframe.
            test: Optional holdout dataframe.
            target_column_count: Maximum number of columns to keep.
            mandatory_columns: Columns that must be included regardless.

        Returns:
            Tuple of (training, synthetic, test) dataframes restricted to the
            selected column set.

        Raises:
            ValueError: If training and synthetic share no columns.
        """
        if mandatory_columns is None:
            mandatory_columns = []
        # Check and subsample columns
        shared_columns = set(training.columns).intersection(set(synthetic.columns))
        if len(shared_columns) == 0:
            raise ValueError(
                "Training and Synthetic dataframes contain no columns in common. Please check dataframes for mismatch."
            )
        if target_column_count < len(shared_columns):
            logger.info(
                f"Found {len(shared_columns)} shared columns. Attempting to sample down to {target_column_count} columns. Will include {len(mandatory_columns)} mandatory columns."
            )
            col_set = set()
            col_set.update([col for col in mandatory_columns if col in shared_columns])

            if len(col_set) < target_column_count:
                # Use a fixed seed for reproducibility. In particular, we want to sample the
                # same columns for correlation and everything else.
                r = random.Random()
                r.seed(2112)
                shared_columns = shared_columns.difference(set(mandatory_columns))
                col_set.update(r.sample(list(shared_columns), k=(target_column_count - len(col_set))))
        else:
            # Even without sampling, we only want to use shared columns.
            col_set = shared_columns
        training = training[list(col_set)]  # ty: ignore[invalid-assignment]
        synthetic = synthetic[list(col_set)]  # ty: ignore[invalid-assignment]

        # Check and subsample test columns, or split out a test set if not provided.
        if test is not None and not test.empty:
            test_shared_columns = shared_columns.intersection(set(test.columns))
            if len(test_shared_columns) == 0:
                raise ValueError(
                    "Test dataframe has no columns in common with Training and Synthetic dataframes. Please check dataframes for mismatch."
                )
            else:
                test = test[list(test_shared_columns)]  # ty: ignore[invalid-assignment]

        return training, synthetic, test

    @staticmethod
    def subsample_rows(
        training: pd.DataFrame,
        synthetic: pd.DataFrame,
        target_record_count: int = DEFAULT_RECORD_COUNT,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Downsample both dataframes to at most ``target_record_count`` rows."""
        target_record_count = min(target_record_count, training.shape[0], synthetic.shape[0])
        if target_record_count < training.shape[0]:
            logger.info(f"Subsampling training data from {training.shape[0]} records to {target_record_count}.")
            training = training.sample(target_record_count, ignore_index=True, random_state=424242)
        if target_record_count < synthetic.shape[0]:
            logger.info(f"Subsampling synthetic data from {synthetic.shape[0]} records to {target_record_count}.")
            synthetic = synthetic.sample(target_record_count, ignore_index=True, random_state=424242)
        return training, synthetic

    @staticmethod
    def from_dataframes(
        training: pd.DataFrame,
        synthetic: pd.DataFrame,
        test: pd.DataFrame | None = None,
        column_statistics: dict[str, ColumnStatistics] | None = None,
        rows: int = DEFAULT_RECORD_COUNT,
        cols: int = DEFAULT_SQS_REPORT_COLUMNS,
        mandatory_columns: list[str] | None = None,
        enable_sampling: bool = True,
    ) -> EvaluationDatasets:
        """Build an ``EvaluationDatasets`` with optional column/row subsampling.

        This is the primary constructor for evaluation. It validates inputs,
        optionally subsamples columns and rows, then delegates to the
        Pydantic model validator which computes per-column evaluation fields.

        Args:
            training: Training dataframe.
            synthetic: Synthetic dataframe.
            test: Optional holdout dataframe for text-similarity and privacy metrics.
            column_statistics: Per-column PII entity metadata.
            rows: Target row count for subsampling.
            cols: Target column count for subsampling.
            mandatory_columns: Columns to always include in subsampling.
            enable_sampling: When ``False``, skip all subsampling.

        Returns:
            A fully initialized ``EvaluationDatasets``.
        """
        # Spot check df's before doing anything.
        EvaluationDatasets.check_dataframe(training, "Training")
        EvaluationDatasets.check_dataframe(synthetic, "Synthetic")

        # Sample while we have config params in hand.
        if enable_sampling:
            training, synthetic, test = EvaluationDatasets.subsample_columns(
                training, synthetic, test, cols, mandatory_columns
            )
            training, synthetic = EvaluationDatasets.subsample_rows(training, synthetic, rows)

        return EvaluationDatasets(
            training=training, synthetic=synthetic, test=test, column_statistics=column_statistics
        )
