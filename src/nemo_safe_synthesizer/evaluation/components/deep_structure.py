# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
from category_encoders.count import CountEncoder
from pydantic import ConfigDict, Field

from ...artifacts.analyzers.field_features import (
    FieldType,
    describe_field,
)
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_field import EvaluationField
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...evaluation.statistics import stats
from ...observability import get_logger
from . import multi_modal_figures as figures

logger = get_logger(__name__)


class DeepStructure(Component):
    """Deep Structure Stability metric via joined PCA.

    Projects training and synthetic data into a shared principal-component
    space and scores the distributional similarity of the projections.
    """

    name: str = Field(default="Deep Structure Stability")
    training_pca: pd.DataFrame | None = Field(default=None, description="PCA-projected training dataframe.")
    synthetic_pca: pd.DataFrame | None = Field(default=None, description="PCA-projected synthetic dataframe.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        """Template context with PCA scatter plot figure."""
        d = super().jinja_context
        d["anchor_link"] = "#structure-stability"
        if Component.is_nonempty([self.training_pca, self.synthetic_pca]):
            d["figure"] = figures.structure_stability_figure(
                self.training_pca,  # ty: ignore[invalid-argument-type]
                self.synthetic_pca,  # ty: ignore[invalid-argument-type]
            ).to_html(full_html=False, include_plotlyjs=False)
        else:
            d["figure"] = None
        return d

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets, config: SafeSynthesizerParameters | None = None
    ) -> DeepStructure:
        """Compute PCA projections and the principal component stability score.

        Args:
            evaluation_datasets: Paired training/synthetic data.
            config: Pipeline configuration (unused, reserved for future use).

        Returns:
            A ``DeepStructure`` with PCA dataframes and the stability score.
        """
        tabular_columns = evaluation_datasets.get_tabular_columns(based_on="both")
        if not tabular_columns:
            return DeepStructure(score=EvaluationScore(notes="No columns detected for PCA."))

        training_pca, synthetic_pca = DeepStructure._calculate_pca(
            evaluation_datasets.training[tabular_columns],  # ty: ignore[invalid-argument-type]
            evaluation_datasets.synthetic[tabular_columns],  # ty: ignore[invalid-argument-type]
        )

        principal_component_stability = DeepStructure.get_principal_component_stability(
            training_pca,
            synthetic_pca,
        )

        return DeepStructure(
            score=principal_component_stability, training_pca=training_pca, synthetic_pca=synthetic_pca
        )

    @staticmethod
    def _fill_in_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame | None:
        """Fill missing values in numeric columns with the column median.

        Args:
            df: Dataframe to fill in.
            numeric_columns: Columns to retain and fill.

        Returns:
            A dataframe containing only ``numeric_columns`` with NaNs replaced,
            or ``None`` if ``numeric_columns`` is empty.
        """
        df_num = df.reindex(columns=numeric_columns)
        int64_dtypes = df_num.columns[df_num.dtypes == pd.Int64Dtype()]
        # Check that all columns are numeric, especially needed for the synthetic_df
        # which may have different column types than the training_df
        for col in numeric_columns:
            if not pd.api.types.is_numeric_dtype(df_num[col]):
                raise ValueError(f"Column {col} is not numeric.")
        if len(numeric_columns) == 0:
            return None
        if len(int64_dtypes) > 0:
            # pd.Int64Dtype only accepts integers, hence convert any float median to an integer.
            df_num[int64_dtypes] = df_num[int64_dtypes].fillna(df_num[int64_dtypes].median().astype(int))
        df_num = df_num.fillna(df_num.median())
        return df_num

    @staticmethod
    def _encode_categorical_columns(
        training_df: pd.DataFrame, synthetic_df: pd.DataFrame, categorical_columns: list[str]
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Frequency-encode categorical columns, fitted on the training data.

        New values in ``synthetic_df`` that were not seen in ``training_df`` are
        encoded as 0.

        Args:
            training_df: Training dataframe (used to fit the encoder).
            synthetic_df: Synthetic dataframe (transformed only).
            categorical_columns: Columns to encode.

        Returns:
            Tuple of (encoded training, encoded synthetic), or ``(None, None)``
            if ``categorical_columns`` is empty.
        """
        if len(categorical_columns) == 0:
            return None, None

        training_df_cat = training_df.reindex(columns=categorical_columns)
        synthetic_df_cat = synthetic_df.reindex(columns=categorical_columns)

        # Convert columns to object because count encoder only encodes string or categorical columns
        training_df_cat = training_df_cat.astype("object")
        synthetic_df_cat = synthetic_df_cat.astype("object")

        # Encode categorical columns by the frequency of each value
        # By default the encoder treats nans as a countable category at fit time
        encoder = CountEncoder(handle_missing="value", handle_unknown=0)
        training_df_cat_labels = pd.DataFrame(encoder.fit_transform(training_df_cat))
        synthetic_df_cat_labels = pd.DataFrame(encoder.transform(synthetic_df_cat))
        return training_df_cat_labels, synthetic_df_cat_labels

    @staticmethod
    def _prep_datasets_for_joined_pca(
        training_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Preprocess dataframes for joined PCA.

        Splits columns into numeric and categorical, fills missing values,
        frequency-encodes categoricals, and drops text/other columns that
        would distort the PCA projection.

        Args:
            training_df: Training dataframe (used to fit encoders and PCA).
            synthetic_df: Synthetic dataframe (transformed only).

        Returns:
            Tuple of (preprocessed training, preprocessed synthetic) ready
            for ``compute_joined_pcas``.
        """
        # Identify field types
        training_field_types = [describe_field(name, training_df[name]).type for name in training_df.columns]
        synthetic_field_types = [describe_field(name, synthetic_df[name]).type for name in synthetic_df.columns]
        # Only keep columns that are numeric or categorical.
        # Added this because unique text columns or unique ID columns would distort the results
        # now that we are projecting the count encoding to the synthetic set.
        # When we were encoding them separately, every row would get a value of 1 on both dfs
        # and that column effectively would not contribute to the PCA calculation;
        # in the new joined calculation, the column in the training set would be a 1s,
        # but the column in the synthetic set would be all 0s, because they are all out of distribution.
        categorical_columns = [
            name
            for name, field_type in zip(training_df.columns, training_field_types)
            if field_type in [FieldType.CATEGORICAL, FieldType.BINARY]
        ]
        numeric_columns_training = [
            name
            for name, field_type in zip(training_df.columns, training_field_types)
            if field_type == FieldType.NUMERIC
        ]
        numeric_columns_synthetic = [
            name
            for name, field_type in zip(synthetic_df.columns, synthetic_field_types)
            if field_type == FieldType.NUMERIC
        ]
        numeric_columns = list(set(numeric_columns_training).intersection(set(numeric_columns_synthetic)))
        # TODO: use embeddings to represent text columns
        # TODO: try to include more "OTHER" columns and handle column type mismatch

        if len(categorical_columns) + len(numeric_columns) == 0:
            return pd.DataFrame(), pd.DataFrame()

        training_df_cat_labels, synthetic_df_cat_labels = DeepStructure._encode_categorical_columns(
            training_df, synthetic_df, categorical_columns
        )
        training_df_num = DeepStructure._fill_in_numeric_columns(training_df, numeric_columns)
        synthetic_df_num = DeepStructure._fill_in_numeric_columns(synthetic_df, numeric_columns)

        # Merge numeric and categorical back into one dataframe
        training_df = pd.concat([training_df_num, training_df_cat_labels], axis=1, sort=False)
        synthetic_df = pd.concat([synthetic_df_num, synthetic_df_cat_labels], axis=1, sort=False)
        return training_df, synthetic_df

    @staticmethod
    def _calculate_pca(training_df: pd.DataFrame, synthetic_df: pd.DataFrame):
        """Compute PCA projections for training and synthetic dataframes.

        Subsamples, preprocesses, and runs joined PCA. Returns ``(None, None)``
        if data is insufficient (fewer than 2 rows or 2 columns after prep).

        Args:
            training_df: Tabular training dataframe.
            synthetic_df: Tabular synthetic dataframe.

        Returns:
            Tuple of (training PCA dataframe, synthetic PCA dataframe).
        """
        training_pca = None
        synthetic_pca = None
        try:
            # PCA, start by taking a subsample and dropping empty columns and NAs.
            training_rows = training_df.shape[0]
            synthetic_rows = synthetic_df.shape[0]

            training_subsample = (
                (
                    training_df.sample(n=synthetic_rows, random_state=333)
                    if training_rows > synthetic_rows
                    else training_df
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna(axis="columns", how="all")
            )
            synthetic_subsample = (
                (
                    synthetic_df.sample(n=training_rows, random_state=333)
                    if synthetic_rows > training_rows
                    else synthetic_df
                )
                .replace([np.inf, -np.inf], np.nan)
                .dropna(axis="columns", how="all")
            )

            # Fill in missing values and encode categorical columns to numeric values
            training_subsample_prepped, synthetic_subsample_prepped = DeepStructure._prep_datasets_for_joined_pca(
                training_subsample, synthetic_subsample
            )

            training_ss_rows, training_ss_cols = training_subsample_prepped.shape
            synthetic_ss_rows, synthetic_ss_cols = synthetic_subsample_prepped.shape
            # PCA will fail if number of rows is less than two or number of columns is less than two
            if (training_ss_cols < 2) or (synthetic_ss_cols < 2) or (training_ss_rows < 2) or (synthetic_ss_rows < 2):
                logger.warning(
                    "Need at least 2x2 matrix to calculate PCA values. Not enough non-missing rows or numeric columns after pre-processing."
                )
                return None, None
            else:
                training_pca, synthetic_pca = stats.compute_joined_pcas(
                    training_subsample_prepped, synthetic_subsample_prepped
                )
        except Exception:
            logger.exception("Failed to calculate PCA.")
        return training_pca, synthetic_pca

    @staticmethod
    def get_principal_component_stability(
        training_pca: pd.DataFrame | None,
        synthetic_pca: pd.DataFrame | None,
    ) -> EvaluationScore:
        """Score the distributional similarity of PCA projections.

        Computes per-component Jensen-Shannon divergence, averages,
        and applies an exponential function to produce a 0--10 score.
        """
        if training_pca is None or synthetic_pca is None:
            return EvaluationScore(notes="Missing input Dataframe.")

        try:
            sum_pca_distances = 0.0
            pca_df_fields = [
                EvaluationField.from_series(field, training=training_pca[field], synthetic=synthetic_pca[field])
                for field in training_pca.columns
            ]
            for field in pca_df_fields:
                # field.distribution_distance is None for highly unique fields
                if field.distribution_distance:
                    sum_pca_distances += field.distribution_distance

            raw_score = sum_pca_distances / len(pca_df_fields)

            if np.isnan(raw_score):
                return EvaluationScore()
            # Scale the raw score to between ~2 and 10
            # The factor of 1.6 is to ensure rough consistency with the legacy score
            score = 10 * np.exp(-1.6 * raw_score)
            return EvaluationScore.finalize_grade(raw_score, score)
        except Exception as e:
            logger.exception("Failed to calculate Principal Component Stability SQS")
            return EvaluationScore(notes=str(e))
