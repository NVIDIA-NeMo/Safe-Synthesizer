# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

import nemo_safe_synthesizer.evaluation.components.multi_modal_figures as figures
import numpy as np
import pandas as pd
from category_encoders.count import CountEncoder
from nemo_safe_synthesizer.artifacts.analyzers.field_features import (
    FieldType,
    describe_field,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import (
    EvaluationDataset,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_field import (
    EvaluationField,
)
from nemo_safe_synthesizer.evaluation.data_model.evaluation_score import (
    EvaluationScore,
)
from nemo_safe_synthesizer.evaluation.statistics import stats
from nemo_safe_synthesizer.observability import get_logger
from pydantic import ConfigDict, Field

logger = get_logger(__name__)


class DeepStructure(Component):
    name: str = Field(default="Deep Structure Stability")
    reference_pca: pd.DataFrame | None = Field(default=None)
    output_pca: pd.DataFrame | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        d = super().jinja_context
        d["anchor_link"] = "#structure-stability"
        if Component.is_nonempty([self.reference_pca, self.output_pca]):
            d["figure"] = figures.structure_stability_figure(
                reference=self.reference_pca, output=self.output_pca
            ).to_html(full_html=False, include_plotlyjs=False)
        else:
            d["figure"] = None
        return d

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> DeepStructure:
        """
        Hook into _calculate_pca to get our PCA values and then use these to get the principal_component_stability SQS.

        Args:
            evaluation_dataset: EvaluationDataset
            config: the main configuration object

        Returns: tuple of reference and output PCA and the principal_component_stability SQS.

        """
        tabular_columns = evaluation_dataset.get_tabular_columns(mode="both")
        if not tabular_columns:
            return DeepStructure(score=EvaluationScore(notes="No columns detected for PCA."))

        reference_pca, output_pca = DeepStructure._calculate_pca(
            evaluation_dataset.reference[tabular_columns],  # ty: ignore[invalid-argument-type]
            evaluation_dataset.output[tabular_columns],  # ty: ignore[invalid-argument-type]
        )

        principal_component_stability = DeepStructure.get_principal_component_stability(
            reference_pca,
            output_pca,
        )

        return DeepStructure(score=principal_component_stability, reference_pca=reference_pca, output_pca=output_pca)

    @staticmethod
    def _fill_in_numeric_columns(df: pd.DataFrame, numeric_columns: list[str]) -> pd.DataFrame | None:
        """
        Fill in missing values in numeric columns with the median value.

        Args:
            df: The dataframe to fill in.
            numeric_columns: The columns to fill in.

        Returns:
            The dataframe with missing values filled in, keeping only the numeric columns.
        """
        df_num = df.reindex(columns=numeric_columns)
        int64_dtypes = df_num.columns[df_num.dtypes == pd.Int64Dtype()]
        # Check that all columns are numeric, especially needed for the output_df
        # which may have different column types than the reference_df
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
        reference_df: pd.DataFrame, output_df: pd.DataFrame, categorical_columns: list[str]
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """
        Encode the categorical columns of the train and synthetic dataframes by the frequency of each value,
        as observed in the train dataframe. Any new values in the synthetic dataframe will be encoded as 0.

        Args:
            reference_df: The train dataframe.
            output_df: The synthetic dataframe.
            categorical_columns: The columns to encode.

        Returns:
            Encoded dataframes of categorical columns from the train and synthetic dataframes.
        """
        if len(categorical_columns) == 0:
            return None, None

        reference_df_cat = reference_df.reindex(columns=categorical_columns)
        output_df_cat = output_df.reindex(columns=categorical_columns)

        # Convert columns to object because count encoder only encodes string or categorical columns
        reference_df_cat = reference_df_cat.astype("object")
        output_df_cat = output_df_cat.astype("object")

        # Encode categorical columns by the frequency of each value
        # By default the encoder treats nans as a countable category at fit time
        encoder = CountEncoder(handle_missing="value", handle_unknown=0)
        reference_df_cat_labels = pd.DataFrame(encoder.fit_transform(reference_df_cat))
        output_df_cat_labels = pd.DataFrame(encoder.transform(output_df_cat))
        return reference_df_cat_labels, output_df_cat_labels

    @staticmethod
    def _prep_datasets_for_joined_pca(
        reference_df: pd.DataFrame,
        output_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess dataframes for joined PCA. Divide the dataframe into numeric/categorical/others,
        fill missing values and encode categorical columns by the frequency of each value.
        Drop the columns that are not numeric or categorical.

        Args:
            reference_df: The training dataframe to be subjected to PCA.
            output_df: The synthetic dataframe. It will be projected onto the principal components of the training dataframe.

        Returns:
            The preprocessed dataframes, ready for joined PCA.

        """
        # Identify field types
        reference_field_types = [describe_field(name, reference_df[name]).type for name in reference_df.columns]
        output_field_types = [describe_field(name, output_df[name]).type for name in output_df.columns]
        # Only keep columns that are numeric or categorical.
        # Added this because unique text columns or unique ID columns would distort the results
        # now that we are projecting the count encoding to the synthetic set.
        # When we were encoding them separately, every row would get a value of 1 on both dfs
        # and that column effectively would not contribute to the PCA calculation;
        # in the new joinedvcalculation, the column in the training set would be a 1s,
        # but the column in the synth set would be all 0s, because they are all out of distribution.
        categorical_columns = [
            name
            for name, field_type in zip(reference_df.columns, reference_field_types)
            if field_type in [FieldType.CATEGORICAL, FieldType.BINARY]
        ]
        numeric_columns_ref = [
            name
            for name, field_type in zip(reference_df.columns, reference_field_types)
            if field_type == FieldType.NUMERIC
        ]
        numeric_columns_out = [
            name for name, field_type in zip(output_df.columns, output_field_types) if field_type == FieldType.NUMERIC
        ]
        numeric_columns = list(set(numeric_columns_ref).intersection(set(numeric_columns_out)))
        # TODO: use embeddings to represent text columns
        # TODO: try to include more "OTHER" columns and handle column type mismatch

        if len(categorical_columns) + len(numeric_columns) == 0:
            return pd.DataFrame(), pd.DataFrame()

        reference_df_cat_labels, output_df_cat_labels = DeepStructure._encode_categorical_columns(
            reference_df, output_df, categorical_columns
        )
        reference_df_num = DeepStructure._fill_in_numeric_columns(reference_df, numeric_columns)
        output_df_num = DeepStructure._fill_in_numeric_columns(output_df, numeric_columns)

        # Merge numeric and categorical back into one dataframe
        reference_df = pd.concat([reference_df_num, reference_df_cat_labels], axis=1, sort=False)
        output_df = pd.concat([output_df_num, output_df_cat_labels], axis=1, sort=False)
        return reference_df, output_df

    @staticmethod
    def _calculate_pca(reference: pd.DataFrame, output: pd.DataFrame):
        """
        Compute the PCA values for reference and output dataframes, after doing some sanity checks and sampling.

        Args:
            reference: pd.DataFrame
            output: pd.DataFrame

        Returns: tuple of reference and output PCA values

        """
        reference_pca = None
        output_pca = None
        try:
            # PCA, start by taking a subsample and dropping empty columns and NAs.
            reference_rows = reference.shape[0]
            output_rows = output.shape[0]

            reference_subsample = (
                (reference.sample(n=output_rows, random_state=333) if reference_rows > output_rows else reference)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(axis="columns", how="all")
            )
            output_subsample = (
                (output.sample(n=reference_rows, random_state=333) if output_rows > reference_rows else output)
                .replace([np.inf, -np.inf], np.nan)
                .dropna(axis="columns", how="all")
            )

            # Fill in missing values and encode categorical columns to numeric values
            reference_subsample_prepped, output_subsample_prepped = DeepStructure._prep_datasets_for_joined_pca(
                reference_subsample, output_subsample
            )

            reference_ss_rows, reference_ss_cols = reference_subsample_prepped.shape
            output_ss_rows, output_ss_cols = output_subsample_prepped.shape
            # PCA will fail if number of rows is less than two or number of columns is less than two
            if (reference_ss_cols < 2) or (output_ss_cols < 2) or (reference_ss_rows < 2) or (output_ss_rows < 2):
                logger.warning(
                    "Need at least 2x2 matrix to calculate PCA values. Not enough non-missing rows or numeric columns after pre-processing."
                )
                return None, None
            else:
                reference_pca, output_pca = stats.compute_joined_pcas(
                    reference_subsample_prepped, output_subsample_prepped
                )
        except Exception:
            logger.exception("Failed to calculate PCA.")
        return reference_pca, output_pca

    @staticmethod
    def get_principal_component_stability(
        reference_pca: pd.DataFrame | None,
        output_pca: pd.DataFrame | None,
    ) -> EvaluationScore:
        if reference_pca is None or output_pca is None:
            return EvaluationScore(notes="Missing input Dataframe.")

        try:
            sum_pca_distances = 0.0
            pca_df_fields = [
                EvaluationField.from_series(field, reference_pca[field], output_pca[field])
                for field in reference_pca.columns
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
