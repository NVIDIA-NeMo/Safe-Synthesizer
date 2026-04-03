# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import math
import re
from datetime import datetime
from decimal import Decimal
from functools import cached_property
from math import e
from typing import cast

import category_encoders as ce
import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype
from pydantic import ConfigDict, Field
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import QuantileTransformer

from ...config.evaluate import QUASI_IDENTIFIER_COUNT
from ...config.parameters import SafeSynthesizerParameters
from ...observability import get_logger
from ..components.component import Component
from ..data_model.evaluation_dataset import EvaluationDataset
from ..data_model.evaluation_score import EvaluationScore, PrivacyGrade
from ..nearest_neighbors import NearestNeighborSearch
from . import multi_modal_figures as figures
from .privacy_metric_utils import divide_tabular_text, embed_text, find_text_fields

logger = get_logger(__name__)


class AttributeInferenceProtection(Component):
    """Attribute Inference Protection privacy metric.

    Simulates an attribute inference attack: given quasi-identifier columns,
    can an adversary use synthetic nearest-neighbors to predict the remaining
    attributes of a training record?  A higher score indicates better
    protection (lower prediction accuracy).

    See Also:
        https://arxiv.org/abs/2501.03941 -- Synthetic Data Privacy Metrics.
    """

    name: str = Field(default="Attribute Inference Protection")
    col_accuracy_df: pd.DataFrame | None = Field(
        default=None, description="Per-column prediction risk scores and grades."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self) -> dict[str, str]:
        """Template context with the attribute-inference bar chart figure."""
        d = super().jinja_context
        d["anchor_link"] = "#aia"
        if self.col_accuracy_df is not None and not self.col_accuracy_df.empty:
            d["figure"] = figures.generate_aia_figure(self.col_accuracy_df).to_html(
                full_html=False, include_plotlyjs=False
            )
        else:
            d["figure"] = None
        return d

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> AttributeInferenceProtection:
        """Run the attribute inference attack and return the protection score."""
        quasi_identifier_count = config.evaluation.quasi_identifier_count if config else QUASI_IDENTIFIER_COUNT

        score, col_accuracy_df = AttributeInferenceProtection._aia(
            df_train=evaluation_dataset.reference,
            df_synth=evaluation_dataset.output,
            quasi_identifier_count=quasi_identifier_count,
        )
        return AttributeInferenceProtection(score=score, col_accuracy_df=col_accuracy_df)

    @staticmethod
    def _normalize(df_train: pd.DataFrame, df_synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_train = df_train.infer_objects()
        df_synth = df_synth.infer_objects()
        df = pd.concat([df_train, df_synth])

        nominal_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        numeric_columns = []
        for column in df_train.columns:
            if column not in nominal_columns:
                numeric_columns.append(column)

        encoder = ce.BinaryEncoder(cols=nominal_columns, return_df=True)
        df_norm = encoder.fit_transform(df)

        n_quantiles = min(1000, len(df_norm))
        num_encoder = QuantileTransformer(n_quantiles=n_quantiles, random_state=0)
        model: QuantileTransformer = cast(
            QuantileTransformer,
            num_encoder.fit(df_norm),
        )
        df_norm = pd.DataFrame(model.transform(df_norm), columns=df_norm.columns).fillna(0)

        df_train_norm = df_norm.head(len(df_train))
        df_synth_norm = df_norm.tail(len(df_synth))

        return df_train_norm, df_synth_norm

    @staticmethod
    def _normalize_onehot(df_train: pd.DataFrame, df_synth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_train = df_train.infer_objects()
        df_synth = df_synth.infer_objects()
        df = pd.concat([df_train, df_synth])

        nominal_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        numeric_columns = []
        for column in df_train.columns:
            if column not in nominal_columns:
                numeric_columns.append(column)

        df_norm = pd.get_dummies(df, columns=nominal_columns)

        num_encoder = QuantileTransformer()
        num_encoder.fit(df_norm)
        df_norm = pd.DataFrame(num_encoder.transform(df_norm), columns=df_norm.columns).fillna(0)

        df_train_norm = df_norm.head(len(df_train))
        df_synth_norm = df_norm.tail(len(df_synth))

        return df_train_norm, df_synth_norm

    @staticmethod
    def _pandas_entropy(column: pd.Series, base: float | None = None) -> np.float64:
        vc = column.value_counts(normalize=True, sort=False)
        base = e if base is None else base
        return -(vc * np.log(vc) / np.log(base)).sum()

    @staticmethod
    def _is_really_categorical(column: str) -> bool:
        # Break the header up into parts
        for separator in ["_", " ", "-", "."]:
            if separator in column:
                col_parts = column.split(separator)
                break
            else:
                col_parts = [column]

        # Go through the parts and divide up camel case
        col_final_parts = []
        for part in col_parts:
            splitted = re.sub("([A-Z][a-z]+)", r" \1", re.sub("([A-Z]+)", r" \1", part)).split()
            for token in splitted:
                col_final_parts.append(token.lower())

        for part in col_final_parts:
            if part in [
                "class",
                "code",
                "cpt",
                "csn",
                "date",
                "end",
                "icd",
                "id",
                "key",
                "mrn",
                "nbr",
                "number",
                "postcode",
                "start",
                "time",
                "timestamp",
                "year",
                "yr",
                "yyyy",
                "yyyymm",
                "zip",
                "zipcode",
            ]:
                return True

        return False

    @staticmethod
    def _parse_dates(value: str | int | float, scalar_type: str | None = None) -> list[tuple[str, datetime]] | None:
        if scalar_type == "number" and isinstance(value, str):
            # TODO(PROD-276): this is necessary, as our regex predictors change values from kv_pair
            #  to strings, so we need to use "scalar_type" field to figure out if the
            #  original value was a number.
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass

        if isinstance(value, float) and math.isnan(value):
            # don't try to match something that is NaN
            # when converted to string, "nan" is name for August
            # in this locale: https://www.localeplanet.com/icu/mgh/index.html
            return None

    @staticmethod
    def _get_synth_nn(
        df_train_use: pd.DataFrame,
        df_synth_use: pd.DataFrame,
        df_synth: pd.DataFrame,
        text_columns: list[str],
        numeric_columns: list[str],
        nominal_columns: list[str],
        embedder,
    ) -> pd.DataFrame:
        # Note, when entering this function, df_train_use is exactly one record

        # Number of synthetic nearest neighbors to be retrieved
        k = 5

        # First divide out text and non-text
        if len(text_columns) > 0:
            df_train_use, df_train_text = divide_tabular_text(df_train_use, text_columns)
            df_synth_use, df_synth_text = divide_tabular_text(df_synth_use, text_columns)

        # Normalize the tabular data if there is any
        tabular_columns = numeric_columns + nominal_columns
        if len(tabular_columns) > 0:
            try:
                df_train_norm, df_synth_norm = AttributeInferenceProtection._normalize(df_train_use, df_synth_use)
            except Exception:
                df_train_norm, df_synth_norm = AttributeInferenceProtection._normalize_onehot(
                    df_train_use, df_synth_use
                )

        # If all tabular, use nearest neighbor search (torch CUDA or sklearn CPU fallback)
        if len(text_columns) == 0:
            # Create the nearest neighbors index on the synthetic data
            nn = NearestNeighborSearch(n_neighbors=k)
            nn.fit(np.ascontiguousarray(np.array(df_synth_norm)).astype(np.float32))

            # Get nearest neighbors to this attack record
            _, indexes = nn.kneighbors(np.ascontiguousarray(np.array(df_train_norm)).astype(np.float32))
            synth_rows = pd.DataFrame()
            for idx_row in indexes:
                synth_rows = pd.concat([synth_rows, df_synth.iloc[idx_row].copy()])
            return synth_rows

        # If all text, just use Sentence Transformer to get NN
        if len(tabular_columns) == 0:
            # Create embeddings for text fields
            df_train_embeddings = embed_text(df_train_text, embedder)
            df_synth_embeddings = embed_text(df_synth_text, embedder)
            hits = util.semantic_search(
                np.array(list(df_train_embeddings["embedding"])),  # ty: ignore[invalid-argument-type]
                np.array(list(df_synth_embeddings["embedding"])),  # ty: ignore[invalid-argument-type]
                top_k=k,
            )
            synth_rows = pd.DataFrame()
            for i in range(k):
                corpus_id = hits[0][i]["corpus_id"]
                synth_rows = pd.concat(
                    [synth_rows, pd.DataFrame([df_synth.iloc[int(corpus_id)]])],
                    ignore_index=True,
                )

            return synth_rows

        # If we made it to here then we need to handle the scenario where it's a combo of text and non-text

        # Get the text embeddings and then the 1000 NN based on just the text

        df_train_embeddings = embed_text(df_train_text, embedder)
        df_synth_embeddings = embed_text(df_synth_text, embedder)
        search_synth_k = min(1000, len(df_synth_embeddings))
        hits = util.semantic_search(
            np.array(list(df_train_embeddings["embedding"])),  # ty: ignore[invalid-argument-type]
            np.array(list(df_synth_embeddings["embedding"])),  # ty: ignore[invalid-argument-type]
            top_k=search_synth_k,
        )
        synth_NN = pd.DataFrame()
        text_dist = {}
        corpus_ids = []

        for i in range(search_synth_k):
            corpus_id = hits[0][i]["corpus_id"]
            sim = hits[0][i]["score"]
            dist = 1 - sim
            text_dist[i] = dist
            corpus_ids.append(corpus_id)
            synth_NN = pd.concat([synth_NN, pd.DataFrame([df_synth_norm.iloc[int(corpus_id)]])], ignore_index=True)

        # Now get the tabular similarity for these 1000 NN using nearest neighbor search
        nn = NearestNeighborSearch(n_neighbors=search_synth_k)
        nn.fit(np.ascontiguousarray(np.array(synth_NN)).astype(np.float32))
        dists, indexes = nn.kneighbors(np.ascontiguousarray(np.array(df_train_norm)).astype(np.float32))
        # Scale the Euclidean distance to [0,1]
        max_dist = np.amax(dists)
        if max_dist > 0:
            dist_scaled = dists / max_dist
        else:
            dist_scaled = dists
        tab_dist = {}
        for i in range(search_synth_k):
            idx = indexes[0][i]
            tab_dist[idx] = dist_scaled[0][i]

        # Now get the hybrid distance

        hybrid_dist = {}
        text_weight = len(text_columns) / len(df_train_use.columns)
        tab_weight = len(tabular_columns) / len(df_train_use.columns)
        for i in range(search_synth_k):
            dist = text_weight * text_dist[i] + tab_weight * tab_dist[i]
            real_index = corpus_ids[i]
            hybrid_dist[real_index] = dist

        # Get the indices of the k smallest distances

        synth_rows = pd.DataFrame()
        sorted_dist = sorted(hybrid_dist)

        for i in range(k):
            index = sorted_dist[i]
            entry = pd.DataFrame([df_synth.iloc[index]])
            synth_rows = pd.concat([synth_rows, entry], ignore_index=True)

        return synth_rows

    @staticmethod
    def _aia(
        df_train: pd.DataFrame,
        df_synth: pd.DataFrame,
        quasi_identifier_count: int,
    ) -> tuple[EvaluationScore, pd.DataFrame | None]:
        """Core attribute inference attack implementation.

        Iterates over random quasi-identifier subsets, finds nearest
        synthetic neighbors, and measures attribute prediction accuracy
        weighted by column entropy.

        Args:
            df_train: Training dataframe.
            df_synth: Synthetic dataframe.
            quasi_identifier_count: Number of columns to use as quasi-identifiers.

        Returns:
            Tuple of (overall protection score, per-column accuracy dataframe).
        """
        ias = EvaluationScore(grade=PrivacyGrade.UNAVAILABLE)
        col_accuracy_df = None
        if quasi_identifier_count is None:
            logger.warning("Missing quasi_identifier_count for Attribute Inference Attack.")
            return (ias, col_accuracy_df)
        if df_train is None or df_synth is None:
            logger.warning("Missing input data for Attribute Inference Attack.")
            return (ias, col_accuracy_df)

        try:
            if len(df_train.columns) < (quasi_identifier_count + 1):
                quasi_identifier_count = len(df_train.columns) - 1
            if quasi_identifier_count == 0:
                logger.warning("Too few columns for Attribute Inference Attack.")
                return (ias, col_accuracy_df)

            # Get all combinations of columns to be the quasi-identifiers
            # This gets explosive when column count > 500
            if len(df_train.columns) < 500:
                qi_combos = list(itertools.combinations(df_train.columns, quasi_identifier_count))
            else:
                columns = list(df_train.columns)
                qi_combos = []
                for i in range(len(columns) - quasi_identifier_count):
                    combo = set()
                    for j in range(quasi_identifier_count):
                        combo.add(columns[i + j])
                    qi_combos.append(combo)

            np.random.seed(5)
            np.random.shuffle(qi_combos)

            # Calculate nominal and numeric columns
            nominal_columns = list(df_train.select_dtypes(include=["object", "category", "bool"]).columns)
            numeric_columns = [column for column in df_train.columns if column not in nominal_columns]

            # Now separate out the text columns from the nominal
            text_columns = find_text_fields(cast(pd.DataFrame, df_train[nominal_columns]))
            nominal_columns = [x for x in nominal_columns if x not in text_columns]

            # If there are text columns, create an embedder
            if len(text_columns) > 0:
                embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")
            else:
                embedder = None

            # Correct the numeric/nominal column lists
            # Certain types of numeric fields like IDs, dates, timestamps, codes
            # should be treated as categorical in the below processing
            true_numeric_columns = []
            for column in numeric_columns:
                if AttributeInferenceProtection._is_really_categorical(column):
                    nominal_columns.append(column)
                else:
                    true_numeric_columns.append(column)
            numeric_columns = true_numeric_columns

            # Pick the attack dataset. This will be 1% of the training data, with a minimum of 200 records
            desired_size = max(int(len(df_train) * 0.01), 200)
            # If this is multi-modal limit the attack set to 100
            if len(text_columns) > 0:
                if desired_size > 100:
                    desired_size = 100
            attack_data = df_train.sample(n=desired_size, random_state=1, replace=True)

            # As we process the attack dataset, we'll accumulate for each column the number of
            # correct and incorrect predictions
            correct = {predict_column: 0 for predict_column in df_train.columns}
            incorrect = {predict_column: 0 for predict_column in df_train.columns}

            qi_index = 0

            # Loop through the attack dataset
            next = -1
            more_to_process = True
            while more_to_process:
                next += 1

                if next == len(attack_data):
                    more_to_process = False
                    continue

                # Randomly sample columns to be the `qi`
                qi = qi_combos[qi_index]
                qi_index += 1
                # We stop processing if all qi combos have been processed.
                if qi_index == len(qi_combos):
                    more_to_process = False
                    continue

                # Predict columns are all but the `qi`
                predict_columns = [column for column in df_train if column not in qi]

                # Filter down to the `qi`
                train_row_all = attack_data.iloc[[next]].copy()
                # If any of the `qi` are nan, skip the attack record
                train_row = train_row_all.filter(qi).dropna()

                if len(train_row) == 0:
                    continue
                df_synth_use = df_synth.filter(qi).dropna()

                # Divide the QI up into numeric, nominal, text
                qi_text = [x for x in qi if x in text_columns]
                qi_nominal = [x for x in qi if x in nominal_columns]
                qi_numeric = [x for x in qi if x in numeric_columns]

                # Get the synth NN
                synth_rows = AttributeInferenceProtection._get_synth_nn(
                    train_row,
                    df_synth_use,
                    df_synth,
                    qi_text,
                    qi_numeric,
                    qi_nominal,
                    embedder,
                )

                # Get majority wins/mean for the close synthetic records
                synth_values = {}
                for column in predict_columns:
                    if column in numeric_columns:
                        try:
                            synth_values[column] = np.mean(synth_rows[column])
                        except Exception:
                            synth_values[column] = 0
                    elif column in nominal_columns:
                        temp = synth_rows[column].value_counts()
                        if len(temp.index) == 0:
                            synth_values[column] = ""
                        else:
                            synth_values[column] = temp.index[0]
                    # Else it's text, get the mean embedding
                    else:
                        values = list(synth_rows[column])
                        cleaned_values = [x for x in values if str(x) != "nan"]
                        if len(cleaned_values) == 0:
                            synth_values[column] = 0
                        else:
                            if embedder is None:
                                raise RuntimeError("Embedder is not available for text column in AIA.")
                            embeddings = embedder.encode(
                                cleaned_values,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                            )
                            synth_values[column] = np.mean(list(embeddings), axis=0)

                # Compare original training record to this matched synthetic record
                # Categorical matches must be exact, and numeric is default 1% but
                # increasingly less depending on how many digits to the right of the decimal
                # Lat/lon values inspired this. Text must be dist .35 or less
                for column in predict_columns:
                    synth_val = synth_values[column]
                    train_val = train_row_all.iloc[0][column]  # ty: ignore[invalid-argument-type]

                    if pd.isna(train_val):
                        continue
                    numeric_sim = 0.01
                    if column in numeric_columns:
                        if is_float_dtype(df_train[column]):
                            d = Decimal(train_val)
                            # Skip nan value
                            try:
                                # Ignoring the type error is okay because the try/except handles that case.
                                d = abs(d.as_tuple().exponent)  # ty: ignore[invalid-argument-type]
                            except (ValueError, AttributeError):
                                continue

                            for i in range(d):
                                numeric_sim = numeric_sim / 10

                        diff = abs(synth_val - train_val)
                        if diff < (numeric_sim * train_val):
                            correct[column] += 1
                        else:
                            incorrect[column] += 1
                    elif column in nominal_columns:
                        if synth_val == train_val:
                            correct[column] += 1
                        else:
                            incorrect[column] += 1
                    else:  # Column is text
                        # Get embedding for train_val
                        train_val_list = [train_val]
                        if embedder is None:
                            raise RuntimeError("Embedder is not available for text column in AIA.")
                        train_val_embedding = embedder.encode(
                            list(train_val_list),
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        # Compare synth and train embeddings
                        # if all embeddings were nan, then synth_val is an int with value 0
                        if not isinstance(synth_val, int):
                            denom = np.linalg.norm(train_val_embedding) * np.linalg.norm(synth_val)
                            if denom == 0:
                                sim = 0
                            else:
                                sim = float(np.dot(train_val_embedding, synth_val) / denom)
                            dist = 1 - sim
                            if dist <= 0.35:
                                correct[column] += 1
                            else:
                                incorrect[column] += 1

            # Compute overall accuracy for each column
            accuracy = {}
            col_assess_cnt = {}
            for column in df_train.columns:
                col_assess_cnt[column] = correct[column] + incorrect[column]
                if (col_assess_cnt[column]) <= 0:
                    accuracy[column] = 0
                else:
                    accuracy[column] = round(
                        correct[column] / (correct[column] + incorrect[column]),
                        2,
                    )

            # Use entropy as weights to combine the individual column scores into an overall score
            entropy = []
            for column in accuracy:
                entropy.append(round(AttributeInferenceProtection._pandas_entropy(df_train[column]), 2))
            arr = []
            entropy_wts = []
            if max(entropy) == min(entropy):
                for i in range(len(entropy)):
                    entropy_wts.append(0)
            else:
                arr = (entropy - min(entropy)) / (max(entropy) - min(entropy))
                entropy_wts = arr / arr.sum()

            i = 0
            score = 0
            column_accuracy = []
            column_grade = []
            column_list = []

            for column, value in accuracy.items():
                norm = round((value * entropy_wts[i]), 4)
                accur = norm * len(df_train.columns)
                # Cap the worst accur at 0.7, everything above that is Poor
                if accur > 0.7:
                    accur = 0.7

                # Translate accuracy into a grade and 0-10 range
                mapped_score = (0.7 - accur) / 0.7 * 10
                column_grade.append(EvaluationScore.score_to_grade(mapped_score, is_privacy=True).value)
                column_accuracy.append(mapped_score)
                column_list.append(column)
                score += mapped_score
                i += 1

            # Get overall score
            mapped_score = round((score / len(df_train.columns)), 2)
            protection = EvaluationScore.score_to_grade(mapped_score, is_privacy=True)

            col_accuracy_df = pd.DataFrame({"Column": column_list, "Risk": column_accuracy, "Protection": column_grade})
            ias = EvaluationScore(raw_score=score, grade=protection, score=round(mapped_score, 1))

        except Exception as e:
            logger.exception("Failed to calculate Attribute Inference Attack Score.")
            score = EvaluationScore(grade=PrivacyGrade.UNAVAILABLE, notes=str(e))
            return score, None

        return ias, col_accuracy_df
