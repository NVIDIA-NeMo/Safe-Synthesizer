# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property
from statistics import mean

import category_encoders as ce
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import ConfigDict, Field
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import QuantileTransformer

from ...artifacts.analyzers.field_features import describe_field
from ...config.evaluate import DEFAULT_RECORD_COUNT
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore, PrivacyGrade
from ...evaluation.nearest_neighbors import NearestNeighborSearch
from ...observability import get_logger
from . import multi_modal_figures as figures

logger = get_logger(__name__)


class MembershipInferenceProtection(Component):
    """Membership Inference Protection privacy metric.

    Simulates a membership inference attack: can an adversary determine
    whether a specific record was in the training set by comparing it
    to the synthetic data?  The attack is repeated across multiple
    similarity thresholds and data proportions for stability.

    See Also:
        https://arxiv.org/abs/2501.03941 -- Synthetic Data Privacy Metrics.
    """

    name: str = Field(default="Membership Inference Protection")
    attack_sum_df: pd.DataFrame | None = Field(
        default=None, description="Summary of attack outcomes by protection grade."
    )
    tps_values: dict[float, int] | None = Field(
        default=None, description="True positive counts per similarity threshold."
    )
    fps_values: dict[float, int] | None = Field(
        default=None, description="False positive counts per similarity threshold."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def jinja_context(self):
        """Template context with the membership-inference pie chart figure."""
        d = super().jinja_context
        d["anchor_link"] = "#mia"
        if self.attack_sum_df is not None and not self.attack_sum_df.empty:
            d["figure"] = figures.generate_mia_figure(df=self.attack_sum_df).to_html(
                full_html=False, include_plotlyjs=False
            )
        else:
            d["figure"] = None
        return d

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets, config: SafeSynthesizerParameters | None = None
    ) -> MembershipInferenceProtection:
        """Run the membership inference attack and return the protection score."""
        score, attack_sum_df, tps_values, fps_values = MembershipInferenceProtection.mia(
            training_df=evaluation_datasets.training,
            synthetic_df=evaluation_datasets.synthetic,
            test_df=evaluation_datasets.test,
            # FIXME config setting?
            # column_name: str | None = None,
        )
        return MembershipInferenceProtection(
            score=score, attack_sum_df=attack_sum_df, tps_values=tps_values, fps_values=fps_values
        )

    @staticmethod
    def _normalize(
        training_df: pd.DataFrame, synthetic_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.concat([training_df, synthetic_df, test_df]).reset_index(drop=True)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        df[numeric_columns] = df[numeric_columns].fillna(0)  # Fill NaNs with 0 for numeric values
        nominal_columns = []
        for column in df.columns:
            if column not in numeric_columns:
                nominal_columns.append(column)
        encoder = ce.BinaryEncoder(
            cols=nominal_columns,
            return_df=True,
            handle_missing="value",
        )
        df_norm = encoder.fit_transform(df)

        num_encoder = QuantileTransformer()
        num_encoder.fit(df_norm)
        df_norm = pd.DataFrame(num_encoder.transform(df_norm), columns=df_norm.columns).fillna(0)

        training_df_norm = df_norm.head(len(training_df))
        synthetic_df_norm = (
            df_norm.head(len(training_df) + len(synthetic_df)).tail(len(synthetic_df)).reset_index(drop=True)
        )
        test_df_norm = df_norm.tail(len(test_df)).reset_index(drop=True)

        return training_df_norm, synthetic_df_norm, test_df_norm

    @staticmethod
    def _normalize_onehot(
        training_df: pd.DataFrame, synthetic_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = pd.concat([training_df, synthetic_df, test_df]).reset_index(drop=True).fillna(0)

        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        nominal_columns = []
        for column in df.columns:
            if column not in numeric_columns:
                nominal_columns.append(column)

        df_norm = pd.get_dummies(df, columns=nominal_columns)

        num_encoder = QuantileTransformer()
        num_encoder.fit(df_norm)
        df_norm = pd.DataFrame(num_encoder.transform(df_norm), columns=df_norm.columns).fillna(0)

        training_df_norm = df_norm.head(len(training_df))
        synthetic_df_norm = df_norm.head(len(training_df) + len(synthetic_df)).tail(len(synthetic_df))
        test_df_norm = df_norm.tail(len(test_df))

        return training_df_norm, synthetic_df_norm, test_df_norm

    @staticmethod
    def _get_true_labels(training_data_indexes: list[int], attacker_data_indexes: list[int]) -> list[int]:
        true_labels = []

        for index in attacker_data_indexes:
            if index in training_data_indexes:
                true_labels.append(1)
            else:
                true_labels.append(0)

        return true_labels

    @staticmethod
    def _assess_individual_mia(true_labels: list[int], predicted_labels: list[int]):
        """Calculate precision, accuracy, true-positive, and false-positive counts."""
        precision = round(precision_score(true_labels, predicted_labels, zero_division=0), 1)
        accuracy = round(accuracy_score(true_labels, predicted_labels), 1)

        tp_cnt = 0
        fp_cnt = 0
        for i, v in enumerate(true_labels):
            if predicted_labels[i] == 1:
                if v == 1:
                    tp_cnt += 1
                else:
                    fp_cnt += 1

        return precision, accuracy, tp_cnt, fp_cnt

    @staticmethod
    def _get_attack_dist(
        attack_synth_dist_tabular: npt.NDArray,
        attack_synth_indices_text: npt.NDArray,
        attack_synth_dist_text: npt.NDArray,
        text_cnt: int,
        tabular_cnt: int,
        search_synth_k: int,
    ) -> list[np.float64]:
        """Combine text and tabular distances into an overall nearest-neighbor distance per record."""
        # Compute dist for tabular only datasets
        if tabular_cnt > 0 and text_cnt == 0:
            attack_synth_dist = []
            for i in range(len(attack_synth_dist_tabular)):
                attack_synth_dist.append(attack_synth_dist_tabular[i][0])

        # Compute dist for text only datasets
        if tabular_cnt == 0 and text_cnt > 0:
            attack_synth_dist = []
            for i in range(len(attack_synth_dist_text)):
                attack_synth_dist.append(min(attack_synth_dist_text[i]))

        # Compute dist for combined text and tabular
        if tabular_cnt > 0 and text_cnt > 0:
            # Here we have the 1000 text NN for the attack dataset and the complete set of
            # tabular attack to synth distances.  We take the 1000 text NN and reweight
            # them with the tabular similarity.  We then take the minimum of these new
            # 1000 distances and call that the nearest neighbor distance.

            attack_synth_dist = []
            text_weight = text_cnt / (text_cnt + tabular_cnt)
            tabular_weight = tabular_cnt / (text_cnt + tabular_cnt)

            for i in range(len(attack_synth_dist_tabular)):
                all_scores = []

                for j in range(search_synth_k):
                    text_dist = attack_synth_dist_text[i][j]
                    synth_index = attack_synth_indices_text[i][j]
                    tabular_dist = attack_synth_dist_tabular[i][synth_index]
                    comb_score = tabular_weight * tabular_dist + text_weight * text_dist
                    all_scores.append(comb_score)

                new_score = min(all_scores)
                attack_synth_dist.append(new_score)

        return attack_synth_dist

    @staticmethod
    def _get_grades(precision: float, accuracy: float, score: float, attack_summary: list) -> tuple[float, list]:
        """Map precision and accuracy into a cumulative score and grade label."""
        if precision <= 0.5 and accuracy <= 0.5:
            score += 3
            attack_summary.append(PrivacyGrade.EXCELLENT.value)
        elif (precision <= 0.8 and accuracy <= 0.5) or (precision <= 0.5 and accuracy <= 0.8):
            score += 2.5
            attack_summary.append(PrivacyGrade.VERY_GOOD.value)
        elif precision <= 0.8 and accuracy <= 0.8:
            score += 2
            attack_summary.append(PrivacyGrade.GOOD.value)
        elif precision <= 0.8 or accuracy <= 0.8:
            score += 1.5
            attack_summary.append(PrivacyGrade.MODERATE.value)
        else:
            score += 1
            attack_summary.append(PrivacyGrade.POOR.value)

        return score, attack_summary

    @staticmethod
    def _compute_mia(
        training_df_norm: pd.DataFrame,
        synthetic_df_norm: pd.DataFrame,
        test_df_norm: pd.DataFrame,
        nn_index: NearestNeighborSearch | None,
        run: int,
        text_cnt: int,
        tabular_cnt: int,
    ) -> tuple[
        float,
        list[str],
        dict[str, list[int]],
        dict[str, list[int]],
    ]:
        """Core membership inference attack implementation for a single run.

        Builds an attack dataset from a slice of training rows mixed with
        test rows, computes nearest-neighbor distances to the synthetic
        data (text via semantic search, tabular via L2 nearest neighbor), and
        classifies each record as member or non-member.

        Args:
            training_df_norm: Normalized training dataframe.
            synthetic_df_norm: Normalized synthetic dataframe.
            test_df_norm: Normalized holdout (test) dataframe.
            nn_index: Pre-built NearestNeighborSearch index over the tabular columns of
                the synthetic data, or ``None`` if no tabular columns exist.
            run: Zero-based run index controlling which training slice to use.
            text_cnt: Number of text columns in the dataset.
            tabular_cnt: Number of tabular columns in the dataset.

        Returns:
            Tuple of (attack score, grade labels, true label dict,
            predicted label dict).
        """
        # For multimodal we will first get the 1000 NN for each attack record using the text embedding
        # We then adjust these scores by the tabular distance and then the min of all these score
        # is the nearest neighbor distance
        search_synth_k = min(1000, len(synthetic_df_norm))

        # Gather the attack data
        prefix = len(test_df_norm) * (run + 1)
        prefix_head = training_df_norm.head(prefix)
        training_df_attack = prefix_head.tail(len(test_df_norm)).reset_index(drop=True)
        training_data_indexes = training_df_attack.index.tolist()
        attack_df = (
            pd.concat([training_df_attack, test_df_norm]).reset_index(drop=True).sample(frac=1, random_state=run)
        )

        attack_synth_dist_text = [[0] for i in range(len(attack_df))]
        attack_synth_indices_text = [[0] for i in range(len(attack_df))]

        # Get the NN dist for text for the entire attack dataset

        if text_cnt > 0:
            # For multimodal we get the 1000 nearest neighbors, else if dataset is all text we
            # just get the one NN
            if tabular_cnt > 0:
                k = search_synth_k
            else:
                k = 1
            hits = util.semantic_search(
                np.array(list(attack_df["embedding"])),  # ty: ignore[invalid-argument-type]
                np.array(list(synthetic_df_norm["embedding"])),  # ty: ignore[invalid-argument-type]
                top_k=k,
            )
            for i in range(len(attack_df)):
                all_dist = []
                all_indices = []
                for j in range(k):
                    sim = hits[i][j]["score"]
                    corpus_id = hits[i][j]["corpus_id"]
                    dist = 1 - sim
                    all_dist.append(dist)
                    all_indices.append(corpus_id)
                attack_synth_dist_text[i] = all_dist
                attack_synth_indices_text[i] = all_indices

        # Gather the dist from attack to synth for tabular.  For multimodal, we are gathering the
        # distance between every record in the attack dataset and every record in the synth dataset.

        attack_synth_dist_tabular = np.zeros((len(attack_df), len(synthetic_df_norm)))

        if tabular_cnt > 0:
            if text_cnt > 0:
                attacker_data_tabular = attack_df.drop(["embedding"], axis=1)
                k = len(synthetic_df_norm)
            else:
                attacker_data_tabular = attack_df.copy()
                k = 1

            if nn_index is None:
                raise RuntimeError("Nearest neighbor index not provided for MIA calculation when expected.")

            # Use nearest neighbor search (torch GPU or sklearn CPU fallback) for distance calculation
            dists, indices = nn_index.kneighbors(
                np.ascontiguousarray(np.array(attacker_data_tabular)).astype(np.float32),
                n_neighbors=int(k),
            )
            # Scale the Euclidean distance to [0,1]
            # NearestNeighborSearch.kneighbors() returns L2 distance directly, not squared
            max_dist = np.amax(dists)
            if max_dist > 0:
                dist_scaled = dists / max_dist
            else:
                dist_scaled = dists

            for i in range(len(attacker_data_tabular)):
                if text_cnt > 0:
                    for j in range(k):
                        attack_synth_dist_tabular[i][indices[i][j]] = dist_scaled[i][j]
                else:
                    attack_synth_dist_tabular[i] = dist_scaled[i]

        true_labels = MembershipInferenceProtection._get_true_labels(training_data_indexes, attack_df.index.tolist())

        # We repeat MIA for different similarity thresholds and
        # different sized attack datasets
        thresholds = [0.4, 0.3, 0.2, 0.1]
        proportions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        total_attacks = 0
        score = 0
        attack_summary = []

        tp_cnts = {}
        fp_cnts = {}

        # Using the above text and tabular distances we now compute an overall distance score for
        # every record in the attack dataset. We then conduct 36 individual mia attacks on this one big
        # attack dataset each time varying the threshold and proportion used. We no longer have to compute
        # distance for each of these 36 runs because we already have it for the whole attack dataset.

        attack_synth_dist = MembershipInferenceProtection._get_attack_dist(
            attack_synth_dist_tabular,
            attack_synth_indices_text,  # ty: ignore[invalid-argument-type]
            attack_synth_dist_text,  # ty: ignore[invalid-argument-type]
            text_cnt,
            tabular_cnt,
            search_synth_k,
        )

        for threshold in thresholds:
            tp_cnts[threshold] = 0
            fp_cnts[threshold] = 0

            for proportion in proportions:
                # Get the final attack data dist for some specific proportion
                attack_synth_dist_use = attack_synth_dist[0 : int(len(attack_synth_dist) * proportion)]

                # Gather the predicted and true labels for this subset/proportion
                predicted_labels = []
                true_labels_use = []
                for i in range(len(attack_synth_dist_use)):
                    true_labels_use.append(true_labels[i])
                    if attack_synth_dist_use[i] < threshold:
                        predicted_labels.append(1)
                    else:
                        predicted_labels.append(0)

                precision, accuracy, tp_cnt, fp_cnt = MembershipInferenceProtection._assess_individual_mia(
                    true_labels_use, predicted_labels
                )

                tp_cnts[threshold] += tp_cnt
                fp_cnts[threshold] += fp_cnt

                # Translate the precision and accuracy into score
                total_attacks += 1
                score, attack_summary = MembershipInferenceProtection._get_grades(
                    precision, accuracy, score, attack_summary
                )

        # The raw score is the average over all attacks
        raw_score = round(score / total_attacks, 2)

        return (
            raw_score,
            attack_summary,
            tp_cnts,
            fp_cnts,
        )

    @staticmethod
    def find_text_fields(df: pd.DataFrame) -> list[str]:
        """Return column names classified as free text."""
        text_fields = []
        for col in df.columns:
            field_info = describe_field(col, df[col])
            if field_info.type.value == "text":
                text_fields.append(col)

        return text_fields

    @staticmethod
    def embed_text(df: pd.DataFrame) -> pd.DataFrame:
        """Embed each text column and average into a single embedding per row."""
        embeddings = {}
        embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")
        for col in df.columns:
            data = df[col].to_list()
            data = [str(r) for r in data]
            embeddings[col] = embedder.encode(data, show_progress_bar=False, convert_to_numpy=True)

        avg_embeddings = []
        for i in range(len(df)):
            # TODO: Is this average what we want? When there are more than 2 columns, we will
            # overweight later columns relative to earlier columns.
            norm = embeddings[df.columns[0]][i]
            for j in range(1, len(df.columns)):
                field = df.columns[j]
                norm = np.average([norm, embeddings[field][i]], axis=0)

            avg_embeddings.append(norm)

        df_embeddings = pd.DataFrame({"embedding": list(avg_embeddings)})

        return df_embeddings

    @staticmethod
    def divide_tabular_text(df: pd.DataFrame, text_fields: list) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a dataframe into tabular-only and text-only subsets."""
        tabular_fields = []
        for col in df.columns:
            if col not in text_fields:
                tabular_fields.append(col)
        df_tabular = df.filter(tabular_fields)
        df_text = df.filter(text_fields)

        return (df_tabular, df_text)

    @staticmethod
    def mia(
        training_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        test_df: pd.DataFrame | None,
        column_name: str | None = None,
    ) -> tuple[
        EvaluationScore,
        pd.DataFrame | None,
        dict[float, int],
        dict[float, int],
    ]:
        """Run the full membership inference attack pipeline.

        Normalizes data, builds FAISS indexes and/or text embeddings, then
        repeats the attack across multiple runs for stability. The final score
        is the average across all runs, mapped to a 0--10 privacy grade.

        Args:
            training_df: Training dataframe.
            synthetic_df: Synthetic dataframe.
            test_df: Holdout dataframe (required -- returns unavailable if ``None``).
            column_name: Optional single column to restrict the attack to.

        Returns:
            Tuple of (score, attack summary dataframe, TP counts, FP counts).
        """
        ias = EvaluationScore(grade=PrivacyGrade.UNAVAILABLE)
        attack_sum_df = None
        tps_values = {}
        fps_values = {}
        if test_df is None:
            logger.info("No test data provided for Membership Inference Attack. Skipping Membership Inference Attack.")
            return ias, attack_sum_df, tps_values, fps_values

        try:
            # If user entered column_name, reduce dataframes down to that field
            if column_name:
                training_df = training_df.filter([column_name])
                test_df = test_df.filter([column_name])
                synthetic_df = synthetic_df.filter([column_name])

            text_fields = MembershipInferenceProtection.find_text_fields(training_df)
            text_cnt = len(text_fields)
            tabular_cnt = len(training_df.columns) - text_cnt

            # For multimodal we limit the test size to DEFAULT_RECORD_COUNT
            if text_cnt > 0 and tabular_cnt > 0:
                if len(test_df) > DEFAULT_RECORD_COUNT:
                    test_df = test_df.sample(n=DEFAULT_RECORD_COUNT, random_state=2)

            # Repeat MIA for stability
            repeat_count = 10
            # Sampling what we need for all MIA runs upfront speeds things up
            training_size_needed = len(test_df) * repeat_count
            training_df_use = training_df.copy()
            training_df_use.columns = training_df.columns
            if len(training_df_use) > training_size_needed:
                training_df_use = training_df.sample(n=training_size_needed, random_state=1)

            # Divide the dataframes into text and tabular
            text_fields = MembershipInferenceProtection.find_text_fields(training_df_use)
            if len(text_fields) > 0:
                training_df_use, training_df_text = MembershipInferenceProtection.divide_tabular_text(
                    training_df_use, text_fields
                )
                test_df, test_df_text = MembershipInferenceProtection.divide_tabular_text(test_df, text_fields)
                synthetic_df, synthetic_df_text = MembershipInferenceProtection.divide_tabular_text(
                    synthetic_df, text_fields
                )

            # Normalize the tabular data (adjusted for multimodal)
            if tabular_cnt > 0:
                try:
                    training_df_norm, synthetic_df_norm, test_df_norm = MembershipInferenceProtection._normalize(
                        training_df_use, synthetic_df, test_df
                    )
                except Exception:
                    training_df_norm, synthetic_df_norm, test_df_norm = MembershipInferenceProtection._normalize_onehot(
                        training_df_use, synthetic_df, test_df
                    )
                # Create nearest neighbor index on the synthetic tabular data (torch GPU or sklearn CPU fallback)
                nn_index = NearestNeighborSearch(n_neighbors=len(synthetic_df_norm))
                nn_index.fit(np.ascontiguousarray(np.array(synthetic_df_norm)).astype(np.float32))
            else:
                training_df_norm = pd.DataFrame()
                test_df_norm = pd.DataFrame()
                synthetic_df_norm = pd.DataFrame()
                nn_index = None

            # Create embeddings for text fields and combine the normalized tabular and the
            # new text embeddings into one dataframe.
            if len(text_fields) > 0:
                training_df_embeddings = MembershipInferenceProtection.embed_text(training_df_text)
                synthetic_df_embeddings = MembershipInferenceProtection.embed_text(synthetic_df_text)
                test_df_embeddings = MembershipInferenceProtection.embed_text(test_df_text)
                training_df_norm = pd.concat([training_df_norm, training_df_embeddings], axis=1)
                synthetic_df_norm = pd.concat([synthetic_df_norm, synthetic_df_embeddings], axis=1)
                test_df_norm = pd.concat([test_df_norm, test_df_embeddings], axis=1)

            scores = []
            attack_sum_values = []
            tps_values = {}
            fps_values = {}
            for i in [0.1, 0.2, 0.3, 0.4]:
                tps_values[i] = 0
                fps_values[i] = 0

            for i in range(repeat_count):
                score, attack_sum, tp_cnts, fp_cnts = MembershipInferenceProtection._compute_mia(
                    training_df_norm,
                    synthetic_df_norm,
                    test_df_norm,
                    nn_index,
                    i,
                    text_cnt,
                    tabular_cnt,
                )
                for tp_cnt in tp_cnts:
                    tps_values[tp_cnt] += tp_cnts[tp_cnt]
                for fp_cnt in fp_cnts:
                    fps_values[fp_cnt] += fp_cnts[fp_cnt]
                scores.append(score)
                attack_sum_values = attack_sum_values + attack_sum

            values = {}
            for grade in PrivacyGrade:
                values[grade.value] = 0

            total = 0
            for value in attack_sum_values:
                total += 1
                values[value] += 1

            for i in values:
                values[i] = int((values[i] / total) * 100)

            attack_sum_df = pd.DataFrame(
                {
                    "Protection": values.keys(),
                    "Attack Percentage": values.values(),
                }
            )

            # The final score is the average over all MIA runs
            score_avg = mean(scores)

            # Translate score to 0 to 10 range
            final_score = round(((score_avg - 1) / 2 * 10), 1)

            # Translate score to overall grade
            grade = EvaluationScore.score_to_grade(final_score, is_privacy=True)

            ias = EvaluationScore(raw_score=score_avg, grade=grade, score=final_score)
        except Exception as e:
            logger.exception("Failed to calculate Membership Inference Attack Score.")
            ias = EvaluationScore(notes=str(e))

        return (
            ias,
            attack_sum_df,
            tps_values,
            fps_values,
        )
