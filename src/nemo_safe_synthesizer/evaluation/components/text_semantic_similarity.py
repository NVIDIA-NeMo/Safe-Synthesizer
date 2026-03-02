# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from functools import cached_property

import numpy as np
import numpy.typing as npt
import pandas as pd
from numpy.linalg import norm
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import ks_2samp
from sentence_transformers import SentenceTransformer
from tenacity import (
    RetryError,
    Retrying,
    before_sleep_log,
    stop_after_attempt,
    wait_exponential,
)

from ...artifacts.analyzers.field_features import FieldType
from ...config.evaluate import DEFAULT_RECORD_COUNT
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.constants import (
    MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS,
    MIN_RECORDS_FOR_TEXT_METRICS_WITHOUT_WARNING,
)
from ...evaluation.data_model.evaluation_dataset import EvaluationDataset
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...evaluation.statistics import stats
from ...observability import get_logger
from . import multi_modal_figures as figures

logger = get_logger(__name__)


class TextSemanticSimilarityDatum(BaseModel):
    """Per-column text semantic similarity scores and PCA projections."""

    text_semantic_similarity: EvaluationScore = Field(default=EvaluationScore())
    text_semantic_similarity_underfitting_factor: EvaluationScore = Field(default=EvaluationScore())
    text_semantic_similarity_overfitting_factor: EvaluationScore = Field(default=EvaluationScore())

    training_pca: pd.DataFrame = Field(default=pd.DataFrame())
    synthetic_pca: pd.DataFrame = Field(default=pd.DataFrame())

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextSemanticSimilarity(Component):
    """Text Semantic Similarity metric.

    Embeds text columns with a sentence transformer and compares the
    distribution of cosine similarities between reference-synthetic and
    reference-reference (or test-test) pairs using two-sided
    Kolmogorov-Smirnov tests to capture both underfitting and overfitting.
    """

    name: str = Field(default="Text Semantic Similarity")
    text_semantic_similarity_dict: dict[str, TextSemanticSimilarityDatum] = Field(default=dict())

    @cached_property
    def jinja_context(self):
        """Template context with per-column PCA scatter figures."""
        ctx = super().jinja_context
        ctx["anchor_link"] = "#semantic-similarity"
        ctx["figures"] = []
        if self.text_semantic_similarity_dict:
            maybe_figs = [
                figures.generate_text_semantic_similarity_figures(
                    self.text_semantic_similarity_dict[col].training_pca.iloc[:, :2],
                    self.text_semantic_similarity_dict[col].synthetic_pca.iloc[:, :2],
                    col,
                )
                for col in self.text_semantic_similarity_dict.keys()
            ]
            figs = [fig.to_html(full_html=False, include_plotlyjs=False) for fig in maybe_figs if fig is not None]
            ctx["figures"] = figs

        return ctx

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> TextSemanticSimilarity:
        """Compute text semantic similarity across all text columns."""
        if evaluation_dataset.test is None or evaluation_dataset.test.empty:
            return TextSemanticSimilarity(
                score=EvaluationScore(
                    notes="Unable to calculate Text Semantic Similarity. No holdout dataframe provided."
                )
            )

        nrows = min(
            len(evaluation_dataset.reference), len(evaluation_dataset.output)
        )  # MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS ?

        text_semantic_similarity_dict = dict()
        text_fields = [
            f.name for f in evaluation_dataset.evaluation_fields if f.reference_field_features.type == FieldType.TEXT
        ]

        for field in text_fields:
            training = evaluation_dataset.reference[field]
            synthetic = evaluation_dataset.output[field]
            test = evaluation_dataset.test[field]

            # Initialize a stub instance before trying anything.
            text_semantic_similarity = EvaluationScore()
            text_semantic_similarity_underfitting_factor = EvaluationScore()
            text_semantic_similarity_overfitting_factor = EvaluationScore()
            training_pca = pd.DataFrame()
            synthetic_pca = pd.DataFrame()

            try:
                stm = TextSemanticSimilarity._init_sentence_transformer_model()
                if stm is None:
                    raise RuntimeError("Sentence Transformer Model is None, unable to continue.")

                if nrows is None:
                    nrows = DEFAULT_RECORD_COUNT
                # PLAT-914 Easter egg, 0 rows means use everything.
                if nrows == 0:
                    nrows = max(len(training), len(synthetic))
                    if test is not None:
                        nrows = max(nrows, len(test))

                training = TextSemanticSimilarity._preprocess_text_data(training, nrows)
                synthetic = TextSemanticSimilarity._preprocess_text_data(synthetic, nrows)

                training_embedding_vector = TextSemanticSimilarity._get_embedding_vectors(training, stm)
                synthetic_embedding_vector = TextSemanticSimilarity._get_embedding_vectors(synthetic, stm)

                test_embedding_vector = None
                warning_message = None
                if test is not None and not test.empty:
                    # Only calculate the semantic similarity score if we have a test set.
                    # And only if there at least 100 total input records
                    test = TextSemanticSimilarity._preprocess_text_data(test, nrows)
                    total_input_records = len(training) + len(test)
                    if total_input_records >= MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS:
                        test_embedding_vector = TextSemanticSimilarity._get_embedding_vectors(test, stm)
                        # TODO: Use dynamic calculation based on training/test/synthetic data sizes
                        # to determine if we should log this warning.
                        if total_input_records < MIN_RECORDS_FOR_TEXT_METRICS_WITHOUT_WARNING:
                            warning_message = f"Warning: Consider using at least {MIN_RECORDS_FOR_TEXT_METRICS_WITHOUT_WARNING} input records for a more accurate semantic similarity score."
                    else:
                        warning_message = (
                            f"Not enough input records for text semantic similarity score. "
                            f"Need at least {MIN_RECORDS_FOR_TEXT_AND_PRIVACY_METRICS} non-empty records. Skipping text semantic similarity."
                        )
                else:
                    warning_message = "No test data provided, skipping text semantic similarity."

                if test_embedding_vector is not None:
                    (
                        text_semantic_similarity,
                        text_semantic_similarity_underfitting_factor,
                        text_semantic_similarity_overfitting_factor,
                    ) = TextSemanticSimilarity._get_text_semantic_similarity(
                        real_embed=training_embedding_vector,
                        synth_embed=synthetic_embedding_vector,
                        test_embed=test_embedding_vector,
                        warning_message=warning_message,
                    )
                else:
                    (
                        text_semantic_similarity,
                        text_semantic_similarity_underfitting_factor,
                        text_semantic_similarity_overfitting_factor,
                    ) = (EvaluationScore(notes=warning_message),) * 3

                if warning_message:
                    logger.info(warning_message)

                # I'm PCA I've got nothing to prove pay attention my intention is to bust a move.
                training_pca, synthetic_pca = TextSemanticSimilarity._get_pca(
                    pd.DataFrame(training_embedding_vector),
                    pd.DataFrame(synthetic_embedding_vector),
                )

                text_semantic_similarity_dict[field] = TextSemanticSimilarityDatum(
                    text_semantic_similarity=text_semantic_similarity,
                    text_semantic_similarity_underfitting_factor=text_semantic_similarity_underfitting_factor,
                    text_semantic_similarity_overfitting_factor=text_semantic_similarity_overfitting_factor,
                    training_pca=training_pca,
                    synthetic_pca=synthetic_pca,
                )
            except Exception:
                logger.exception("Failed to initialize TextSemanticSimilarity.")

        score_numerator = 0
        score_denominator = 0
        for v in text_semantic_similarity_dict.values():
            if v.text_semantic_similarity.score is not None:
                score_numerator += v.text_semantic_similarity.score
                score_denominator += 1
        if score_denominator == 0:
            return TextSemanticSimilarity(
                score=EvaluationScore(), text_semantic_similarity_dict=text_semantic_similarity_dict
            )
        else:
            score = score_numerator / score_denominator
            return TextSemanticSimilarity(
                score=EvaluationScore.finalize_grade(score, score),
                text_semantic_similarity_dict=text_semantic_similarity_dict,
            )

    @staticmethod
    def _preprocess_text_data(text_data: pd.Series, nrows: int) -> pd.Series:
        """Helper function to clean and possibly downsample text data."""
        # Use first column only as a pd.Series
        # Longwinded NOTE:
        # take a df with multiple columns and sniff out the best one.

        # Drop na's and cast everything to string first so we are only selecting good rows.
        # We call dropna instead of adding string padding
        text_data = text_data.dropna().astype(str)
        text_data = text_data.sample(n=min(len(text_data), nrows), random_state=333, ignore_index=True)
        return text_data

    ##
    ## text_semantic_similarity
    ##
    @staticmethod
    def _init_sentence_transformer_model() -> SentenceTransformer | None:
        """Load the sentence transformer model with exponential-backoff retries."""
        try:
            for attempt in Retrying(
                # TODO(PLAT-2537): Temporarily increase retries, but we will bundle this model next, so download is not required.
                # Retry: 10 times, start with 1 second and increase by 2x each time, until 120 secs.
                # Retry timings: 1, 2, 4, 8, 16, 32, 64, 120, 120, 120. Total wait max: 367 seconds.
                stop=stop_after_attempt(10),
                wait=wait_exponential(max=120),
                before_sleep=before_sleep_log(logger, logging.WARNING),
            ):
                with attempt:
                    return SentenceTransformer("distiluse-base-multilingual-cased-v2")
        except RetryError:
            return None

    @staticmethod
    def _get_embedding_vectors(
        text_data: pd.Series, sentence_transformer_model: SentenceTransformer
    ) -> npt.NDArray[np.single]:
        """Generate embedding vectors for a text series.

        Args:
            text_data: Text column to embed.
            sentence_transformer_model: Pre-loaded sentence transformer.

        Returns:
            2-D array of shape ``(len(text_data), embedding_dim)``.
        """
        try:
            return sentence_transformer_model.encode(
                text_data.values,
                normalize_embeddings=True,
            )  # ty: ignore[no-matching-overload]
        except Exception:
            logger.exception("Error getting embedding vector.")
            return np.ndarray(shape=0)

    @staticmethod
    def _average_embedded_vectors(
        embedded_vector: npt.NDArray[np.single],
    ) -> npt.NDArray[np.single]:
        """Average embedding vectors across all records.

        Args:
            embedded_vector: 2-D array of shape ``(n_records, embedding_dim)``.

        Returns:
            1-D array of shape ``(embedding_dim,)``.
        """
        try:
            return np.mean(embedded_vector, axis=0)
        except Exception:
            logger.exception("Error getting average embedded vector.")
            return np.ndarray(shape=0)

    @staticmethod
    def _get_cosine_similarity(real_mean: npt.NDArray[np.single], synth_mean: npt.NDArray[np.single]) -> float:
        """Compute cosine similarity between two mean embedding vectors.

        Args:
            real_mean: Mean embedding of the real (reference) data.
            synth_mean: Mean embedding of the synthetic data.

        Returns:
            Cosine similarity in ``[-1, 1]``, or ``0`` on failure.
        """
        try:
            denom = norm(real_mean) * norm(synth_mean)
            if denom == 0:
                return 0
            else:
                # Cast to plain float to avoid 'Object of type float32 is not JSON serializable'
                return float(np.dot(real_mean, synth_mean) / denom)
        except Exception:
            logger.exception("Error getting cosine similarity.")
            return 0

    @staticmethod
    def _scale_semantic_similarity(sem_sim: float) -> float | None:
        """Scale a 0--1 semantic similarity value to the 0--10 score range.

        No curving is applied; practically the scores fall between 3.7
        (worst) and 10 (best).
        """
        try:
            return sem_sim * 10

        except Exception:
            logger.exception("Error scaling cosine similarity.")
            return None

    @staticmethod
    def _get_text_semantic_similarity(
        real_embed: npt.NDArray[np.single],
        synth_embed: npt.NDArray[np.single],
        test_embed: npt.NDArray[np.single],
        warning_message: str | None = None,
    ) -> tuple[EvaluationScore, EvaluationScore, EvaluationScore]:
        """
        Compute the semantic similarity between the real and synthetic
        embeddings (normalized to have length 1). The metric is based on
        two one-sided Kolmogorov-Smirnov tests:

        **Overfitting (KS "less")** -- detects whether synthetic samples are
        *more* similar to the training data than the training data is to
        itself (i.e. memorisation):
          - F(x): for each synthetic sample, the max cosine similarity to
            any training sample.
          - G(x): for each training sample, the max cosine similarity to
            any other training sample (self-similarity excluded via zeroed
            diagonal).

        **Underfitting (KS "greater")** -- detects whether synthetic samples
        are *less* similar to the held-out test data than the test data is
        to itself (i.e. poor generalisation):
          - F(x): for each synthetic sample, the max cosine similarity to
            any test sample.
          - G(x): for each test sample, the max cosine similarity to any
            other test sample (self-similarity excluded via zeroed
            diagonal).

        Each KS statistic (in [0, 1]) is mapped to a factor via
        ``exp(-statistic)`` (in [exp(-1), 1] ≈ [0.37, 1]), and the two
        factors are multiplied to produce the final raw score, which is
        then rescaled to [0, 10] and graded.
        """
        # Compare distributions

        # Calculate cosine similarity using the dot product (equivalent as
        # embeddings are normalized)
        real_similarity_matrix = real_embed @ real_embed.T
        for i in range(len(real_similarity_matrix)):
            real_similarity_matrix[i, i] = 0
        real_synth_similarity_matrix = real_embed @ synth_embed.T

        test_similarity_matrix = test_embed @ test_embed.T
        for i in range(len(test_similarity_matrix)):
            test_similarity_matrix[i, i] = 0
        test_synth_similarity_matrix = test_embed @ synth_embed.T

        # KS tests:
        # Using the train set to calculate the overfitting metric and the test
        # set to calculate the underfitting metric. Overfitting is measured as
        # the extent the synthetic data is more similar to the training data than
        # the training data is to itself. Underfitting is measured as the extent
        # the synthetic data is less similar to the test data than the test data
        # is to itself.

        # The null hypothesis is that F(x) >= G(x) for all x; the alternative is
        # that F(x) < G(x) for at least one x. The statistic is the magnitude of
        # the minimum (most negative) difference between the empirical
        # distribution functions of the samples. The range of this statistic is
        # [0, 1], where 0 indicates no overfitting.
        ks_test_overfitting = ks_2samp(
            real_synth_similarity_matrix.max(axis=0),  # F(x)
            real_similarity_matrix.max(axis=0),  # G(x)
            alternative="less",
            method="auto",
        )

        # The null hypothesis is that F(x) <= G(x) for all x; the alternative is
        # that F(x) > G(x) for at least one x. The statistic is the maximum
        # (most positive) difference between the empirical distribution
        # functions of the samples. The range of this statistic is [0, 1].
        ks_test_underfitting = ks_2samp(
            test_synth_similarity_matrix.max(axis=0),  # F(x)
            test_similarity_matrix.max(axis=0),  # G(x)
            alternative="greater",
            method="auto",
        )

        # The overall semantic simlarity score combines underfitting and overfitting
        # The range of this score is [0.37, 1], where 1 indicates perfect model and
        # exp(-1) = 0.37 indicates extreme underfitting or overfitting.
        # The raw score is the product of the underfitting and overfitting factors.
        underfitting_factor = np.exp(-ks_test_underfitting.statistic)
        overfitting_factor = np.exp(-ks_test_overfitting.statistic)
        text_semantic_similarity_raw = underfitting_factor * overfitting_factor

        text_semantic_similarity = EvaluationScore.finalize_grade(
            raw_score=text_semantic_similarity_raw,
            score=TextSemanticSimilarity._scale_semantic_similarity(text_semantic_similarity_raw),
        )
        text_semantic_similarity.notes = warning_message

        text_semantic_similarity_underfitting_factor = EvaluationScore.finalize_grade(
            raw_score=underfitting_factor, score=TextSemanticSimilarity._scale_semantic_similarity(underfitting_factor)
        )
        text_semantic_similarity_underfitting_factor.notes = warning_message

        text_semantic_similarity_overfitting_factor = EvaluationScore.finalize_grade(
            raw_score=overfitting_factor, score=TextSemanticSimilarity._scale_semantic_similarity(overfitting_factor)
        )
        text_semantic_similarity_overfitting_factor.notes = warning_message

        try:
            return (
                text_semantic_similarity,
                text_semantic_similarity_underfitting_factor,
                text_semantic_similarity_overfitting_factor,
            )
        except Exception:
            logger.exception("Failed to scale and finalize text semantic similarity.")
        return EvaluationScore(), EvaluationScore(), EvaluationScore()

    ##
    ## PCA
    ##

    @staticmethod
    def _get_pca(
        reference: pd.DataFrame, output: pd.DataFrame, n_components: int = 4
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute joined PCA on text embedding dataframes.

        Args:
            reference: Training text embeddings (all numeric).
            output: Synthetic text embeddings (all numeric).
            n_components: Number of principal components to keep.

        Returns:
            Tuple of (reference PCA dataframe, output PCA dataframe).
        """
        try:
            reference = reference.replace([np.inf, -np.inf], np.nan).dropna(axis="columns", how="all")
            output = output.replace([np.inf, -np.inf], np.nan).dropna(axis="columns", how="all")

            reference_pca, output_pca = stats.compute_joined_pcas(
                reference,
                output,
                n_components=n_components,
                include_variance=True,
            )

            reference_pca["data"] = "train"
            output_pca["data"] = "synthetic"

            return (reference_pca, output_pca)
        except Exception:
            logger.exception("Error calculating PCA.")
            return pd.DataFrame(), pd.DataFrame()
