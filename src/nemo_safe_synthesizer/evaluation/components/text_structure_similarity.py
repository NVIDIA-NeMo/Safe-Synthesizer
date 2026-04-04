# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from functools import cached_property

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from ...artifacts.analyzers.field_features import FieldType
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_dataset import EvaluationDataset
from ...evaluation.data_model.evaluation_field import EvaluationField
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from . import multi_modal_figures as figures

logger = get_logger(__name__)

_SENTENCE_REGEX = re.compile(r"[^\.。︀?？؟⸮!！…:।෴።။]+")
_WORD_REGEX = re.compile(r"\w+")


class TextDataSetStatistics(BaseModel):
    """Per-column text structure statistics (sentence count, word length, etc.)."""

    row_count: int = Field(
        default=0, description="Number of non-empty records analyzed (after dropping NAs and optional downsampling)."
    )
    column_count: int = Field(default=0, description="Always 1; each instance describes a single text column.")
    duplicate_lines: int = Field(
        default=0,
        description="Number of text values appearing in both reference and synthetic series. Populated on the synthetic instance only; 0 on reference.",
    )
    missing_values: int = Field(
        default=0, description="Always 0; NAs are dropped during preprocessing before statistics are computed."
    )
    unique_values: int = Field(default=0, description="Number of distinct values in the preprocessed text series.")
    per_record_statistics: pd.DataFrame = Field(
        default=pd.DataFrame(),
        description="DataFrame with per-record sentence_count, average_words_per_sentence, and average_characters_per_word.",
    )
    average_sentence_count: float = Field(default=0, description="Mean sentence count per record.")
    average_words_per_sentence: float = Field(default=0, description="Mean per-record words-per-sentence ratio.")
    average_characters_per_word: float = Field(default=0, description="Mean per-record characters-per-word ratio.")
    text_statistic_score: EvaluationScore | None = Field(
        default=None,
        description="JS-divergence-based similarity score comparing reference and synthetic structure. Populated on the synthetic instance only.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class TextStructureSimilarity(Component):
    """Text Structure Similarity metric.

    Compares per-record sentence count, words-per-sentence, and
    characters-per-word distributions between reference and output
    text columns using Jensen-Shannon divergence.
    """

    name: str = Field(default="Text Structure Similarity")
    training_statistics: dict[str, TextDataSetStatistics] = Field(
        default=dict(), description="Per-column text structure statistics for the reference data."
    )
    synthetic_statistics: dict[str, TextDataSetStatistics] = Field(
        default=dict(), description="Per-column text structure statistics for the synthetic data."
    )

    @cached_property
    def jinja_context(self) -> dict:
        """Template context with per-column text structure histogram figures."""
        d = super().jinja_context
        d["anchor_link"] = "#structure-similarity"
        d["figures"] = []
        if self.training_statistics:
            maybe_figs = [
                figures.generate_text_structure_similarity_figures(
                    self.training_statistics[col],
                    self.synthetic_statistics[col],
                    col,
                )
                for col in self.training_statistics
            ]
            d["figures"] = [
                fig.to_html(full_html=False, include_plotlyjs=False) for fig in maybe_figs if fig is not None
            ]

        return d

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> TextStructureSimilarity:
        """Compute text structure similarity across all text columns."""
        text_fields = [
            f.name for f in evaluation_dataset.evaluation_fields if f.reference_field_features.type == FieldType.TEXT
        ]

        training = evaluation_dataset.reference
        synthetic = evaluation_dataset.output
        nrows = min(len(evaluation_dataset.reference), len(evaluation_dataset.output))

        # Initialize a stub instance before trying anything.
        training_statistics_dict = dict()
        synthetic_statistics_dict = dict()

        try:
            for field in text_fields:
                try:
                    training = evaluation_dataset.reference[field]
                    synthetic = evaluation_dataset.output[field]

                    training = TextStructureSimilarity._preprocess_text_data(training, nrows)
                    synthetic = TextStructureSimilarity._preprocess_text_data(synthetic, nrows)

                    # Text statistics.
                    training_statistics = TextStructureSimilarity._get_text_statistics(training)
                    synthetic_statistics = TextStructureSimilarity._get_text_statistics(synthetic)
                    synthetic_statistics.duplicate_lines = TextStructureSimilarity._count_duplicate_lines(
                        training, synthetic
                    )

                    synthetic_statistics.text_statistic_score = TextStructureSimilarity._get_text_statistics_score(
                        training_statistics=training_statistics, synthetic_statistics=synthetic_statistics
                    )

                    training_statistics_dict[field] = training_statistics
                    synthetic_statistics_dict[field] = synthetic_statistics
                except Exception:
                    logger.exception(f"Failed to calculate Text Structure stats for field {field}.")
                    continue

            total_score = 0
            for val in synthetic_statistics_dict.values():
                total_score += val.text_statistic_score.score
            if len(synthetic_statistics_dict) > 0:
                score = total_score / len(synthetic_statistics_dict)
                text_statistic_score = EvaluationScore.finalize_grade(score, score)
            else:
                text_statistic_score = EvaluationScore()

            return TextStructureSimilarity(
                score=text_statistic_score,
                training_statistics=training_statistics_dict,
                synthetic_statistics=synthetic_statistics_dict,
            )

        except Exception as e:
            logger.exception("Failed to initialize Text Structure Similarity.")
            return TextStructureSimilarity(score=EvaluationScore(notes=str(e)))

    @staticmethod
    def _preprocess_text_data(text_data: pd.Series, nrows: int) -> pd.Series:
        """Clean and possibly downsample text data."""
        # Use first column only as a pd.Series

        # Drop na's and cast everything to string first so we are only selecting good rows.
        # We call dropna instead of adding string padding.
        text_data = text_data.dropna().astype(str)
        text_data = text_data.sample(n=min(len(text_data), nrows), random_state=333, ignore_index=True)
        return text_data

    ##
    ## text statistics
    ##

    @staticmethod
    def _get_sentence_count(text: str) -> int:
        """Count sentences in a text string using a multilingual regex.

        Args:
            text: A string of text in (almost) any language.

        Returns:
            Number of non-empty sentence segments.
        """
        return sum([1 if len(s.strip()) > 0 else 0 for s in _SENTENCE_REGEX.findall(text)])

    @staticmethod
    def _get_words(text: str) -> list[str]:
        return _WORD_REGEX.findall(text)

    @staticmethod
    def _get_average_words_per_sentence(sentence_count: int, words: list[str]) -> float:
        return len(words) / max(1, sentence_count)

    @staticmethod
    def _get_average_characters_per_word(words: list[str]) -> float:
        return float(np.mean([len(w) for w in words]))

    @staticmethod
    def _count_duplicate_lines(train: pd.Series, synth: pd.Series) -> int:
        return len(pd.merge(pd.DataFrame(train), pd.DataFrame(synth), how="inner"))

    @staticmethod
    def _get_text_statistics(text: pd.Series) -> TextDataSetStatistics:
        """Compute text structure statistics for a single text column."""
        if text is None or len(text) == 0:
            logger.error("Empty text series. Returning empty text statistics.")
            return TextDataSetStatistics()
        try:
            _data = []
            for t in text:
                _sentence_count = TextStructureSimilarity._get_sentence_count(t)
                _words = TextStructureSimilarity._get_words(t)
                _average_words_per_sentence = TextStructureSimilarity._get_average_words_per_sentence(
                    _sentence_count, _words
                )
                _average_characters_per_word = TextStructureSimilarity._get_average_characters_per_word(_words)
                _df_record = {
                    "sentence_count": _sentence_count,
                    "average_words_per_sentence": _average_words_per_sentence,
                    "average_characters_per_word": _average_characters_per_word,
                }
                _data.append(_df_record)
            _df = pd.DataFrame(data=_data)

            _row_count = len(text)
            _missing_values = text.isna().sum()
            _unique_values = len(text.unique())

            return TextDataSetStatistics(
                row_count=_row_count,
                # For now we only support 1 column.
                column_count=1,
                # We need to examine both train and synth to set this, we do so elsewhere.
                duplicate_lines=0,
                missing_values=_missing_values,
                unique_values=_unique_values,
                per_record_statistics=_df,
                average_sentence_count=float(np.mean(_df["sentence_count"])),
                average_words_per_sentence=float(np.mean(_df["average_words_per_sentence"])),
                average_characters_per_word=float(np.mean(_df["average_characters_per_word"])),
            )
        except Exception:
            logger.exception("Error calculating text statistics.")
            return TextDataSetStatistics()

    @staticmethod
    def _get_text_statistics_score(
        training_statistics: TextDataSetStatistics, synthetic_statistics: TextDataSetStatistics
    ) -> EvaluationScore:
        """Score text structure similarity for a single text column using per-record statistic distributions.

        Computes ``EvaluationField`` JS divergence for sentence count,
        words-per-sentence, and characters-per-word, then averages.

        Args:
            training_statistics: Per-record text statistics for the reference data.
            synthetic_statistics: Per-record text statistics for the synthetic data.

        Returns:
            A finalized ``EvaluationScore`` for text structure similarity.
        """
        try:
            count_df_fields = []
            for col in training_statistics.per_record_statistics.columns:
                count_df_fields.append(
                    EvaluationField.from_series(
                        col,
                        reference=training_statistics.per_record_statistics[col],
                        output=synthetic_statistics.per_record_statistics[col],
                    )
                )
            count_average_divergence = EvaluationField.get_average_divergence(count_df_fields)
            return EvaluationField.get_field_distribution_stability(
                count_average_divergence, js_scaling_func=EvaluationField.text_js_scaling_func
            )
        except Exception as e:
            logger.exception("Error calculating text statistics score.")
            return EvaluationScore(notes=str(e))
