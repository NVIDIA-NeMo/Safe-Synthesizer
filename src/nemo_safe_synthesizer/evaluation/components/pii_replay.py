# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable
from functools import cached_property
from numbers import Integral, Real

import pandas as pd
from pydantic import BaseModel, Field

from ...config.parameters import SafeSynthesizerParameters
from ...config.replace_pii import EntityType
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from ...pii_replacer.phone_numbers import phone_identity, phone_region

logger = get_logger(__name__)

UNKNOWN_ENTITY: str = "none"


class PIIReplayData(BaseModel):
    """Per-column PII data listed in the PII Replay section of the SQS report."""

    column_name: str = Field(description="The name of the column with PII data.")
    column_assigned_type: str = Field(
        description="The assigned type for the column (text, unique identifier, date, email, etc.)."
    )
    pii_type: str = Field(
        default=UNKNOWN_ENTITY,
        description="Type of the PII data in the column. For non-text fields, same as column_assigned_type. For text fields, the PII entities detected within the text (race, SSN, address, etc.).",
    )
    total_training_data: int = Field(default=0, description="Total rows in the training data that contain PII values.")
    unique_training_data: int = Field(
        default=0, description="Count of distinct PII values for this entity in the training column."
    )
    total_synthetic_data: int = Field(
        default=0, description="Number of synthetic rows whose column value matches a training PII value."
    )
    unique_synthetic_data: int = Field(
        default=0, description="Count of distinct training PII values that appear in the synthetic column."
    )
    unique_synthetic_data_percentage: float = Field(
        default=0,
        description="Percentage of distinct training PII values replayed in the synthetic data (unique_synthetic_data / unique_training_data * 100).",
    )


class PIIReplay(Component):
    """PII Replay metric -- counts PII values from the training data appearing in the synthetic data.

    For each classified PII entity, reports total and unique replay counts.
    This component does not produce a numeric score; it surfaces PII
    leakage details for the HTML report.
    """

    name: str = Field(default="PII Replay")
    training_total_records: int = Field(default=0, description="Total rows in the training data.")
    synthetic_total_records: int = Field(default=0, description="Total rows in the synthetic data.")
    pii_replay_data: list[PIIReplayData] = Field(
        default=list(), description="Per-column / per-entity replay statistics."
    )
    skipped_reason: str | None = Field(
        default=None, description="Why PII replay could not run, such as a failed PII detection, or None."
    )

    @cached_property
    def jinja_context(self) -> dict:
        """Template context with PII replay statistics and entity type list."""
        d = super().jinja_context
        d["training_total_records"] = self.training_total_records
        d["synthetic_total_records"] = self.synthetic_total_records
        d["pii_types"] = list({datum.pii_type for datum in self.pii_replay_data})
        d["pii_replay_data"] = [datum.model_dump() for datum in self.pii_replay_data]
        d["skipped_reason"] = self.skipped_reason
        return d

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets, config: SafeSynthesizerParameters | None = None
    ) -> PIIReplay:
        """Compute PII replay counts from classified entity metadata."""
        pii_replay_entities = config.get("pii_replay_entities") if config else None
        pii_replay_columns = config.get("pii_replay_columns") if config else None
        locale = config.get("replace_pii.locale") if config else None
        region = phone_region(locale) if isinstance(locale, str) else None
        statistics = evaluation_datasets.column_statistics or {}
        if not statistics and not pii_replay_columns:
            logger.warning("No classified entities, skipping PII Replay.")
            return PIIReplay(score=EvaluationScore())

        # Build up a list of keys of tuple(col_name, entity_type)
        classified_entities: list[tuple[str, str, str]] = []
        for col, column_statistics in statistics.items():
            entity_names = column_statistics.detected_entity_counts.keys()
            entity_assigned_type = column_statistics.assigned_type or UNKNOWN_ENTITY
            # Scope down to user supplied set of entities if there is one
            if pii_replay_entities:
                entity_names = set(entity_names).intersection(set(pii_replay_entities))
            classified_entities += [(col, entity_name, entity_assigned_type) for entity_name in entity_names]
        # Add user-specified columns without detected PII, comparing the whole column. Columns missing from
        # either dataset are skipped.
        if pii_replay_columns:
            for user_specified_col in sorted(set(pii_replay_columns).difference(statistics.keys())):
                if (
                    user_specified_col in evaluation_datasets.training.columns
                    and user_specified_col in evaluation_datasets.synthetic.columns
                ):
                    classified_entities.append((user_specified_col, UNKNOWN_ENTITY, UNKNOWN_ENTITY))

        pii_replay_data = []
        for col, entity_name, entity_assigned_type in classified_entities:
            # UNKNOWN_ENTITY case, use the entire column
            training_entity_count = len(evaluation_datasets.training[col])
            training_entity_unique_values = evaluation_datasets.training[col].unique()

            if entity_name != UNKNOWN_ENTITY:
                # Ideal case, use the count of detected entities in training for that (col, entity_type).
                # Also get the set of unique values tagged with that entity type.
                training_entity_count = statistics[col].detected_entity_counts[entity_name]
                training_entity_unique_values = statistics[col].detected_entity_values[entity_name]

            substring_match = entity_name != UNKNOWN_ENTITY and statistics[col].assigned_entity == EntityType.FREE_TEXT
            # Values that differ only in letter case, or in how a number is written, count once. So does one
            # phone number written in several ways, in a column of whole phone numbers.
            key_of = (
                _phone_replay_key(region)
                if entity_name == EntityType.PHONE_NUMBER.value and not substring_match
                else _replay_key
            )
            training_keys = {key_of(value) for value in training_entity_unique_values if not _is_missing(value)}
            training_entity_unique_count = len(training_keys)
            synthetic_col = evaluation_datasets.synthetic[col]
            synthetic_entity_count, synthetic_entity_unique_count = _synthetic_replay_counts(
                synthetic_col,
                training_keys,
                substring_match=substring_match,
                key_of=key_of,
            )
            replay_percentage = (
                synthetic_entity_unique_count / training_entity_unique_count * 100
                if training_entity_unique_count
                else 0
            )
            replay_percentage = math.ceil(replay_percentage * 10) / 10

            pii_replay_data.append(
                PIIReplayData(
                    column_name=col,
                    column_assigned_type=entity_assigned_type,
                    pii_type=entity_name,
                    total_training_data=training_entity_count,
                    unique_training_data=training_entity_unique_count,
                    total_synthetic_data=synthetic_entity_count,
                    unique_synthetic_data=synthetic_entity_unique_count,
                    unique_synthetic_data_percentage=replay_percentage,
                )
            )

        return PIIReplay(
            score=EvaluationScore(),
            training_total_records=evaluation_datasets.training.shape[0],
            synthetic_total_records=evaluation_datasets.synthetic.shape[0],
            pii_replay_data=pii_replay_data,
        )


# Text splits into maximal runs of word characters and single other characters, so "Ada's file" is
# "Ada", "'", "s", " ", "file". A replayed value always covers whole tokens; see ``_EmbeddedValueIndex``.
_TOKEN = re.compile(r"\w+|\W")
# Text of a whole number that a float column wrote with a zero fraction, such as "123456789.0".
_ZERO_FRACTION = re.compile(r"(-?\d+)\.0+")


def _synthetic_replay_counts(
    synthetic: pd.Series,
    training_keys: set[str],
    *,
    substring_match: bool,
    key_of: Callable[[object], str] | None = None,
) -> tuple[int, int]:
    """Count synthetic rows and distinct training PII values that are replayed.

    ``training_keys`` come from ``key_of``, which defaults to ``_replay_key``. Values are compared ignoring
    letter case and as text, so an SSN detected as ``"123456789"`` matches a synthetic column that reads it
    back as the integer ``123456789`` or the float ``123456789.0``, and ``KENNETH`` matches ``Kenneth``.
    Embedded values in free text are always compared by ``_replay_key``.
    """
    if not substring_match:
        key = key_of or _replay_key
        synthetic_keys = synthetic.map(lambda value: None if _is_missing(value) else key(value))
        matches = synthetic_keys[synthetic_keys.isin(training_keys)]
        return len(matches), len(matches.unique())

    index = _EmbeddedValueIndex(key for key in training_keys if key)
    replayed_values: set[str] = set()
    matching_rows = 0
    for synthetic_value in synthetic:
        if not isinstance(synthetic_value, str):
            continue
        row_matches = index.find(synthetic_value.casefold())
        if row_matches:
            matching_rows += 1
            replayed_values.update(row_matches)
    return matching_rows, len(replayed_values)


def _is_missing(value: object) -> bool:
    """Return whether a scalar cell is empty: ``None``, ``NaN``, ``pd.NA``, or ``pd.NaT``."""
    # ``numpy.float32`` is not a ``float`` subclass, so check any real number for NaN.
    return (
        value is None
        or value is pd.NA
        or value is pd.NaT
        or (isinstance(value, Real) and not isinstance(value, Integral) and math.isnan(value))
    )


def _replay_key(value: object) -> str:
    """Return the key a PII value is compared by: its replay text in case-folded form."""
    return _replay_text(value).casefold()


def _phone_replay_key(region: str | None) -> Callable[[object], str]:
    """Return a key function that compares phone numbers by their E.164 form, and other values by ``_replay_key``.

    Example:
        With region ``"US"``, ``"(555) 123-4567"``, ``"555.123.4567"``, and ``5551234567`` all have the key
        ``"+15551234567"``.
    """

    def key_of(value: object) -> str:
        text = _replay_text(value)
        return phone_identity(text, region) or text.casefold()

    return key_of


def _replay_text(value: object) -> str:
    """Return the text a whole PII value is compared by, writing whole numbers without a fraction.

    Example:
        ``123456789``, ``123456789.0``, and ``"123456789.0"`` all become ``"123456789"``.
    """
    if isinstance(value, str):
        zero_fraction = _ZERO_FRACTION.fullmatch(value)
        return zero_fraction.group(1) if zero_fraction else value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real) and math.isfinite(float(value)) and float(value).is_integer():
        return str(int(float(value)))
    return str(value)


class _EmbeddedValueIndex:
    """Find training values embedded in text where no word character extends them.

    A value matches where it is not extended by a word character at an end
    that is itself a word character, so ``Ada`` does not match ``Canada``,
    while ``-7`` matches in ``x-7`` because its first character is not a word
    character. Every such match starts and ends on a token boundary of
    ``_TOKEN``, so it is a run of whole tokens, and every run of whole tokens
    equal to a value is such a match. ``find`` therefore looks up token runs
    in a set instead of searching for each value, which takes time
    proportional to the text length rather than to the number of values.

    Example:
        With values ``{"Ada", "Ada Lovelace"}``, ``find("Ada Lovelace met Canada")``
        returns both values.
    """

    def __init__(self, values: Iterable[str]) -> None:
        self._values = frozenset(values)
        # For each possible first token, the token counts of the values that start with it.
        self._token_counts_by_first_token: dict[str, set[int]] = {}
        for value in self._values:
            tokens = _TOKEN.findall(value)
            self._token_counts_by_first_token.setdefault(tokens[0], set()).add(len(tokens))

    def find(self, text: str) -> set[str]:
        """Return the distinct values that occur in ``text``."""
        tokens = list(_TOKEN.finditer(text))
        found: set[str] = set()
        for first in range(len(tokens)):
            found.update(self._values_starting_at(text, tokens, first))
        return found

    def _values_starting_at(self, text: str, tokens: list[re.Match[str]], first: int) -> set[str]:
        """Return the values that cover whole tokens of ``text`` starting with token ``first``."""
        start = tokens[first].start()
        counts = self._token_counts_by_first_token.get(tokens[first].group(), ())
        candidates = (text[start : tokens[first + count - 1].end()] for count in counts if first + count <= len(tokens))
        return {candidate for candidate in candidates if candidate in self._values}
