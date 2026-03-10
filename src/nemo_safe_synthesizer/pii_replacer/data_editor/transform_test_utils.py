# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from ..ner.ner import NERPrediction
from .detect import (
    UNKNOWN_ENTITY,
    ColumnClassifier,
    EntityExtractorGliner,
    IAPIClassifierConfig,
)


class EntityExtractorMock(EntityExtractorGliner):
    """Mock GLiNER-based extractor returning fixed NER results for tests.

    Replaces model calls with a hardcoded list of entities for a known input string.
    Use ``get_entity_extractor`` to construct; no constructor args.

    Attributes:
        extract_ner_predictions_called: Set to ``True`` when ``extract_ner_predictions`` is called.
    """

    extract_ner_predictions_called: bool

    @classmethod
    def get_entity_extractor(cls, *args, **kwargs) -> EntityExtractorMock | None:
        self = cls()
        self.extract_ner_predictions_called = False
        self._chunk_length = 512
        self._chunk_overlap = 128
        self._entity_cache = {}
        self._batch_size = 1000
        self._entity_types = ["test", "entity", "types"]
        self._model = MagicMock()
        self._ner_threshold = 0.9
        self._batch_mode_enabled = False
        return self

    def _detect_entities_chunked(
        self, text: str, entities: set[str] | None = None
    ) -> list[dict]:
        """Return fixed entity dicts for the known test string (first_name, ssn, nofake).

        Input is assumed to match the string used in transform tests; indices are
        for that string. Ignores ``entities``.
        """
        return [
            {"label": "first_name", "text": "joe", "start": 17, "end": 20},
            {"label": "ssn", "text": "123-12-1234", "start": 32, "end": 43},
            {"label": "nofake", "text": "Unfake-able", "start": 45, "end": 56},
        ]

    def extract_ner_predictions(self, text: str, entities: set[str] | None) -> list[NERPrediction]:
        """Return fixed NER predictions and set ``extract_ner_predictions_called`` to ``True``."""
        self.extract_ner_predictions_called = True
        return [
            NERPrediction(e["text"], e["start"], e["end"], e["label"], "GLiNER", 9.0)
            for e in self._detect_entities_chunked(text, entities)
        ]


class ColumnClassifierMock(ColumnClassifier):
    """Classifier with no backend; uses a hardcoded column-to-entity mapping for tests."""

    _num_samples: int | None

    @classmethod
    def get_iapi_classifier(cls, config: IAPIClassifierConfig) -> ColumnClassifierMock | None:
        """Return a mock classifier using ``config.num_samples``."""
        classifier = cls()
        classifier._num_samples = config.num_samples
        return classifier

    @classmethod
    def get_deployed_llm_classifier(
        cls,
        llm_endpoint: str,
        num_samples: int,
    ) -> ColumnClassifierMock | None:
        """Return a mock classifier with the given ``num_samples`` (endpoint unused)."""
        classifier = cls()
        classifier._num_samples = num_samples
        return classifier

    def detect_types(self, df: pd.DataFrame, all_entities: set[str]) -> dict[str, str | None]:
        """Return a hardcoded column-to-entity map for known test columns.

        Only columns present in the internal mapping and in ``all_entities``
        get a non-``UNKNOWN_ENTITY`` value; ``ghost_entity`` is always included
        in ``all_entities`` for integration tests.

        Args:
            df: DataFrame whose column names are used as keys.
            all_entities: Set of entity names to allow; mapping is filtered to these.

        Returns:
            Map of column name to entity name (or ``UNKNOWN_ENTITY``).
        """
        entities: dict[str, str] = {
            "AddressLine1": "street_address",
            "AddressLine1a": "street_address",
            "AddressLine1b": "street_address",
            "AddressLine3": "city",
            "AddressLine4": "administrative_unit",
            "AddressLine6": "country",
            "AddressLine7": "continent",
            "Hallucination": "ghost_entity",
            "fname": "first_name",
            "lname": "last_name",
        }

        # Unused, but allow for exercise within integration test.
        # TODO: What does "allow for exercise within integration test" mean? Commenting out
        # for now for linting.
        # columns = sample_columns(df, self._num_samples)

        # Always see something that isn't really there..
        all_entities = all_entities | {"ghost_entity"}
        entities = {col: entities[col] for col in entities if entities[col] in all_entities}
        return {col: entities.get(col, UNKNOWN_ENTITY) for col in df.columns}
