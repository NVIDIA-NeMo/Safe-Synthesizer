# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional
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
    """
    Mock EntityExtractor; Override GLiNER model interactions
    with known values for specific text string.
    """

    extract_ner_predictions_called: bool

    @classmethod
    def get_entity_extractor(cls, *args, **kwargs) -> Optional[EntityExtractorMock]:
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

    def _detect_entities_chunked(self, text: str, entities: Optional[set[str]] = None) -> str:
        # Assumes the following input string:
        # "Hello my name is joe. My ssn is 123-12-1234. Unfake-able"
        #  01234567890123456789012345678901234567890123456789012345"
        return [
            {"label": "first_name", "text": "joe", "start": 17, "end": 20},
            {"label": "ssn", "text": "123-12-1234", "start": 32, "end": 43},
            {"label": "nofake", "text": "Unfake-able", "start": 45, "end": 56},
        ]

    def extract_ner_predictions(self, text: str, entities: Optional[set[str]]) -> list[NERPrediction]:
        self.extract_ner_predictions_called = True
        return [
            NERPrediction(e["text"], e["start"], e["end"], e["label"], "GLiNER", 9.0)
            for e in self._detect_entities_chunked(text, entities)
        ]


class ColumnClassifierMock(ColumnClassifier):
    """
    Classifier with no backend; uses hard coded mapping.
    """

    _num_samples: Optional[int]

    @classmethod
    def get_iapi_classifier(cls, config: IAPIClassifierConfig) -> Optional[ColumnClassifierMock]:
        classifier = cls()
        classifier._num_samples = config.num_samples
        return classifier

    @classmethod
    def get_deployed_llm_classifier(
        cls,
        llm_endpoint: str,
        num_samples: int,
    ) -> Optional[ColumnClassifierMock]:
        classifier = cls()
        classifier._num_samples = num_samples
        return classifier

    def detect_types(self, df: pd.DataFrame, all_entities: set[str]) -> dict[str, Optional[str]]:
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
