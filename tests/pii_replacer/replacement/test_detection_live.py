# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in acceptance evaluation for the pinned GLiNER2 privacy checkpoint."""

from __future__ import annotations

import os

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType, FreeTextDetectionConfig
from nemo_safe_synthesizer.pii_replacer.replacement.detection import (
    Gliner2Detector,
    fresh_detection_entity_types,
)
from nemo_safe_synthesizer.pii_replacer.replacement.types import DetectionCell, DetectionCellId

pytestmark = [pytest.mark.slow, pytest.mark.requires_gpu]


@pytest.mark.skipif(
    os.environ.get("NSS_RUN_LIVE_GLINER2") != "1",
    reason="set NSS_RUN_LIVE_GLINER2=1 to evaluate the pinned local checkpoint",
)
def test_pinned_model_detects_representative_name_and_birth_date() -> None:
    text = "Ada Lovelace was born on 10 December 1815; contact ada@example.com."
    cell = DetectionCell(DetectionCellId(0, "notes"), text, fresh_detection_entity_types())

    spans = Gliner2Detector(FreeTextDetectionConfig()).detect([cell])
    entity_types = {span.entity_type for span in spans}

    assert EntityType.DATE_OF_BIRTH in entity_types
    assert entity_types & {
        EntityType.FIRST_NAME,
        EntityType.LAST_NAME,
    }


def test_precision_first_name_entities_reject_representative_medical_terms() -> None:
    texts = [
        "Patient admitted for observation.",
        "M. tuberculosis detected on culture.",
        "Metastatic olfactory neuroblastoma",
        "parathyroid adenoma",
        "Cisplatin and 5-Fluorouracil",
        "Mycobacterium marinum",
    ]
    cells = [
        DetectionCell(DetectionCellId(position, "notes"), text, fresh_detection_entity_types())
        for position, text in enumerate(texts)
    ]

    spans = Gliner2Detector(FreeTextDetectionConfig()).detect(cells)

    assert not {
        EntityType.FIRST_NAME,
        EntityType.MIDDLE_NAME,
        EntityType.LAST_NAME,
    } & {span.entity_type for span in spans}
