# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from stdnum import imei

from ..entity import Entity
from ..predictor import ContextSpan
from ..regex import Pattern, RegexPredictor

LABELS = ["imei", "hardware", "meid", "imeis", "meids"]

SPANNER = ContextSpan(
    pattern_list=LABELS,
)


class IMEI(RegexPredictor):
    """IMEI regex pattern matcher."""

    def __init__(self):
        unlikely_match = Pattern(
            pattern=r"\b[0-9]{15}\b",
            header_contexts=LABELS,
            span_contexts=SPANNER,
            ignore_raw_score=True,
        )
        likely_match = Pattern(
            pattern=r"\b[0-9]{2}\-[0-9]{6}-[0-9]{6}-[0-9]{1}\b",
            header_contexts=LABELS,
            span_contexts=SPANNER,
            ignore_raw_score=True,
        )

        super().__init__(
            entity=Entity.IMEI_HARDWARE_ID,
            patterns=[unlikely_match, likely_match],
        )

    def validate_match(self, in_text: str, _) -> bool:
        try:
            check = imei.validate(in_text)
        except Exception:
            return False
        return True if check else False
