# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.predictor import ContextSpan
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor

REG = [
    r"api.{0,3}(key|token|secret)",
    r"key.{0,3}(api|access|token)",
    r"access.{0,3}(token|key|secret)",
    r"secret",
]

LABELS = [
    "token",
]

for reg in REG:
    LABELS.append(re.compile("^" + reg))

SPANNER_PATTERNS = ["token"]

for reg in REG:
    SPANNER_PATTERNS.append(re.compile(reg))

SPANNER = ContextSpan(pattern_list=SPANNER_PATTERNS, span=24)


class GenericKey(RegexPredictor):
    """Attempt Generic API key/token matching"""

    def __init__(self):
        entity = Entity.GENERIC_KEY

        match = Pattern(
            pattern=r"\b[A-Za-z0-9-_.]{20,255}\b",
            context_score=Score.HIGH,
            raw_score=Score.LOW,
            header_contexts=LABELS,
            span_contexts=SPANNER,
            ignore_raw_score=True,
        )

        super().__init__(entity=entity, patterns=[match])
