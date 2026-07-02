# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

from ..entity import Entity
from ..predictor import ContextSpan
from ..regex import Pattern, RegexPredictor

US_PHONE_REGEX = r"((\+?1)|(001))?-?\(?[0-9]{3}\)?[-.*\s]?[0-9]{3}[-.*\s]?[0-9]{4}(x\d+)?"


US_PHONE_LABELS = [
    "phone",
    "cellular",
    "contact",
    "fax",
    "facsimile",
    "mobile",
    "msisdn",
    # "contact", NOTE: too generic
    re.compile(r"^toll.?free"),
    re.compile(r"^tel[e]?[\W_]+"),
    re.compile(r"^tel[e]?$"),
    re.compile(r"[\W_]+cell[\W_]+"),
    re.compile(r"^cell[\W_]+"),
    re.compile(r"[\W_]+cell$"),
    re.compile(r"^cell$"),
    # "^ph\W*$",  NOTE: too generic
    re.compile(r"(?:work|home|office|local|campus|^lab|campus|^day|^eve|evening|^main|business|biz)[\W_]*(?:num|no|#)"),
]


CONTEXTS = [
    ContextSpan(pattern_list=US_PHONE_LABELS, span=24),
    # some "action" verbs we might see (i.e. contact me at X, reach me at X)
    ContextSpan(pattern_list=["call", "contact", "reach"], span=16),
]


class USPhone(RegexPredictor):
    """US Phone regex pattern matcher."""

    def __init__(self):
        likely_match = Pattern(
            pattern=US_PHONE_REGEX,
            header_contexts=US_PHONE_LABELS,
            span_contexts=CONTEXTS,
            ignore_raw_score=True,
        )

        super().__init__(entity=Entity.PHONE_NUMBER, patterns=[likely_match])
