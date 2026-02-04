# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re

from ..entity import Entity
from ..predictor import ContextSpan
from ..regex import Pattern, RegexPredictor

US_ZIPCODE_REGEX = r"\b[0-9]{5}(?:-[0-9]{4})?\b"

LABELS = [
    "postal",
    "zip",
    "us postal code",
    "zipcode",
    re.compile(r"post.code"),
]

SPANNER = ContextSpan(
    pattern_list=[
        "postal",
        "zipcode",
        re.compile(r"zip.code"),
        re.compile(r"post.code"),
        # re.compile(r"(?<!\.)zip\b"),
    ]
)


class USZipcode(RegexPredictor):
    """US Zip Code regex pattern matcher."""

    def __init__(self):
        likely_match = Pattern(
            pattern=US_ZIPCODE_REGEX,
            header_contexts=LABELS,
            span_contexts=SPANNER,
            ignore_raw_score=True,
        )

        super().__init__(entity=Entity.US_ZIP_CODE, patterns=[likely_match])
