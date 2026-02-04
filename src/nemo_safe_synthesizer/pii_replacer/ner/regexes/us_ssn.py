# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re

from ..entity import Entity
from ..regex import (
    Pattern,
    RegexPredictor,
    create_exact_field_matcher,
)

STRONG_MATCH_DASHES_REGEX = r"(?!\b(\d)\1+-(\d)\\1+-(\d)\1+\b)(?!123-45-6789|219-09-9999|078-05-1120)(?!666|000|9\d{2})\d{3}-(?!00)\d{2}-(?!0{4})\d{4}"  # noqa

STRONG_MATCH_NO_DASHES_REGEX = (
    r"(?!\b(\d)\1+\b)(?!123456789|219099999|078051120)(?!666|000|9\\d{2})\d{3}(?!00)\d{2}(?!0{4})\d{4}"  # noqa
)

SSN_LABELS = [
    re.compile(r"social.?security"),
    re.compile(r"social.?sec"),
    re.compile(r"soc.?sec"),
    create_exact_field_matcher("ssn"),
]

# NOTE(jm): disabling spanner b/c of too many FPs
# SPANNER = ContextSpan(pattern_list=SSN_LABELS)


class US_SSN(RegexPredictor):
    """US Social Security Number regex pattern matcher."""

    def __init__(self):
        likely_matches = [
            Pattern(
                pattern=STRONG_MATCH_DASHES_REGEX,
                header_contexts=SSN_LABELS,
                ignore_raw_score=True,
            ),
            Pattern(
                pattern=STRONG_MATCH_NO_DASHES_REGEX,
                header_contexts=SSN_LABELS,
                ignore_raw_score=True,
            ),
        ]

        super().__init__(entity=Entity.US_SOCIAL_SECURITY_NUMBER, patterns=likely_matches)
