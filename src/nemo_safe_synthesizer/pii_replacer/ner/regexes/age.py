# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Age detection - generally looking for ages of individuals in years or
months, so we support 4 digits in a row without any digits before/afterwards
and also descriptive words that can describe the age of an individual.
"""

import re

from ..entity import Entity
from ..regex import Pattern, RegexPredictor

# Relevant issue: detection should support descriptive age terms
HEADERS = ["age", "ages"]


class Age(RegexPredictor):
    """Determine Age in years or months, or via descriptive terms"""

    def __init__(self):
        entity = Entity.AGE

        age = Pattern(
            pattern=r"(?<!\d)\d{1,4}(?:\.\d*)?(?!\d)",
            header_contexts=HEADERS,
            ignore_raw_score=True,
        )
        desc = Pattern(
            pattern=re.compile(
                r"(?:adult|teen|infant|toddler|child|young|old|elderly|elder)",
                flags=re.IGNORECASE,
            ),
            header_contexts=HEADERS,
            ignore_raw_score=True,
        )

        super().__init__(entity=entity, patterns=[age, desc])

    def validate_match(self, matched_str: str, _):
        try:
            age = float(matched_str)
            return 0 <= age <= 120 * 12
        except ValueError:
            pass

        # it's a string and we matched on one of the
        # descriptive terms, so assume a match
        return True
