# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.predictor import ContextSpan
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor

GIT_TOKEN = re.compile(r"(?:^|\b|_)git(?:\b|_|$)")
COMMIT_URL = r"https://github.com/.*?/.*?/commit/{}"

LABELS = ["github", re.compile("^git"), GIT_TOKEN]

SPANNER = ContextSpan(pattern_list=["github", GIT_TOKEN])

NEG_HEADERS = [re.compile("git.?(?:head|commit|hash|blame)")]


class Github(RegexPredictor):
    """Match Github tokens"""

    def __init__(self):
        entity = Entity.GITHUB_TOKEN

        # NOTE: require at least one char and one number to not match long ass
        # char only strings
        _match_1 = Pattern(
            pattern=r"\b(?=.*[a-fA-F])(?=.*[0-9])[0-9a-fA-F]{35,40}\b",
            context_score=Score.MED,
            header_contexts=LABELS,
            span_contexts=SPANNER,
            ignore_raw_score=True,
            neg_header_contexts=NEG_HEADERS,
        )

        super().__init__(
            entity=entity,
            patterns=[_match_1],
        )

    def validate_match(self, matched_text, original_value):
        if re.search(COMMIT_URL.format(matched_text), original_value):
            return False
        if re.search(f"(?:id|h)={matched_text}", original_value):
            return False
        if "commit" in original_value:
            return False
        return True
