# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tldextract
from typing_extensions import override

from ..entity import Entity, Score
from ..predictor import ContextSpan
from ..regex import Pattern, RegexPredictor

URL_REGEX = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,4}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"  # noqa

URL_LABELS = ["url", "web", "address", "uri", "urn", "http", "internet", "www"]

SPANNER = ContextSpan(pattern_list=URL_LABELS)


class URL(RegexPredictor):
    """Web url regex pattern matcher."""

    tld_extract: tldextract.TLDExtract

    def __init__(self):
        match = Pattern(
            pattern=URL_REGEX,
            raw_score=Score.HIGH,
            header_contexts=URL_LABELS,
            span_contexts=SPANNER,
        )
        self.tld_extract = tldextract.TLDExtract(suffix_list_urls=())
        super().__init__(entity=Entity.URL, patterns=[match])

    @override
    def validate_match(self, matched_text: str, original_text: str) -> bool:
        result = self.tld_extract(matched_text)
        return result.fqdn != ""
