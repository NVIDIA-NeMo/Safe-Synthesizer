# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import tldextract

from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.predictor import ContextSpan
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor

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
        self.tld_extract = tldextract.TLDExtract(suffix_list_urls=None)
        super().__init__(entity=Entity.URL, patterns=[match])

    def validate_match(self, in_text: str, orig: str) -> bool:
        result = self.tld_extract(in_text)
        return result.fqdn != ""
