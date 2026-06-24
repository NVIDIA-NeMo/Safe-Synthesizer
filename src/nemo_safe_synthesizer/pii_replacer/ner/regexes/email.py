# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tldextract
from typing_extensions import override

from ..entity import Entity, Score
from ..regex import Pattern, RegexPredictor


class Email(RegexPredictor):
    """Email address regex pattern matcher."""

    tld_extract: tldextract.TLDExtract

    def __init__(self):
        entity = Entity.EMAIL_ADDRESS
        match = Pattern(
            pattern=r"\b((([!#$%&'*+\-/=?^_`{|}~\w])|([!#$%&'*+\-/=?^_`{|}~\w][!#$%&'*+\-/=?^_`{|}~\.\w]{0,}[!#$%&'"
            r"*+\-/=?^_`{|}~\w]))[@]\w+([-.]\w+)*\.\w+([-.]\w+)*)\b",
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )

        self.tld_extract = tldextract.TLDExtract(suffix_list_urls=())
        super().__init__(entity=entity, patterns=[match])

    @override
    def validate_match(self, matched_text: str, original_text: str) -> bool:
        result = self.tld_extract(matched_text)
        return result.fqdn != ""
