# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tldextract

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

        self.tld_extract = tldextract.TLDExtract(suffix_list_urls=None)
        super().__init__(entity=entity, patterns=[match])

    def validate_match(self, in_text: str, _) -> bool:
        result = self.tld_extract(in_text)
        return result.fqdn != ""
