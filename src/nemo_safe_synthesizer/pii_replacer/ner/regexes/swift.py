# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

from ..entity import Entity
from ..regex import Pattern, RegexPredictor
from .iban import regex_per_country

# NOTE that https://docs.microsoft.com/en-us/exchange/policy-and-compliance/data-loss-prevention/sensitive-information-types?view=exchserver-2019#swift-code  # noqa
# and https://en.wikipedia.org/wiki/ISO_9362 specify different specs.  Using ISO_9362 spec.

SWIFT_GENERIC_REGEX = r"\b[a-zA-Z]{6}[a-zA-Z0-9]{2}([a-zA-Z0-9]{3})?\b"

LABELS = [
    re.compile(r"international organization for standardization 9362"),
    re.compile(r"iso.?9362"),
    "swift",
    "bic",
    re.compile(r"bank.identifier.code"),
    re.compile(r"標準化9362"),
    re.compile(r"迅速＃"),
    re.compile("迅速なルーティング番号"),
    re.compile("銀行識別コードのための国際組織"),
    re.compile(r"Organisation internationale de normalisation 9362"),
    re.compile(r"rapide #"),
    re.compile(r"code identificateur de banque"),
]


class SWIFT(RegexPredictor):
    """
    SWIFT regex pattern matcher based on https://en.wikipedia.org/wiki/ISO_9362.
    Matches are validated against the same list of country codes used for IBAN regex.

    Labels are based on https://docs.microsoft.com/en-us/exchange/policy-and-compliance/data-loss-prevention/sensitive-information-types?view=exchserver-2019#swift-code  # noqa

    Examples:
        {"bic#": "DEUTDEFF500"}
        {"迅速＃": "NEDSZAJJXXX"}
        {"iso9362": "nedszajj"}

    """

    def __init__(self):
        patterns = [
            Pattern(
                pattern=SWIFT_GENERIC_REGEX,
                header_contexts=LABELS,
                ignore_raw_score=True,
            )
        ]

        super().__init__(entity=Entity.SWIFT_CODE, patterns=patterns)

    def validate_match(self, in_text: str, _) -> bool:
        country_code = in_text[4:6]
        # Keys of the more extensive dict in iban regex are country codes used for swift
        return country_code.upper() in regex_per_country
