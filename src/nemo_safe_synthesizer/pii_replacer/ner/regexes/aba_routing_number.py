# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity
from nemo_safe_synthesizer.pii_replacer.ner.predictor import ContextSpan
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor

ABA_ROUTING_NUMBER_REGEX_1 = r"\b[0,1,2,3,6,7,8]\d{3}-\d{4}-\d\b"
ABA_ROUTING_NUMBER_REGEX_2 = r"\b[0,1,2,3,6,7,8]\d{8}\b"


LABELS = [
    re.compile(r"^aba"),
    re.compile(r"american bank association routing"),
    re.compile(r"americanbankassociationrouting"),
    re.compile(r"bank routing"),
    re.compile(r"bankrouting(?:number)?"),
    re.compile(r"routing transit number"),
    re.compile(r"^rtn"),
    "routing",
]

SPANNER = ContextSpan(pattern_list=["routing"], span=24)

PATTERNS = [
    Pattern(
        pattern=ABA_ROUTING_NUMBER_REGEX_1,
        header_contexts=LABELS,
        span_contexts=SPANNER,
        ignore_raw_score=True,
    ),
    Pattern(
        pattern=ABA_ROUTING_NUMBER_REGEX_2,
        header_contexts=LABELS,
        span_contexts=SPANNER,
        ignore_raw_score=True,
    ),
]


class AbaRoutingNumber(RegexPredictor):
    """
    ABA Routing Number regex pattern matcher.  Pattern and label keywords both based on:
    https://docs.microsoft.com/en-us/exchange/policy-and-compliance/data-loss-prevention/sensitive-information-types?view=exchserver-2019#aba-routing-number

    Formatted and unformatted strings are both supported.

    Examples:
        {"aba": "8675-3090-1"}
        {"bank routing #": "867530901"}
    """

    def __init__(self):
        entity = Entity.ABA_ROUTING_NUMBER
        super().__init__(entity=entity, patterns=PATTERNS)
