# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ipaddress

from typing_extensions import override

from ..entity import Entity, Score
from ..regex import Pattern, RegexPredictor

NEG_HEADERS = ["ver", "version", "versions"]


class IpAddress(RegexPredictor):
    """IP Address regex pattern matcher."""

    def __init__(self):
        possible_match = Pattern(
            pattern=r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}",
            raw_score=Score.LOW,
            neg_header_contexts=NEG_HEADERS,
        )

        likely_match = Pattern(
            pattern=r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)",  # noqa
            raw_score=Score.HIGH,
            neg_header_contexts=NEG_HEADERS,
        )

        super().__init__(
            entity=Entity.IP_ADDRESS,
            patterns=[possible_match, likely_match],
        )

    @override
    def validate_match(self, matched_text: str, original_text: str) -> bool:
        try:
            ipaddress.ip_address(matched_text)
        except ValueError:
            return False
        else:
            return True
