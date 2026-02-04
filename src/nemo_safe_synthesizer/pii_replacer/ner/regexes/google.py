# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..entity import Entity, Score
from ..regex import Pattern, RegexPredictor


class Google(RegexPredictor):
    """Match Google Credentials"""

    def __init__(self):
        entity = Entity.GOOGLE_DATA

        _api_key = Pattern(
            pattern=r"AIza[0-9A-Za-z-_]{35}",
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )
        _oath_token = Pattern(
            pattern=r"ya29\.[0-9A-Za-z-_]+$",
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )

        super().__init__(entity=entity, patterns=[_api_key, _oath_token])
