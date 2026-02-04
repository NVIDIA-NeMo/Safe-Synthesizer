# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ..entity import Entity, Score
from ..regex import Pattern, RegexPredictor

CODE_DIGIT = "23456789CFGHJMPQRVWX"


class GoogleOLC(RegexPredictor):
    """OLC"""

    def __init__(self):
        entity = Entity.GOOGLE_OLC

        match = Pattern(
            pattern=r"^[{digit}]{{4,8}}\+[{digit}]{{2,4}}$".format(digit=CODE_DIGIT),
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )

        super().__init__(entity=entity, patterns=[match])
