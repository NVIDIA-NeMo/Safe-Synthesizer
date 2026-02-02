# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor


class SquareAPIKeys(RegexPredictor):
    """Match Square API Keys."""

    def __init__(self):
        entity = Entity.SQUARE_API_KEY

        # atp: personal access token
        # idp: sandbox application id
        # atb: sandbox access token
        # idp: application id
        _match_1 = Pattern(pattern=r"sq0(atp|idp|atb|idp)-[0-9A-Za-z-_]{22}", raw_score=Score.HIGH)

        # OAuth Secret
        _match_2 = Pattern(pattern=r"sq0csp-[0-9A-Za-z-_]{43}", raw_score=Score.HIGH)

        super().__init__(entity=entity, patterns=[_match_1, _match_2])
