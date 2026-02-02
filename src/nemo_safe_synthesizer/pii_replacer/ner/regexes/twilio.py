# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor


class TwilioAPIKeys(RegexPredictor):
    """Match Twilio API Keys"""

    def __init__(self):
        entity = Entity.TWILIO_DATA

        _match_1 = Pattern(pattern=r"\b(SK|AC)[0-9a-fA-F]{32}\b", raw_score=Score.HIGH)

        super().__init__(entity=entity, patterns=[_match_1])
