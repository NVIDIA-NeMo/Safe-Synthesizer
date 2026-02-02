# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor


class SendGrid(RegexPredictor):
    """Match SendGrid API key(s)"""

    def __init__(self):
        entity = Entity.SENDGRID_CREDENTIALS

        _match_1 = Pattern(pattern=r"SG\.[a-zA-Z0-9-_]{22}\.[a-zA-Z0-9-_]{43}", raw_score=Score.HIGH)
        super().__init__(entity=entity, patterns=[_match_1])
