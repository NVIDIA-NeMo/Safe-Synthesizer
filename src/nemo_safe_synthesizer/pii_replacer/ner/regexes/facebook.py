# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor


class Facebook(RegexPredictor):
    """Match Google Credentials"""

    def __init__(self):
        entity = Entity.FACEBOOK_DATA

        # https://developers.facebook.com/docs/facebook-login/access-tokens/
        _access_token = Pattern(
            pattern=r"EAACEdE[A-Za-z0-9-_.]{0,255}$",
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )

        super().__init__(entity=entity, patterns=[_access_token])
