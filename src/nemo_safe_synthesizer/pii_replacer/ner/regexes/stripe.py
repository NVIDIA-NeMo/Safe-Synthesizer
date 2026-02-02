# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor


class StripeAPIKey(RegexPredictor):
    """Match Stripe API Keys."""

    def __init__(self):
        entity = Entity.STRIPE_API_KEY

        match = Pattern(pattern=r"\b(rk|sk|pk)_live_[0-9a-zA-Z]{24}\b", raw_score=Score.HIGH)

        super().__init__(entity=entity, patterns=[match])
