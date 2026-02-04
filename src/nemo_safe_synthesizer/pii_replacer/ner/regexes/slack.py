# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Detection for Slack webhooks and tokens

We set both score values to 1.0 here because these are such precise matches.
"""

from ..entity import Entity, Score
from ..regex import Pattern, RegexPredictor


class SlackSecrets(RegexPredictor):
    """Match Slack Secrets"""

    def __init__(self):
        entity = Entity.SLACK_SECRETS

        _webhook = Pattern(
            pattern=r"https://hooks.slack.com/services/T[a-zA-Z0-9_]{8,12}/B[a-zA-Z0-9_]{8,12}/[a-zA-Z0-9_]{24}",
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )  # noqa

        # bot:                  : xoxb-
        # user:                 : xoxp-
        # workspace access:     : xoxa-2-
        # workspace refresh:    : xoxr-
        _token_legacy = Pattern(
            pattern=r"(xox[p|b|o|a]-[0-9]{12}-[0-9]{12}-[0-9]{12}-[a-z0-9]{32})",
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )

        # https://api.slack.com/authentication/token-types
        _new_tokens = Pattern(
            pattern=r"xox[b|p|r]-[0-9a-zA-Z-_]{12,255}",
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )
        _new_access = Pattern(
            pattern=r"xoxa-2[0-9a-zA-Z-_]{12,255}",
            context_score=Score.HIGH,
            raw_score=Score.HIGH,
        )

        _patterns = [_webhook, _token_legacy, _new_tokens, _new_access]

        super().__init__(entity=entity, patterns=_patterns)
