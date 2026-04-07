# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from ..entity import Entity, Score
from ..regex import Pattern, RegexPredictor

UUID_REGEX = r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[1-5][0-9a-fA-F]{3}-?[89abAB][0-9a-fA-F]{3}-?[0-9a-fA-F]{12}$"


class UUID(RegexPredictor):
    """UUID regex pattern matcher."""

    def __init__(self):
        patterns = [Pattern(pattern=UUID_REGEX, raw_score=Score.HIGH)]

        super().__init__(entity=Entity.UUID, patterns=patterns)
