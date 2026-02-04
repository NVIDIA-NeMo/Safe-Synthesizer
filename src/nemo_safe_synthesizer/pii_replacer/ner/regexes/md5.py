# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from ..entity import Entity, Score
from ..regex import Pattern, RegexPredictor

MD5_REGEX = r"^(?=.*[a-fA-F])(?=.*[0-9])[a-fA-F0-9]{32}$"


class MD5(RegexPredictor):
    """
    MD5 regex pattern matcher.  Will match any length 32 hex string in
    isolation (not embedded in a longer string), regardless of label or lack thereof.

    Examples:
        {"foo": "f63e93baeb60ca18fdc9a81e9358417d"}
        "f63e93baeb60ca18fdc9a81e9358417d"

    Counterexamples:
        {"bar": "baz f63e93baeb60ca18fdc9a81e9358417d"}
    """

    def __init__(self):
        patterns = [Pattern(pattern=MD5_REGEX, raw_score=Score.HIGH)]

        super().__init__(entity=Entity.MD5, patterns=patterns)
