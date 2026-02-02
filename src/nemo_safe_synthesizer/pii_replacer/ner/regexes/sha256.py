# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor

SHA256_REGEX = r"^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9]{64}$"


class SHA256(RegexPredictor):
    """
    SHA256 regex pattern matcher. Will match any length 64 hex string in
    isolation (not embedded in a longer string), regardless of label or lack thereof.

    Examples:
        {"foo": "f63e93baeb60ca18fdc9a81e9358417df63e93baeb60ca18fdc9a81e9358417d"}
        "f63e93baeb60ca18fdc9a81e9358417df63e93baeb60ca18fdc9a81e9358417d"

    Counterexamples:
        {"bar": "baz f63e93baeb60ca18fdc9a81e9358417df63e93baeb60ca18fdc9a81e9358417d"}
    """

    def __init__(self):
        patterns = [Pattern(pattern=SHA256_REGEX, raw_score=Score.HIGH)]

        super().__init__(entity=Entity.SHA256, patterns=patterns)
