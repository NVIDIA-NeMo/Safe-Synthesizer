# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from nemo_safe_synthesizer.pii_replacer.ner.entity import Entity, Score
from nemo_safe_synthesizer.pii_replacer.ner.regex import Pattern, RegexPredictor

SHA512_REGEX = r"^(?=.*[a-zA-Z])(?=.*[0-9])[a-zA-Z0-9]{128}$"


class SHA512(RegexPredictor):
    """
    SHA512 regex pattern matcher. Will match any length 128 hex string in
    isolation (not embedded in a longer string), regardless of label or lack thereof.

    Examples:
        {"foo": "f63e93baeb60ca18fdc9a81e9358417dacbd18db4cc2f85cedef654fccc4a4d8f63e93baeb60ca18fdc9a81e9358417dacbd18db4cc2f85cedef654fccc4a4d8"}  # noqa
        "f63e93baeb60ca18fdc9a81e9358417dacbd18db4cc2f85cedef654fccc4a4d8f63e93baeb60ca18fdc9a81e9358417dacbd18db4cc2f85cedef654fccc4a4d8"

    Counterexamples:
        {"bar": "baz f63e93baeb60ca18fdc9a81e9358417dacbd18db4cc2f85cedef654fccc4a4d8f63e93baeb60ca18fdc9a81e9358417dacbd18db4cc2f85cedef654fccc4a4d8"}  # noqa
    """

    def __init__(self):
        patterns = [Pattern(pattern=SHA512_REGEX, raw_score=Score.HIGH)]

        super().__init__(entity=Entity.SHA512, patterns=patterns)
