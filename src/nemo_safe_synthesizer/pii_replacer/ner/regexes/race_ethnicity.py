# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Race and ethnicity detection."""

import re

from ..entity import Entity, Score
from ..regex import (
    Pattern,
    RegexPredictor,
    create_exact_field_matcher,
)

# These headers will be used for the more ambiguous
# racial terms such as White, Black, etc
HEADERS = [
    create_exact_field_matcher("race"),
    create_exact_field_matcher("ethnicity"),
    create_exact_field_matcher("ethnic"),
    "race",
    "ethnicity",
    "ethnic",
]

SEP = "[\\s-]"

RACES = [
    "american{sep}indian or alaska{sep}native".format(sep=SEP),
    "americanindian",
    "amer{sep}indian".format(sep=SEP),
    "amer{sep}indian{sep}eskimo".format(sep=SEP),
    "american{sep}indian{sep}eskimo".format(sep=SEP),
    "alaska{sep}native".format(sep=SEP),
    "native{sep}alaskan".format(sep=SEP),
    "african{sep}american".format(sep=SEP),
    "native{sep}hawaiian".format(sep=SEP),
    "native{sep}american".format(sep=SEP),
    "other{sep}pacific{sep}islander".format(sep=SEP),
    "pacific{sep}islander".format(sep=SEP),
    "caucasian",
]

# These terms on their own could have other meanings, so
# these are the ones we combine with header context to
# boost the scores
AMBIG_RACES = ["asian", "white", "black"]


class Race(RegexPredictor):
    def __init__(self):
        entity = Entity.RACE

        # exact field matches only, no extra text in the value
        ambig_concat = "|".join(AMBIG_RACES)
        ambig_pattern = Pattern(
            pattern=re.compile("^(?:{data})$".format(data=ambig_concat), flags=re.IGNORECASE),
            header_contexts=HEADERS,
            ignore_raw_score=True,
        )

        # phrases that are a race on their own, regardless
        # of position in the value
        concat = "|".join(RACES)
        phrase_pattern = Pattern(
            pattern=re.compile(r"\b(?:{data})\b".format(data=concat), flags=re.IGNORECASE),
            raw_score=Score.HIGH,
        )

        super().__init__(name="race", entity=entity, patterns=[ambig_pattern, phrase_pattern])


ETHS = [
    "hispanic or latino",
    "hispanic",
    "latino",
    "not hispanic or latino",
    "not-hispanic or latino",
    "not-hispanicnon-hispanic or latnino",
    "non-hispanic",
    "non hispanic",
]


class Ethnicity(RegexPredictor):
    def __init__(self):
        entity = Entity.ETHNICITY

        concat = "|".join(ETHS)
        pattern = Pattern(
            pattern=re.compile(r"\b(?:{data})\b".format(data=concat), flags=re.IGNORECASE),
            raw_score=Score.HIGH,
        )

        super().__init__(name="ethnicity", entity=entity, patterns=[pattern])
