# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

from ..entity import Entity, Score
from ..regex import (
    Pattern,
    RegexPredictor,
    create_exact_field_matcher,
)

# https://github.com/Gretellabs/monogretel/issues/190
SEX_HEADERS = [
    create_exact_field_matcher("sex"),
    create_exact_field_matcher("sexo"),
    create_exact_field_matcher("sexe"),
    "sex",
]
GENDER_HEADERS = [
    create_exact_field_matcher("identity"),
    create_exact_field_matcher("gender"),
    create_exact_field_matcher("genero"),
    "gender",
    "genero",
]


class Gender(RegexPredictor):
    """Match Gender"""

    def __init__(self):
        entity = Entity.GENDER

        non_binary = Pattern(
            pattern=re.compile(r"^non[-_ ]?binary$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            context_score=Score.MAX,
            header_contexts=GENDER_HEADERS,
        )
        trans_gender = Pattern(
            pattern=re.compile(r"^trans[-_ ]?(gender)?$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            context_score=Score.MAX,
            header_contexts=GENDER_HEADERS,
        )
        inter_sex = Pattern(
            pattern=re.compile(r"^inter[-_ ]?sex$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            context_score=Score.MAX,
            header_contexts=GENDER_HEADERS,
        )
        m = Pattern(
            pattern=re.compile(r"^m$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            context_score=Score.HIGH,
            header_contexts=GENDER_HEADERS,
        )
        f = Pattern(
            pattern=re.compile(r"^f$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            context_score=Score.HIGH,
            header_contexts=GENDER_HEADERS,
        )
        male = Pattern(
            pattern=re.compile(r"^male$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            context_score=Score.MAX,
            header_contexts=GENDER_HEADERS,
        )
        female = Pattern(
            pattern=re.compile(r"^female$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            context_score=Score.MAX,
            header_contexts=GENDER_HEADERS,
        )

        all_patterns = [non_binary, trans_gender, inter_sex, male, female, m, f]
        super().__init__(name="gender", entity=entity, patterns=all_patterns)


class Sex(RegexPredictor):
    """Determine Male or Female"""

    def __init__(self):
        entity = Entity.SEX

        m = Pattern(
            pattern=re.compile(r"^m$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            header_contexts=SEX_HEADERS,
        )
        f = Pattern(
            pattern=re.compile(r"^f$", flags=re.IGNORECASE),
            ignore_raw_score=True,
            header_contexts=SEX_HEADERS,
        )
        male = Pattern(
            pattern=re.compile(r"^male$", flags=re.IGNORECASE),
            raw_score=Score.HIGH,
        )
        female = Pattern(
            pattern=re.compile(r"^female$", flags=re.IGNORECASE),
            raw_score=Score.HIGH,
        )

        all_patterns = [male, female, m, f]
        super().__init__(name="sex", entity=entity, patterns=all_patterns)
