# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from random import Random

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.planning.patterns import render_character_mask, render_name_pattern


@pytest.mark.unit
class TestPatternRendering:
    def test_invalid_character_mask_raises_parameter_error(self) -> None:
        with pytest.raises(ParameterError):
            render_character_mask("[", Random(0))

    def test_email_name_parts_join_their_words_with_dots(self) -> None:
        values = {"first": "Mary Ann", "last": "De  La Cruz", "domain": "example.com"}

        email = render_name_pattern(EntityType.EMAIL, "{first}_{Last}@{domain}", values, Random(0))
        full_name = render_name_pattern(EntityType.FULL_NAME, "{First} {Last}", values, Random(0))

        assert email == "mary.ann_De.La.Cruz@example.com"
        assert full_name == "Mary Ann De  La Cruz"

    def test_invalid_name_pattern_raises_parameter_error(self) -> None:
        with pytest.raises(ParameterError):
            render_name_pattern(EntityType.EMAIL, "{unknown}@example.com", {}, Random(0))
