# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, cast

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config import GenerateParameters
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.sdk.config_builder import ConfigBuilder


def test_with_generate_validates_raw_config_with_kwargs():
    with pytest.raises(ValidationError, match="patience"):
        ConfigBuilder().with_generate(config={"num_records": 10}, patience=0)


def test_with_generate_validates_typed_config_with_kwargs():
    with pytest.raises(ValidationError, match="patience"):
        ConfigBuilder().with_generate(config=GenerateParameters(num_records=10), patience=0)


def test_with_generate_rejects_wrong_typed_config_object():
    wrong_config = cast(Any, PiiReplacerConfig.get_default_config())

    with pytest.raises(TypeError, match="Expected GenerateParameters"):
        ConfigBuilder().with_generate(config=wrong_config)


def test_with_replace_pii_validates_default_config_with_kwargs():
    with pytest.raises(ValidationError, match="Invalid locale"):
        ConfigBuilder().with_replace_pii(globals={"locales": ["not-a-locale"]})
