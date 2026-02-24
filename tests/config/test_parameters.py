# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig


def test_safe_synthesizer_parameters():
    config = SafeSynthesizerParameters(
        enable_synthesis=True,
        enable_replace_pii=False,
        replace_pii=None,
    )
    assert config.enable_synthesis is True
    assert config.enable_replace_pii is False
    assert config.replace_pii is None
    assert config.training.batch_size == 1


def test_pii_replacer_default():
    with pytest.raises(ValidationError):
        PiiReplacerConfig()
