# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig


def test_safe_synthesizer_parameters():
    config = SafeSynthesizerParameters(
        replace_pii=None,
    )
    assert config.replace_pii is None
    assert config.training.batch_size == 1


def test_pii_replacer_default():
    with pytest.raises(ValidationError):
        PiiReplacerConfig()


# --- replace_pii default_factory invariants ---


def test_default_constructor_enables_pii():
    assert SafeSynthesizerParameters().replace_pii is not None


def test_model_validate_empty_dict_enables_pii():
    assert SafeSynthesizerParameters.model_validate({}).replace_pii is not None


def test_model_validate_null_disables_pii():
    assert SafeSynthesizerParameters.model_validate({"replace_pii": None}).replace_pii is None


def test_from_yaml_str_absent_key_enables_pii():
    c = SafeSynthesizerParameters.from_yaml_str("training:\n  batch_size: 4\n")
    assert c.replace_pii is not None


def test_from_yaml_str_null_disables_pii():
    c = SafeSynthesizerParameters.from_yaml_str("replace_pii: null\n")
    assert c.replace_pii is None


def test_old_yaml_with_enable_replace_pii_loads_cleanly():
    # Migration: configs written before this change had enable_replace_pii: true
    # and no replace_pii key. The extra field must be silently ignored and
    # default_factory must fire so PII stays on.
    c = SafeSynthesizerParameters.model_validate({"enable_replace_pii": True})
    assert c.replace_pii is not None


def test_to_yaml_from_yaml_round_trip_enabled():
    c1 = SafeSynthesizerParameters()
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=True) as f:
        fname = f.name
    c1.to_yaml(fname, exclude_unset=False)
    c2 = SafeSynthesizerParameters.from_yaml(fname)
    assert c2.replace_pii is not None


def test_to_yaml_from_yaml_round_trip_disabled():
    c1 = SafeSynthesizerParameters(replace_pii=None)
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=True) as f:
        fname = f.name
    c1.to_yaml(fname, exclude_unset=False)
    c2 = SafeSynthesizerParameters.from_yaml(fname)
    assert c2.replace_pii is None


def test_from_params_absent_enables_pii():
    assert SafeSynthesizerParameters.from_params().replace_pii is not None


def test_from_params_none_disables_pii():
    assert SafeSynthesizerParameters.from_params(replace_pii=None).replace_pii is None
