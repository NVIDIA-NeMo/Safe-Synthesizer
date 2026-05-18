# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.config.training import QuantizationScheme


def test_safe_synthesizer_parameters(monkeypatch):
    monkeypatch.delenv("NEMO_TELEMETRY_ENABLED", raising=False)
    config = SafeSynthesizerParameters(
        replace_pii=None,
    )
    assert config.replace_pii is None
    assert config.training.batch_size == 1
    assert config.emit_telemetry is True


def test_emit_telemetry_can_be_disabled_from_yaml():
    c = SafeSynthesizerParameters.from_yaml_str("emit_telemetry: false\n")
    assert c.emit_telemetry is False


def test_emit_telemetry_can_be_disabled_from_params():
    c = SafeSynthesizerParameters.from_params(emit_telemetry=False)
    assert c.emit_telemetry is False


def test_emit_telemetry_defaults_from_env_when_unset(monkeypatch):
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")

    c = SafeSynthesizerParameters()

    assert c.emit_telemetry is False


def test_emit_telemetry_explicit_value_overrides_env(monkeypatch):
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")

    c = SafeSynthesizerParameters(emit_telemetry=True)

    assert c.emit_telemetry is True


def test_emit_telemetry_from_yaml_uses_env_when_unset(monkeypatch):
    monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")

    c = SafeSynthesizerParameters.from_yaml_str("training: {}\n")

    assert c.emit_telemetry is False


def test_pii_replacer_default():
    with pytest.raises(ValidationError):
        PiiReplacerConfig()  # ty: ignore[missing-argument] -- intentionally testing validation error


def test_quantization_scheme_from_alias_rejects_invalid_legacy_bit_alias() -> None:
    with pytest.raises(ValueError, match="Expected 4 or 8"):
        QuantizationScheme.from_alias(5)  # ty: ignore[invalid-argument-type] -- intentionally invalid alias


def test_quantization_scheme_8bit_uses_valid_bitsandbytes_kwargs() -> None:
    config = MagicMock()

    with patch("transformers.BitsAndBytesConfig", return_value=config) as bitsandbytes_config:
        result = QuantizationScheme.BNB_8BIT.to_transformers_config()

    assert result is config
    bitsandbytes_config.assert_called_once_with(load_in_8bit=True)


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


def test_to_yaml_from_yaml_round_trip_enabled(tmp_path: Path):
    c1 = SafeSynthesizerParameters()
    yaml_path = tmp_path / "config.yaml"
    c1.to_yaml(str(yaml_path), exclude_unset=False)
    c2 = SafeSynthesizerParameters.from_yaml(str(yaml_path))
    assert c2.replace_pii is not None


def test_to_yaml_from_yaml_round_trip_disabled(tmp_path: Path):
    c1 = SafeSynthesizerParameters(replace_pii=None)
    yaml_path = tmp_path / "config.yaml"
    c1.to_yaml(str(yaml_path), exclude_unset=False)
    c2 = SafeSynthesizerParameters.from_yaml(str(yaml_path))
    assert c2.replace_pii is None


def test_from_params_absent_enables_pii():
    assert SafeSynthesizerParameters.from_params().replace_pii is not None


def test_from_params_none_disables_pii():
    assert SafeSynthesizerParameters.from_params(replace_pii=None).replace_pii is None
