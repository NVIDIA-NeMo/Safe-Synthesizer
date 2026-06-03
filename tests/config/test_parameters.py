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


def _resolve(obj: object, path: str) -> object:
    """Resolve a dotted attribute ``path`` (e.g. ``generation.validation.foo``)."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _saved_config() -> SafeSynthesizerParameters:
    """Saved training-run config with customized generation, validation, and telemetry."""
    saved = SafeSynthesizerParameters()
    saved.training.batch_size = 8
    saved.emit_telemetry = False
    saved.generation.num_records = 3000
    saved.generation.use_structured_generation = True
    saved.generation.validation.group_by_ignore_invalid_records = True
    return saved


def _runtime_num_records() -> SafeSynthesizerParameters:
    """Runtime config overriding only a top-level generation field via mutation."""
    runtime = SafeSynthesizerParameters()
    runtime.generation.num_records = 100
    return runtime


def _runtime_nested_validation() -> SafeSynthesizerParameters:
    """Runtime config overriding a nested generation.validation field via mutation."""
    runtime = SafeSynthesizerParameters()
    runtime.generation.validation.group_by_fix_unordered_records = True
    return runtime


class TestWithRuntimeOverrides:
    """Tests for resume-time generation/evaluation/telemetry override merging."""

    @pytest.mark.parametrize(
        ("make_runtime", "expected"),
        [
            pytest.param(
                SafeSynthesizerParameters,
                {
                    "generation.num_records": 3000,
                    "generation.use_structured_generation": True,
                    "generation.validation.group_by_ignore_invalid_records": True,
                    "training.batch_size": 8,
                    "emit_telemetry": False,
                },
                id="empty-runtime-preserves-saved",
            ),
            pytest.param(
                _runtime_num_records,
                {
                    "generation.num_records": 100,
                    "generation.use_structured_generation": True,  # unset runtime field preserved
                    "training.batch_size": 8,  # non-overridable section inherited
                },
                id="top-level-generation-override",
            ),
            pytest.param(
                _runtime_nested_validation,
                {
                    "generation.validation.group_by_fix_unordered_records": True,
                    "generation.validation.group_by_ignore_invalid_records": True,  # saved sibling kept
                    "generation.num_records": 3000,
                },
                id="nested-validation-via-mutation",
            ),
            pytest.param(
                lambda: SafeSynthesizerParameters.model_validate(
                    {"generation": {"validation": {"group_by_fix_unordered_records": True}}}
                ),
                {
                    "generation.validation.group_by_fix_unordered_records": True,
                    "generation.validation.group_by_ignore_invalid_records": True,  # saved sibling kept
                },
                id="nested-validation-via-dict",
            ),
            pytest.param(
                lambda: SafeSynthesizerParameters.model_validate({}),
                {"emit_telemetry": False},
                id="telemetry-unset-keeps-saved",
            ),
            pytest.param(
                lambda: SafeSynthesizerParameters.model_validate({"emit_telemetry": True}),
                {"emit_telemetry": True},
                id="telemetry-set-applied",
            ),
        ],
    )
    def test_overrides(self, make_runtime, expected: dict[str, object]):
        merged = _saved_config().with_runtime_overrides(make_runtime())
        for path, value in expected.items():
            assert _resolve(merged, path) == value, path

    def test_does_not_mutate_saved(self):
        saved = _saved_config()
        saved.with_runtime_overrides(_runtime_num_records())
        assert saved.generation.num_records == 3000
