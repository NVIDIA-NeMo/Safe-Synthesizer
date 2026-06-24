# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import Field, ValidationError

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.config.training import QuantizationScheme
from nemo_safe_synthesizer.configurator.parameters import Parameters
from nemo_safe_synthesizer.errors import ParameterError


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


def test_from_params_accepts_explicit_dotted_path():
    config = SafeSynthesizerParameters.from_params(**{"generation.validation.group_by_fix_unordered_records": True})

    assert config.generation.validation.group_by_fix_unordered_records is True


def test_from_params_rejects_top_level_none_before_nested_override():
    with pytest.raises(
        ParameterError, match="Cannot assign nested parameter path 'generation.num_records'.*'generation'"
    ):
        SafeSynthesizerParameters.from_params(generation=None, **{"generation.num_records": 10})


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"generation": {"temperature": 0.7}, "generation.num_records": 10}, id="section-then-dotted"),
        pytest.param({"generation.num_records": 10, "generation": {"temperature": 0.7}}, id="dotted-then-section"),
        pytest.param({"generation": {"temperature": 0.7}, "num_records": 10}, id="section-then-bare-leaf"),
        pytest.param({"num_records": 10, "generation": {"temperature": 0.7}}, id="bare-leaf-then-section"),
    ],
)
def test_from_params_merges_section_and_leaf_overrides_order_independently(kwargs: dict[str, object]):
    config = SafeSynthesizerParameters.from_params(**kwargs)

    assert config.generation.num_records == 10
    assert config.generation.temperature == 0.7


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"privacy": None, "privacy.dp_enabled": True}, id="none-then-dotted"),
        pytest.param({"privacy.dp_enabled": True, "privacy": None}, id="dotted-then-none"),
    ],
)
def test_from_params_rejects_section_none_with_nested_override(kwargs: dict[str, object]):
    with pytest.raises(ParameterError, match="Cannot assign nested parameter path 'privacy.dp_enabled'.*'privacy'"):
        SafeSynthesizerParameters.from_params(**kwargs)


def test_from_params_rejects_duplicate_specific_parameter_paths():
    with pytest.raises(ParameterError, match="Duplicate parameter path 'generation.num_records'"):
        SafeSynthesizerParameters.from_params(num_records=10, **{"generation.num_records": 20})


def test_from_params_rejects_unknown_flat_parameter():
    with pytest.raises(ParameterError, match="Unknown parameter name 'not_a_parameter'"):
        SafeSynthesizerParameters.from_params(not_a_parameter=True)


def test_from_config_patch_validates_sparse_config():
    config = SafeSynthesizerParameters.from_config_patch({"replace_pii": None})

    assert config.replace_pii is None


def test_with_config_patch_merges_sparse_patch_and_keeps_defaults_implicit():
    config = SafeSynthesizerParameters.model_validate({"generation": {"num_records": 77}})

    merged = config.with_config_patch({"generation": {"temperature": 0.7}, "training": {"batch_size": 4}})

    assert merged.generation.num_records == 77
    assert merged.generation.temperature == 0.7
    assert merged.generation.use_structured_generation is False
    assert merged.training.batch_size == 4
    assert merged.model_dump(exclude_unset=True) == {
        "generation": {"num_records": 77, "temperature": 0.7},
        "training": {"batch_size": 4},
    }


class _LeftParameters(Parameters):
    value: int = 1


class _RightParameters(Parameters):
    value: int = 2


class _DuplicateLeafParameters(Parameters):
    left: _LeftParameters = Field(default_factory=_LeftParameters)
    right: _RightParameters = Field(default_factory=_RightParameters)


def test_parameters_get_supports_explicit_dotted_paths():
    params = _DuplicateLeafParameters()

    assert params.get("left.value") == 1
    assert params.get("right.value") == 2
    assert params.get("missing.value", "fallback") == "fallback"


def test_parameters_get_rejects_ambiguous_bare_leaf_names():
    params = _DuplicateLeafParameters()

    with pytest.raises(ParameterError, match="left.value.*right.value"):
        params.get("value")


def test_parameters_has_supports_explicit_dotted_paths():
    params = _DuplicateLeafParameters()

    assert params.has("left.value") is True
    assert params.has("right.value") is True
    assert params.has("missing.value") is False


def test_parameters_has_rejects_ambiguous_bare_leaf_names():
    params = _DuplicateLeafParameters()

    with pytest.raises(ParameterError, match="left.value.*right.value"):
        params.has("value")


def test_parameters_get_does_not_warn_for_unrelated_deprecated_fields():
    params = SafeSynthesizerParameters()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        assert params.get("batch_size") == params.training.batch_size

    assert not [warning for warning in caught if issubclass(warning.category, DeprecationWarning)]


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
    saved.generation.structured_generation.enabled = True
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
                    "generation.structured_generation.enabled": True,
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
                    "generation.structured_generation.enabled": True,  # unset runtime field preserved
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

    def test_returned_config_is_independent_of_saved(self):
        """Mutating the returned config must not affect the original (no shared references)."""
        saved = _saved_config()
        merged = saved.with_runtime_overrides(_runtime_num_records())

        # Inherited section, overridden section, and an unchanged section.
        merged.data.holdout = 0.42
        merged.training.batch_size = 99
        merged.generation.structured_generation.enabled = False

        assert merged.data is not saved.data
        assert merged.training is not saved.training
        assert saved.data.holdout != 0.42
        assert saved.training.batch_size == 8
        assert saved.generation.structured_generation.enabled is True
