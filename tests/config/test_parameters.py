# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar, cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

from nemo_safe_synthesizer.config.generate import GenerateParameters, StructuredGenerationParameters
from nemo_safe_synthesizer.config.job import SafeSynthesizerJobConfig
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.replace_pii import ReplacePiiConfig
from nemo_safe_synthesizer.config.training import QuantizationScheme, TrainingHyperparams
from nemo_safe_synthesizer.configurator.parameter_paths import (
    AmbiguousParameterName,
    ParameterFieldKind,
    ParameterPath,
    ParameterSchema,
    ResolvedParameterName,
    UnknownParameterName,
    classify_parameter_annotation,
)
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


def test_replace_pii_default():
    cfg = ReplacePiiConfig()
    assert cfg.replacement.locale == "en_US"


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


def test_model_validate_accepts_an_existing_config():
    config = SafeSynthesizerParameters(unknown_fields="ignore")

    assert SafeSynthesizerParameters.model_validate(config) is config


def test_model_validate_non_mapping_input_uses_default_unknown_field_policy():
    with pytest.raises(ValidationError) as exc_info:
        SafeSynthesizerParameters.model_validate(None)

    assert exc_info.value.errors()[0]["type"] == "model_attributes_type"


def test_from_yaml_str_absent_key_enables_pii():
    c = SafeSynthesizerParameters.from_yaml_str("training:\n  batch_size: 4\n")
    assert c.replace_pii is not None


def test_from_yaml_str_null_disables_pii():
    c = SafeSynthesizerParameters.from_yaml_str("replace_pii: null\n")
    assert c.replace_pii is None


def test_old_yaml_with_enable_replace_pii_requires_ignore_policy():
    with pytest.raises(ValidationError, match="enable_replace_pii"):
        SafeSynthesizerParameters.model_validate({"enable_replace_pii": True})

    c = SafeSynthesizerParameters.model_validate({"unknown_fields": "ignore", "enable_replace_pii": True})
    assert c.replace_pii is not None


def test_legacy_replace_pii_globals_steps_always_rejected():
    """Legacy transform keys must not silently become auto-discovery under ignore."""
    for key in ("globals", "steps"):
        with pytest.raises((ParameterError, ValidationError), match=rf"replace_pii\.{key}"):
            SafeSynthesizerParameters.model_validate(
                {
                    "unknown_fields": "ignore",
                    "replace_pii": {key: {"anything": True}},
                }
            )


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


def test_from_params_rejects_unknown_explicit_dotted_path():
    with pytest.raises(ParameterError, match=r"Unknown parameter path 'generation\.not_a_field'"):
        SafeSynthesizerParameters.from_params(**{"generation.not_a_field": True})


def test_from_params_rejects_top_level_none_before_nested_override():
    with pytest.raises(
        ParameterError, match=r"Cannot assign nested parameter path 'generation\.num_records'.*'generation'"
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
    with pytest.raises(ParameterError, match=r"Cannot assign nested parameter path 'privacy\.dp_enabled'.*'privacy'"):
        SafeSynthesizerParameters.from_params(**kwargs)


def test_from_params_rejects_duplicate_specific_parameter_paths():
    with pytest.raises(ParameterError, match=r"Duplicate parameter path 'generation\.num_records'"):
        SafeSynthesizerParameters.from_params(num_records=10, **{"generation.num_records": 20})


def test_from_params_rejects_unknown_flat_parameter():
    with pytest.raises(ParameterError, match="Unknown parameter name 'not_a_parameter'"):
        SafeSynthesizerParameters.from_params(not_a_parameter=True)


def test_parameter_schema_indexes_optional_branches_without_an_instance():
    with patch.object(SafeSynthesizerParameters, "__init__", side_effect=AssertionError("model instantiated")):
        schema = ParameterSchema.from_model(SafeSynthesizerParameters)
    fields = {str(field.path): field.kind for field in schema.fields}

    assert "privacy" in fields
    assert fields["privacy"] is ParameterFieldKind.BRANCH
    assert fields["privacy.dp_enabled"] is ParameterFieldKind.LEAF
    assert isinstance(schema.resolve("privacy.dp_enabled"), ResolvedParameterName)
    assert isinstance(schema.resolve("privacy.not_a_field"), UnknownParameterName)


def test_parameter_schema_reports_ambiguous_bare_names_with_candidates():
    result = ParameterSchema.from_model(SafeSynthesizerParameters).resolve("enabled")

    assert isinstance(result, AmbiguousParameterName)
    assert {str(path) for path in result.candidates} >= {
        "evaluation.enabled",
        "generation.structured_generation.enabled",
    }


def test_mapping_valued_step_vars_annotation_is_a_leaf():
    class _StepLike(Parameters):
        vars: dict[str, object] = Field(default_factory=dict)

    annotation = _StepLike.model_fields["vars"].annotation

    assert classify_parameter_annotation(annotation) is ParameterFieldKind.LEAF


def test_parameters_queries_remain_instance_based_for_disabled_optional_branch():
    params = SafeSynthesizerParameters(privacy=None)
    schema_result = ParameterSchema.from_model(SafeSynthesizerParameters).resolve("privacy.dp_enabled")

    assert params.get("privacy.dp_enabled", "missing") == "missing"
    assert params.has("privacy.dp_enabled") is False
    assert isinstance(schema_result, ResolvedParameterName)


def test_from_config_patch_validates_sparse_config():
    config = SafeSynthesizerParameters.from_config_patch({"replace_pii": None})

    assert config.replace_pii is None


def test_from_config_source_keyword_policy_takes_precedence_over_mapping():
    relaxed = SafeSynthesizerParameters.from_config_source(
        {"unknown_fields": "reject", "unknown": True},
        unknown_fields="ignore",
    )
    assert relaxed.unknown_fields == "ignore"

    with pytest.raises(ParameterError, match="unknown"):
        SafeSynthesizerParameters.from_config_source(
            {"unknown_fields": "ignore", "unknown": True},
            unknown_fields="reject",
        )


def test_from_config_source_preserves_policy_from_an_existing_config():
    source = SafeSynthesizerParameters(unknown_fields="ignore")

    assert SafeSynthesizerParameters.from_config_source(source).unknown_fields == "ignore"


def test_from_config_source_resolves_policy_without_a_source_mapping():
    assert SafeSynthesizerParameters.from_config_source(unknown_fields="ignore").unknown_fields == "ignore"


@pytest.mark.parametrize(
    "patch",
    [
        {"unknown": True},
        {"generation": {"unknown": True}},
    ],
)
def test_from_config_patch_rejects_unknown_mapping_keys_by_default(patch: dict[str, object]):
    with pytest.raises(ParameterError, match="unknown"):
        SafeSynthesizerParameters.from_config_patch(patch)


@pytest.mark.parametrize(
    ("patch", "expected"),
    [
        ({"unknown_fields": "ignore", "unknown": True}, {"unknown_fields": "ignore"}),
        (
            {"unknown_fields": "ignore", "generation": {"unknown": True}},
            {"unknown_fields": "ignore", "generation": {}},
        ),
    ],
)
def test_from_config_patch_ignores_unknown_mapping_keys_when_non_strict(
    patch: dict[str, object], expected: dict[str, object]
):
    config = SafeSynthesizerParameters.from_config_patch(patch)

    assert config.model_dump(exclude_unset=True) == expected


def test_with_config_patch_ignores_unknown_mapping_keys_and_preserves_sparse_base():
    config = SafeSynthesizerParameters.model_validate({"unknown_fields": "ignore", "generation": {"num_records": 77}})

    merged = config.with_config_patch({"unknown": True, "generation": {"unknown": True, "temperature": 0.7}})

    assert merged.model_dump(exclude_unset=True) == {
        "generation": {"num_records": 77, "temperature": 0.7},
        "unknown_fields": "ignore",
    }


def test_with_config_patch_uses_patch_policy_for_sibling_keys():
    strict = SafeSynthesizerParameters()
    relaxed = strict.with_config_patch({"unknown_fields": "ignore", "unknown": True})
    assert relaxed.unknown_fields == "ignore"

    with pytest.raises(ParameterError, match="unknown"):
        relaxed.with_config_patch({"unknown_fields": "reject", "unknown": True})


@pytest.mark.parametrize(
    "data",
    [
        {"evaluate": {}},
        {"training": {"epoch": 1}},
    ],
)
def test_unknown_fields_are_rejected_recursively_by_default(data: dict[str, object]):
    with pytest.raises(ValidationError, match="Unknown configuration field"):
        SafeSynthesizerParameters.model_validate(data)


def test_unknown_fields_are_ignored_recursively_with_ignore_policy():
    config = SafeSynthesizerParameters.model_validate(
        {"unknown_fields": "ignore", "evaluate": {}, "training": {"epoch": 1}}
    )

    assert config.unknown_fields == "ignore"


def test_constructor_applies_unknown_field_policy_recursively():
    with pytest.raises(ValidationError, match="training.epoch"):
        SafeSynthesizerParameters(training={"epoch": 1})  # ty: ignore[invalid-argument-type]

    config = SafeSynthesizerParameters(
        unknown_fields="ignore",
        training={"epoch": 1},  # ty: ignore[invalid-argument-type]
    )
    assert config.unknown_fields == "ignore"
    assert not hasattr(config.training, "epoch")


def test_unknown_field_policy_reaches_models_inside_collections():
    raw = SafeSynthesizerParameters().model_dump()
    assert isinstance(raw["replace_pii"], dict)
    raw["replace_pii"]["replacement_plan"] = {
        "standalone_columns_to_replace": [{"column_name": "Phone", "unknown": True}]
    }

    with pytest.raises(ValidationError, match="unknown"):
        SafeSynthesizerParameters.model_validate(raw)

    raw["unknown_fields"] = "ignore"
    assert SafeSynthesizerParameters.model_validate(raw).unknown_fields == "ignore"


def test_service_job_config_preserves_dynamic_unknown_field_policy():
    payload: dict[str, object] = {"data_source": "dataset", "config": {"training": {"epoch": 1}}}
    with pytest.raises(ValidationError, match="epoch"):
        SafeSynthesizerJobConfig.model_validate(payload)

    payload["config"] = {"unknown_fields": "ignore", "training": {"epoch": 1}}
    job = SafeSynthesizerJobConfig.model_validate(payload)
    assert job.config.unknown_fields == "ignore"

    config = SafeSynthesizerParameters(unknown_fields="ignore")
    job = SafeSynthesizerJobConfig.model_validate({"data_source": "dataset", "config": config})
    assert job.config is config


def test_arbitrary_wrapper_preserves_dynamic_unknown_field_policy():
    class Wrapper(BaseModel):
        config: SafeSynthesizerParameters

    with pytest.raises(ValidationError, match="epoch"):
        Wrapper.model_validate({"config": {"training": {"epoch": 1}}})

    wrapped = Wrapper.model_validate({"config": {"unknown_fields": "ignore", "training": {"epoch": 1}}})
    assert wrapped.config.unknown_fields == "ignore"
    assert not hasattr(wrapped.config.training, "epoch")


def test_model_validate_does_not_allow_a_third_unknown_field_policy():
    with pytest.raises(ValidationError, match="unknown"):
        SafeSynthesizerParameters.model_validate({"unknown": 1}, extra="allow")


def test_unknown_field_policy_rejects_unsupported_value():
    with pytest.raises(ValidationError, match="Input should be 'ignore' or 'reject'"):
        SafeSynthesizerParameters.model_validate({"unknown_fields": "allow"})


def test_with_config_patch_keeps_validator_and_factory_defaults_implicit():
    config = SafeSynthesizerParameters.model_validate({"generation": {"num_records": 77}})

    merged = config.with_config_patch({"generation": {"temperature": 0.7}, "training": {"batch_size": 4}})

    assert merged.generation.num_records == 77
    assert merged.generation.temperature == 0.7
    assert merged.generation.structured_generation.enabled is False
    assert merged.training.batch_size == 4
    sparse = merged.model_dump(exclude_unset=True)
    assert sparse["generation"] == {"num_records": 77, "temperature": 0.7}
    assert sparse["training"] == {"batch_size": 4}
    assert "data" not in sparse
    assert "replace_pii" not in sparse
    assert "evaluation" not in sparse
    assert "time_series" not in sparse


class _LeftParameters(Parameters):
    value: int = 1


class _RightParameters(Parameters):
    value: int = 2


class _DuplicateLeafParameters(Parameters):
    left: _LeftParameters = Field(default_factory=_LeftParameters)
    right: _RightParameters = Field(default_factory=_RightParameters)


class _NestedParameters(Parameters):
    child: _LeftParameters = Field(default_factory=_LeftParameters)


class _PresenceParameters(Parameters):
    left: _LeftParameters = Field(default_factory=_LeftParameters)
    right: _RightParameters = Field(default_factory=_RightParameters)
    optional: _LeftParameters | None = Field(default_factory=_LeftParameters)


class _MappingParameters(Parameters):
    payload: dict[str, object] = Field(default_factory=dict)


class _OrderedParameters(Parameters):
    low: int = 1
    high: int = 2

    @model_validator(mode="after")
    def validate_order(self):
        if self.low >= self.high:
            raise ValueError("low must be less than high")
        return self


class _ValidatedSafeSynthesizerParameters(SafeSynthesizerParameters):
    validation_runs: ClassVar[int] = 0

    @model_validator(mode="after")
    def record_validation(self):
        type(self).validation_runs += 1
        return self


def test_explicit_patch_captures_sparse_nested_in_place_mutation():
    source = _DuplicateLeafParameters()
    source.left.value = 17

    result = source.explicit_patch().apply()

    assert result.model_dump(exclude_unset=True) == {"left": {"value": 17}}


@pytest.mark.parametrize(
    ("source", "expected_fields"),
    [
        pytest.param(None, set(), id="none"),
        pytest.param({"value": 5, "ignored": True}, {"value"}, id="mapping"),
        pytest.param(_LeftParameters(value=7), {"value"}, id="typed"),
    ],
)
def test_from_config_source_normalizes_supported_sources(
    source: _LeftParameters | Mapping[str, object] | None, expected_fields: set[str]
):
    result = _LeftParameters.from_config_source(source)

    assert result.model_fields_set == expected_fields


def test_from_config_source_rejects_wrong_exact_model_type():
    with pytest.raises(TypeError, match=r"Expected _LeftParameters, got _RightParameters"):
        _LeftParameters.from_config_source(_RightParameters())  # ty: ignore[invalid-argument-type]


def test_from_config_source_kwargs_override_source_and_preserve_nested_siblings():
    result = _DuplicateLeafParameters.from_config_source(
        {"left": {"value": 4}, "right": {"value": 6}},
        left={"value": 9},
    )

    assert result.left.value == 9
    assert result.right.value == 6


def test_from_config_source_kwargs_resolve_dotted_nested_parameter_name():
    result = _NestedParameters.from_config_source(
        **{"child.value": 9}  # ty: ignore[invalid-argument-type] -- dotted names require dynamic keywords
    )

    assert result.child.value == 9
    assert result.model_dump(exclude_unset=True) == {"child": {"value": 9}}


def test_from_config_source_rejects_inferred_bare_nested_parameter_name():
    with pytest.raises(ParameterError, match=r"Nested parameter name 'value'.*child\.value.*child"):
        _NestedParameters.from_config_source(value=9)


def test_from_config_source_rejects_unknown_keyword_override():
    with pytest.raises(ParameterError, match="Unknown parameter name 'unknown'"):
        _NestedParameters.from_config_source(unknown=9)


def test_from_config_source_rejects_ambiguous_keyword_override():
    with pytest.raises(ParameterError, match=r"Ambiguous parameter name 'value'.*left\.value.*right\.value"):
        _DuplicateLeafParameters.from_config_source(value=9)


@pytest.mark.parametrize(
    ("model_type", "name", "expected"),
    [
        pytest.param(
            GenerateParameters,
            "use_structured_generation",
            ParameterPath(("structured_generation", "enabled")),
            id="section",
        ),
        pytest.param(
            SafeSynthesizerParameters,
            "use_structured_generation",
            ParameterPath(("generation", "structured_generation", "enabled")),
            id="nested-bare",
        ),
        pytest.param(
            SafeSynthesizerParameters,
            "generation.use_structured_generation",
            ParameterPath(("generation", "structured_generation", "enabled")),
            id="nested-dotted",
        ),
    ],
)
def test_parameter_schema_resolves_model_declared_aliases(
    model_type: type[Parameters], name: str, expected: ParameterPath
):
    assert ParameterSchema.from_model(model_type).resolve(name) == ResolvedParameterName(expected)


def test_alias_normalization_preserves_sparse_typed_canonical_branch():
    result = GenerateParameters.from_config_source(
        {
            "structured_generation": StructuredGenerationParameters(backend="guidance"),
            "use_structured_generation": True,
        }
    )

    assert result.model_dump(exclude_unset=True) == {"structured_generation": {"enabled": True, "backend": "guidance"}}


def test_from_config_source_copies_mapping_and_returned_mutable_state():
    nested = {"items": [1]}
    source = {"payload": nested}

    result = _MappingParameters.from_config_source(source)
    nested["items"].append(2)  # type: ignore[union-attr]
    cast(list[int], result.payload["items"]).append(3)

    assert source == {"payload": {"items": [1, 2]}}


def test_from_config_source_copies_typed_source_state():
    source = _MappingParameters(payload={"items": [1]})

    result = _MappingParameters.from_config_source(source)
    cast(list[int], result.payload["items"]).append(2)

    assert source.payload == {"items": [1]}


def test_mapping_valued_atomic_leaf_is_not_parsed_as_model_branch():
    result = _MappingParameters.from_config_source({"payload": {"unknown": {"nested": True}}})

    assert result.payload == {"unknown": {"nested": True}}


def test_apply_patch_preserves_sparse_base_and_runs_validation():
    base = _OrderedParameters(low=3, high=5)
    patch = _OrderedParameters.from_config_source({"high": 4}).explicit_patch()

    result = base.apply_patch(patch)

    assert result.model_dump(exclude_unset=True) == {"low": 3, "high": 4}
    with pytest.raises(ValidationError, match="low must be less than high"):
        base.apply_patch(_OrderedParameters.from_config_source({"high": 2}).explicit_patch())


def test_apply_empty_patch_preserves_recursive_explicit_fields():
    base = _PresenceParameters()
    base.left.value = 17

    result = base.apply_patch(_PresenceParameters().explicit_patch())

    assert result.left.value == 17
    assert result.model_fields_set == set()
    assert result.left.model_fields_set == {"value"}
    assert result.right.model_fields_set == set()
    assert result.optional is not None
    assert result.optional.model_fields_set == set()
    assert result.model_dump(exclude_unset=True) == {}


def test_apply_nonempty_patch_adds_only_patch_explicit_fields():
    base = _PresenceParameters.from_config_source({"left": {}})
    patch = _PresenceParameters.from_config_source(
        {"left": {"value": 1}, "right": {}, "optional": None}
    ).explicit_patch()

    result = base.apply_patch(patch)

    assert result.model_fields_set == {"left", "right", "optional"}
    assert result.left.model_fields_set == {"value"}
    assert result.right.model_fields_set == set()
    assert result.optional is None
    assert result.model_dump(exclude_unset=True) == {
        "left": {"value": 1},
        "right": {},
        "optional": None,
    }


def test_apply_patch_rejects_wrong_exact_target_model():
    with pytest.raises(TypeError, match=r"target model is _RightParameters.*_LeftParameters"):
        _LeftParameters().apply_patch(
            _RightParameters(value=3).explicit_patch()  # ty: ignore[invalid-argument-type] -- runtime rejection tested
        )


def test_parameters_get_supports_explicit_dotted_paths():
    params = _DuplicateLeafParameters()

    assert params.get("left.value") == 1
    assert params.get("right.value") == 2
    assert params.get("missing.value", "fallback") == "fallback"


def test_parameters_get_rejects_ambiguous_bare_leaf_names():
    params = _DuplicateLeafParameters()

    with pytest.raises(ParameterError, match=r"left\.value.*right\.value"):
        params.get("value")


def test_parameters_has_supports_explicit_dotted_paths():
    params = _DuplicateLeafParameters()

    assert params.has("left.value") is True
    assert params.has("right.value") is True
    assert params.has("missing.value") is False


def test_parameters_has_rejects_ambiguous_bare_leaf_names():
    params = _DuplicateLeafParameters()

    with pytest.raises(ParameterError, match=r"left\.value.*right\.value"):
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
    """Tests for supported resume-time override merging."""

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
            pytest.param(
                lambda: SafeSynthesizerParameters.model_validate({"unknown_fields": "ignore"}),
                {"unknown_fields": "ignore"},
                id="unknown-fields-set-applied",
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

    def test_ignores_explicit_non_runtime_sections(self):
        saved = _saved_config()
        runtime = SafeSynthesizerParameters()
        runtime.training.batch_size = 99
        assert runtime.privacy is not None
        runtime.privacy.dp_enabled = True

        merged = saved.with_runtime_overrides(runtime)

        assert merged.training.batch_size == 8
        assert merged.privacy == saved.privacy

    def test_runs_top_level_validation_once(self):
        saved = _ValidatedSafeSynthesizerParameters()
        runtime = SafeSynthesizerParameters.model_validate({"generation": {"num_records": 25}})
        _ValidatedSafeSynthesizerParameters.validation_runs = 0

        merged = saved.with_runtime_overrides(runtime)

        assert isinstance(merged, _ValidatedSafeSynthesizerParameters)
        assert _ValidatedSafeSynthesizerParameters.validation_runs == 1

    def test_preserves_saved_implicit_telemetry_when_environment_changes(self, monkeypatch):
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "false")
        saved = SafeSynthesizerParameters()
        monkeypatch.setenv("NEMO_TELEMETRY_ENABLED", "true")

        merged = saved.with_runtime_overrides(SafeSynthesizerParameters())

        assert saved.emit_telemetry is False
        assert merged.emit_telemetry is False

    def test_empty_overlay_preserves_recursive_explicit_fields(self):
        saved = SafeSynthesizerParameters()
        saved.generation.num_records = 3000

        merged = saved.with_runtime_overrides(SafeSynthesizerParameters())

        assert merged.model_fields_set == saved.model_fields_set == set()
        assert merged.data.max_sequences_per_example == saved.data.max_sequences_per_example == 10
        assert merged.data.model_fields_set == saved.data.model_fields_set == {"max_sequences_per_example"}
        assert merged.generation.model_fields_set == saved.generation.model_fields_set == {"num_records"}
        assert merged.evaluation.model_fields_set == saved.evaluation.model_fields_set == set()
        assert merged.model_dump(exclude_unset=True) == saved.model_dump(exclude_unset=True) == {}

    def test_nonempty_overlay_adds_only_allowlisted_patch_fields(self):
        saved = SafeSynthesizerParameters.model_validate({"training": {"batch_size": 8}, "generation": {}})
        runtime = SafeSynthesizerParameters.model_validate(
            {"generation": {"num_records": 1000}, "evaluation": {}, "emit_telemetry": False}
        )

        merged = saved.with_runtime_overrides(runtime)

        assert merged.model_fields_set == {"training", "generation", "evaluation", "emit_telemetry"}
        assert merged.training.model_fields_set == {"batch_size"}
        assert merged.generation.model_fields_set == {"num_records"}
        assert merged.evaluation.model_fields_set == set()
        assert merged.model_dump(exclude_unset=True) == {
            "training": {"batch_size": 8},
            "generation": {"num_records": 1000},
            "evaluation": {},
            "emit_telemetry": False,
        }

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


class TestWarmupSteps:
    """`warmup_steps` mirrors the transformers contract: ratio below 1, whole steps at or above 1."""

    @pytest.mark.parametrize("value", [0, 0.05, 0.5, 0.999, 1, 1.0, 10, 10.0, 500])
    def test_accepts_ratios_and_whole_step_counts(self, value):
        """0 is legal and disables warmup, matching how transformers treats it."""
        assert TrainingHyperparams(warmup_steps=value).warmup_steps == value

    @pytest.mark.parametrize(
        "value",
        [
            1.5,  # transformers truncates to 1, so reject rather than silently change the schedule
            2.7,
            float("inf"),  # OverflowError once transformers converts it to an int
            float("-inf"),
            float("nan"),
            -1,
        ],
    )
    def test_rejects_fractional_non_finite_and_negative(self, value):
        with pytest.raises((ParameterError, ValidationError, ValueError)):
            TrainingHyperparams(warmup_steps=value)

    def test_default_is_a_ratio(self):
        assert TrainingHyperparams().warmup_steps == 0.05


class TestWarmupRatioDeprecation:
    """`warmup_ratio` stays accepted as a deprecated alias for `warmup_steps`."""

    def test_migrates_to_warmup_steps_and_warns(self):
        with pytest.warns(DeprecationWarning, match="warmup_ratio is deprecated"):
            params = TrainingHyperparams(warmup_ratio=0.2)
        assert params.warmup_steps == 0.2

    def test_explicit_warmup_steps_wins_over_deprecated_alias(self):
        with pytest.warns(DeprecationWarning):
            params = TrainingHyperparams(warmup_steps=0.3, warmup_ratio=0.2)
        assert params.warmup_steps == 0.3

    @pytest.mark.parametrize("value", [1.5, float("inf")])
    def test_deprecated_alias_is_validated_too(self, value):
        """The alias assigns to `warmup_steps` after field validation, so it needs its own guard."""
        with pytest.raises((ParameterError, ValidationError, ValueError)):
            TrainingHyperparams(warmup_ratio=value)

    def test_not_serialized(self):
        with pytest.warns(DeprecationWarning):
            params = TrainingHyperparams(warmup_ratio=0.2)
        assert "warmup_ratio" not in params.model_dump()
