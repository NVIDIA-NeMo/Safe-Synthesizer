# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, cast

import pandas as pd
import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    EvaluationParameters,
    GenerateParameters,
    PiiReplacerConfig,
    PreflightParameters,
    SafeSynthesizerParameters,
    TimeSeriesParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.configurator.parameters import Parameters
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.sdk.config_builder import ConfigBuilder


@pytest.mark.parametrize("as_mapping", [False, True], ids=["typed", "mapping"])
@pytest.mark.parametrize(
    ("method_name", "model_type", "values", "field", "expected"),
    [
        pytest.param("with_data", DataParameters, {"holdout": 0.2}, "holdout", 0.2, id="data"),
        pytest.param("with_train", TrainingHyperparams, {"batch_size": 4}, "batch_size", 4, id="train"),
        pytest.param("with_generate", GenerateParameters, {"num_records": 12}, "num_records", 12, id="generate"),
        pytest.param(
            "with_time_series",
            TimeSeriesParameters,
            {"is_timeseries": True, "timestamp_interval_seconds": 60},
            "is_timeseries",
            True,
            id="time-series",
        ),
        pytest.param(
            "with_differential_privacy",
            DifferentialPrivacyHyperparams,
            {"epsilon": 3.0},
            "epsilon",
            3.0,
            id="privacy",
        ),
        pytest.param(
            "with_evaluate", EvaluationParameters, {"mia_enabled": False}, "mia_enabled", False, id="evaluate"
        ),
    ],
)
def test_builder_methods_accept_typed_and_mapping_sources(
    as_mapping: bool,
    method_name: str,
    model_type: type[Parameters],
    values: dict[str, object],
    field: str,
    expected: object,
):
    source: Parameters | Mapping[str, object] = values if as_mapping else model_type.model_validate(values)

    builder = getattr(ConfigBuilder(), method_name)(config=source)
    section_name = {
        "with_data": "_data_config",
        "with_train": "_training_config",
        "with_generate": "_generation_config",
        "with_time_series": "_time_series_config",
        "with_differential_privacy": "_privacy_config",
        "with_evaluate": "_evaluation_config",
    }[method_name]

    assert getattr(getattr(builder, section_name), field) == expected


@pytest.mark.parametrize(
    "method_name",
    ["with_data", "with_train", "with_generate", "with_time_series", "with_differential_privacy", "with_evaluate"],
)
def test_builder_methods_reject_wrong_typed_model(method_name: str):
    wrong = GenerateParameters() if method_name != "with_generate" else DataParameters()

    with pytest.raises(TypeError, match="Expected"):
        getattr(ConfigBuilder(), method_name)(config=wrong)


def test_builder_rejects_unknown_raw_mapping_keys_by_default():
    with pytest.raises(ParameterError, match="epoch"):
        ConfigBuilder().with_train({"epoch": 1})


def test_builder_ignore_policy_ignores_unknown_raw_mapping_keys():
    builder = ConfigBuilder(unknown_fields="ignore").with_train({"batch_size": 32, "epoch": 1})

    assert builder._effective_unknown_fields == "ignore"
    assert builder._training_config == TrainingHyperparams(batch_size=32)


def test_builder_resolves_and_preserves_unknown_field_policy():
    builder = ConfigBuilder(unknown_fields="ignore").with_data_source(pd.DataFrame({"value": [1]})).resolve()

    assert builder._nss_config is not None
    assert builder._nss_config.unknown_fields == "ignore"


def test_builder_constructor_policy_overrides_seed_config_for_raw_mappings():
    seed = SafeSynthesizerParameters(unknown_fields="ignore")

    with pytest.raises(ParameterError, match="epoch"):
        ConfigBuilder(seed, unknown_fields="reject").with_train({"epoch": 1})


def test_with_generate_validates_raw_config_with_kwargs():
    with pytest.raises(ValidationError, match="patience"):
        ConfigBuilder().with_generate(config={"num_records": 10}, patience=0)


def test_with_generate_validates_typed_config_with_kwargs():
    with pytest.raises(ValidationError, match="patience"):
        ConfigBuilder().with_generate(config=GenerateParameters(num_records=10), patience=0)


def test_with_generate_preserves_sparse_typed_config_fields():
    builder = ConfigBuilder().with_generate(config=GenerateParameters(num_records=10))

    assert builder._generation_config is not None
    assert builder._generation_config.model_fields_set == {"num_records"}


def test_with_generate_marks_typed_config_kwargs_as_explicit_fields():
    builder = ConfigBuilder().with_generate(config=GenerateParameters(num_records=10), patience=7)

    assert builder._generation_config is not None
    assert builder._generation_config.model_fields_set == {"num_records", "patience"}


def test_with_generate_accepts_legacy_alias_keyword():
    builder = ConfigBuilder().with_generate(use_structured_generation=True)

    assert builder._generation_config.structured_generation.enabled is True
    assert builder._generation_config.model_dump(exclude_unset=True) == {"structured_generation": {"enabled": True}}


def test_with_generate_accepts_legacy_alias_in_raw_mapping():
    builder = ConfigBuilder().with_generate(config={"use_structured_generation": True})

    assert builder._generation_config.structured_generation.enabled is True


def test_with_generate_legacy_mapping_alias_overrides_canonical_value():
    builder = ConfigBuilder().with_generate(
        config={
            "structured_generation": {"enabled": False},
            "use_structured_generation": True,
        }
    )

    assert builder._generation_config.structured_generation.enabled is True


def test_with_generate_rejects_duplicate_alias_keyword_path():
    with pytest.raises(ParameterError, match=r"Duplicate parameter path 'structured_generation\.enabled'"):
        ConfigBuilder().with_generate(
            **{
                "structured_generation.enabled": False,
                "use_structured_generation": True,
            }  # ty: ignore[invalid-argument-type] -- dotted names require dynamic keywords
        )


def test_with_generate_rejects_wrong_typed_config_object():
    wrong_config = cast(Any, PiiReplacerConfig())

    with pytest.raises(TypeError, match="Expected GenerateParameters"):
        ConfigBuilder().with_generate(config=wrong_config)


def test_with_replace_pii_resolves_raw_config_with_kwargs():
    builder = ConfigBuilder().with_replace_pii(
        config={"replacement": {"locale": "en_GB"}},
    )

    assert builder._replace_pii_config is not None
    assert builder._replace_pii_config.replacement.locale == "en_GB"


@pytest.mark.parametrize("as_mapping", [False, True])
def test_with_replace_pii_deep_merges_nested_kwargs(as_mapping: bool):
    config_model = PiiReplacerConfig()
    config_model.replacement.locale = "en_GB"
    config = config_model.model_dump() if as_mapping else config_model

    builder = ConfigBuilder().with_replace_pii(
        config=config,
        person={"backend": "faker"},
    )

    assert builder._replace_pii_config is not None
    assert builder._replace_pii_config.replacement.locale == "en_GB"
    assert builder._replace_pii_config.person.backend.value == "faker"


def test_with_replace_pii_none_uses_defaults_with_nested_kwargs():
    builder = ConfigBuilder().with_replace_pii(person={"backend": "faker"})

    assert builder._replace_pii_config is not None
    assert builder._replace_pii_config.person.backend.value == "faker"


def test_with_replace_pii_invalid_source_preserves_value_error_contract():
    with pytest.raises(ValueError, match="Config must be"):
        ConfigBuilder().with_replace_pii(config=GenerateParameters())  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("key", ["globals", "steps"])
def test_with_replace_pii_rejects_legacy_keys_under_ignore(key: str):
    """Legacy replace_pii keys must error on the direct PiiReplacerConfig SDK path."""
    with pytest.raises(ParameterError, match=rf"replace_pii\.{key}"):
        ConfigBuilder(unknown_fields="ignore").with_replace_pii({key: {"anything": True}})


def test_with_generate_captures_nested_mutation_on_sparse_typed_source():
    source = GenerateParameters()
    source.validation.group_by_fix_unordered_records = True

    builder = ConfigBuilder().with_generate(config=source)

    assert builder._generation_config.model_dump(exclude_unset=True) == {
        "validation": {"group_by_fix_unordered_records": True}
    }


def test_resolve_runs_top_level_validation_for_direct_typed_assembly():
    builder = (
        ConfigBuilder()
        .with_data_source(pd.DataFrame({"value": [1]}))
        .with_data(max_sequences_per_example=2)
        .with_differential_privacy(dp_enabled=True)
    )

    with pytest.raises(ValidationError, match="max_sequences_per_example must be 1"):
        builder.resolve()


def test_resolved_config_is_independent_of_mapping_source():
    entities = ["email"]
    source = {"pii_replay_entities": entities}

    builder = ConfigBuilder().with_data_source(pd.DataFrame({"value": [1]})).with_evaluate(config=source).resolve()
    entities.append("phone_number")

    assert builder._nss_config is not None
    assert builder._nss_config.evaluation.pii_replay_entities == ["email"]


def test_resolve_preserves_preflight_from_existing_config():
    config = SafeSynthesizerParameters(preflight=PreflightParameters(disabled_checks=["gpu.vram"]))

    builder = ConfigBuilder(config).with_data_source(pd.DataFrame({"value": [1]})).resolve()

    assert builder._nss_config is not None
    assert builder._nss_config.preflight.disabled_checks == ["gpu.vram"]


def test_direct_assembly_preserves_classify_model_provider_injection():
    builder = ConfigBuilder().with_data_source(pd.DataFrame({"value": [1]}))
    builder._classify_model_provider = "test-provider"

    builder.resolve()

    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None
    assert builder._nss_config.replace_pii.llm.model_provider == "test-provider"
