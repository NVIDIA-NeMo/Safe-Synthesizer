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
    TimeSeriesParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
from nemo_safe_synthesizer.configurator.parameters import Parameters
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


def test_with_generate_validates_raw_config_with_kwargs():
    with pytest.raises(ValidationError, match="patience"):
        ConfigBuilder().with_generate(config={"num_records": 10}, patience=0)


def test_with_generate_validates_typed_config_with_kwargs():
    with pytest.raises(ValidationError, match="patience"):
        ConfigBuilder().with_generate(config=GenerateParameters(num_records=10), patience=0)


def test_with_generate_preserves_sparse_typed_config_fields():
    builder = ConfigBuilder().with_generate(config=GenerateParameters(num_records=10))

    assert builder._generation_config is not None
    assert builder._generation_config.__pydantic_fields_set__ == {"num_records"}


def test_with_generate_marks_typed_config_kwargs_as_explicit_fields():
    builder = ConfigBuilder().with_generate(config=GenerateParameters(num_records=10), patience=7)

    assert builder._generation_config is not None
    assert builder._generation_config.__pydantic_fields_set__ == {"num_records", "patience"}


def test_with_generate_rejects_wrong_typed_config_object():
    wrong_config = cast(Any, PiiReplacerConfig.get_default_config())

    with pytest.raises(TypeError, match="Expected GenerateParameters"):
        ConfigBuilder().with_generate(config=wrong_config)


def test_with_replace_pii_validates_default_config_with_kwargs():
    with pytest.raises(ValidationError, match="Invalid locale"):
        ConfigBuilder().with_replace_pii(globals={"locales": ["not-a-locale"]})


def test_with_replace_pii_resolves_raw_config_with_kwargs():
    builder = ConfigBuilder().with_replace_pii(
        config=PiiReplacerConfig.get_default_config().model_dump(),
        globals={"classify": {"enable_classify": False}},
    )

    assert builder._replace_pii_config is not None
    assert builder._replace_pii_config.globals.classify.enable_classify is False


@pytest.mark.parametrize("as_mapping", [False, True])
def test_with_replace_pii_deep_merges_nested_kwargs(as_mapping: bool):
    config_model = PiiReplacerConfig.get_default_config()
    config_model.globals.locales = ["en_US"]
    config = config_model.model_dump() if as_mapping else config_model

    builder = ConfigBuilder().with_replace_pii(
        config=config,
        globals={"classify": {"enable_classify": False}},
    )

    assert builder._replace_pii_config is not None
    assert builder._replace_pii_config.globals.locales == ["en_US"]
    assert builder._replace_pii_config.globals.classify.enable_classify is False
    assert builder._replace_pii_config.steps[0].vars == config_model.steps[0].vars


def test_with_replace_pii_none_uses_defaults_and_preserves_step_vars_with_nested_kwargs():
    default = PiiReplacerConfig.get_default_config()

    builder = ConfigBuilder().with_replace_pii(globals={"classify": {"enable_classify": False}})

    assert builder._replace_pii_config is not None
    assert builder._replace_pii_config.globals.locales == default.globals.locales
    assert builder._replace_pii_config.steps[0].vars == default.steps[0].vars


def test_with_replace_pii_invalid_source_preserves_value_error_contract():
    with pytest.raises(ValueError, match="Config must be"):
        ConfigBuilder().with_replace_pii(config=GenerateParameters())  # ty: ignore[invalid-argument-type]


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


def test_direct_assembly_preserves_classify_model_provider_injection():
    builder = ConfigBuilder().with_data_source(pd.DataFrame({"value": [1]}))
    builder._classify_model_provider = "test-provider"

    builder.resolve()

    assert builder._nss_config is not None
    assert builder._nss_config.replace_pii is not None
    assert builder._nss_config.replace_pii.globals.classify.classify_model_provider == "test-provider"
