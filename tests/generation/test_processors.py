# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.generation.processors import (
    GroupedDataProcessor,
    TabularDataProcessor,
    create_processor,
)
from nemo_safe_synthesizer.training.backend import ModelMetadata
from transformers import AutoTokenizer, PreTrainedTokenizer

BOS = "<s>"
EOS = "</s>"


# Purpose: Loads the stub tokenizer from disk for tests that require a tokenizer instance.
@pytest.fixture
def fixture_tokenizer(fixture_save_path) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(fixture_save_path)


@pytest.fixture(scope="session")
def fixture_save_path(fixture_session_cache_dir: Path) -> Path:
    return fixture_session_cache_dir / "processors"


# Purpose: Builds training metadata for tests, using the stub tokenizer and explicit params.
@pytest.fixture
def fixture_metadata(
    fixture_save_path,
) -> ModelMetadata:
    config = SafeSynthesizerParameters.from_params(use_unsloth=False, rope_scaling_factor=1)
    metadata = ModelMetadata.from_str_or_path(config.training.pretrained_model, save_path=fixture_save_path)
    return metadata


# Purpose: Validates TabularDataProcessor parses valid JSONL rows per schema.
# Data: First 5 Iris rows (JSONL string) with corresponding schema.
# Asserts: 5 valid records; 0 invalid/errors; correct prompt_number.
def test_tabular_data_processor(fixture_valid_iris_dataset_jsonl_and_schema):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    response = TabularDataProcessor(schema=jsonl_schema)(1, jsonl_str)
    assert len(response.valid_records) == 5
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0
    assert response.prompt_number == 1


# Purpose: GroupedDataProcessor should reject input not wrapped in BOS/EOS group blocks.
# Data: Raw JSONL without BOS/EOS.
# Asserts: 0 valid; entire input counted as 1 invalid; one error emitted.
def test_grouped_data_processor_with_no_groups(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    response = GroupedDataProcessor(schema=jsonl_schema, group_by="variety", bos_token=BOS, eos_token=EOS)(1, jsonl_str)
    assert len(response.valid_records) == 0
    assert len(response.invalid_records) == 1
    assert len(response.errors) == 1
    assert response.invalid_records == [jsonl_str]


# Purpose: Mixed validity across groups: accept the valid group and reject the invalid one.
# Data: Two groups: one valid (BOS + 5 records + EOS), one with extra invalid JSON appended.
# Asserts: 5 valid; 6 invalid/errors; final error details invalid JSON in other group.
def test_grouped_data_processor_with_invalid_json(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    valid_group = BOS + jsonl_str + EOS
    invalid_group = BOS + jsonl_str + '{"a":1, "b":2}\n' + EOS
    groups_jsonl_str = invalid_group + " " + valid_group
    response = GroupedDataProcessor(schema=jsonl_schema, group_by="variety", bos_token=BOS, eos_token="</s>")(
        1, groups_jsonl_str
    )
    assert len(response.valid_records) == 5
    assert len(response.invalid_records) == 6
    assert len(response.errors) == 6
    assert response.errors[-1] == ("Invalid JSON in other groupby records", "groupby")


# Purpose: Group value must be unique within a group; mixing values invalidates the group.
# Data: Valid group plus one record with a different 'variety'.
# Asserts: 0 valid; 6 invalid/errors; last error indicates non-unique group value.
def test_grouped_data_processor_with_non_unique_group_by(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema

    # Add a new record in the group with a different variety
    jsonl_schema["properties"]["variety"]["enum"].append("NewSetosa")
    groups_jsonl_str = (
        BOS
        + jsonl_str
        + '{"sepal.length":5.1,"sepal.width":3.5,"petal.length":1.4,"petal.width":0.2,"variety":"NewSetosa"}\n'
        + EOS
    )
    response = GroupedDataProcessor(schema=jsonl_schema, group_by="variety", bos_token=BOS, eos_token=EOS)(
        1, groups_jsonl_str
    )
    assert len(response.valid_records) == 0
    assert len(response.invalid_records) == 6
    assert len(response.errors) == 6
    assert response.errors[-1] == ("Groupby value is not unique", "groupby")


# Purpose: Enforces ordering constraint within grouped records when order_by is set.
# Data: Group not sorted by 'sepal.length'.
# Asserts: 0 valid; 5 invalid/errors; last error indicates group not ordered.
def test_grouped_data_processor_out_of_order_records(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    groups_jsonl_str = BOS + jsonl_str + EOS
    response = GroupedDataProcessor(
        schema=jsonl_schema,
        group_by="variety",
        order_by="sepal.length",
        bos_token=BOS,
        eos_token=EOS,
    )(1, groups_jsonl_str)
    assert len(response.valid_records) == 0
    assert len(response.invalid_records) == 5
    assert len(response.errors) == 5
    assert response.errors[-1] == ("Group not ordered", "groupby")


# Purpose: Composite group_by works when the tuple uniquely identifies groups.
# Data: Grouped by ["petal.width", "variety"].
# Asserts: 5 valid; 0 invalid/errors.
def test_grouped_data_processor_multiple_group_by(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    groups_jsonl_str = BOS + jsonl_str + EOS
    response = GroupedDataProcessor(
        schema=jsonl_schema,
        group_by=["petal.width", "variety"],
        bos_token=BOS,
        eos_token=EOS,
    )(1, groups_jsonl_str)
    assert len(response.valid_records) == 5
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0


# Purpose: Composite group_by should fail when the tuple does not uniquely identify groups.
# Data: Grouped by ["sepal.length", "variety"] (non-unique).
# Asserts: 0 valid; 5 invalid/errors; last error indicates non-unique group value.
def test_grouped_data_processor_multiple_group_by_error(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    groups_jsonl_str = BOS + jsonl_str + EOS
    response = GroupedDataProcessor(
        schema=jsonl_schema,
        group_by=["sepal.length", "variety"],
        bos_token=BOS,
        eos_token=EOS,
    )(1, groups_jsonl_str)
    assert len(response.valid_records) == 0
    assert len(response.invalid_records) == 5
    assert len(response.errors) == 5
    assert response.errors[-1] == ("Groupby value is not unique", "groupby")


# Purpose: Factory should return TabularDataProcessor when no grouping is configured.
# Data: Minimal schema + training metadata without group_training_examples_by.
# Asserts: TabularDataProcessor is returned
def test_create_processor_tabular(fixture_metadata):
    stub_schema = {"this": "is_a_stub"}
    config = SafeSynthesizerParameters.from_params(use_unsloth=False, rope_scaling_factor=1)
    assert isinstance(
        create_processor(schema=stub_schema, metadata=fixture_metadata, config=config),
        TabularDataProcessor,
    )


# Purpose: Factory should return GroupedDataProcessor when grouping is configured in training metadata.
# Data: Metadata with group_training_examples_by set; valid pretrained model.
# Asserts: GroupedDataProcessor is returned
def test_create_processor_grouped(fixture_metadata):
    stub_schema = {"this": "is_a_stub"}

    config = SafeSynthesizerParameters.from_params(
        use_unsloth=False, rope_scaling_factor=1, group_training_examples_by="patient_id"
    )
    assert isinstance(
        create_processor(
            schema=stub_schema,
            metadata=fixture_metadata,
            config=config,
        ),
        GroupedDataProcessor,
    )
