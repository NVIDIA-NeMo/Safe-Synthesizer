# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.generate import ValidationParameters
from nemo_safe_synthesizer.data_processing.assembler import TrainingExampleAssembler
from nemo_safe_synthesizer.data_processing.dataset import make_json_schema
from nemo_safe_synthesizer.generation.processors import (
    GroupedDataProcessor,
    TabularDataProcessor,
    TimeSeriesDataProcessor,
    create_processor,
)
from nemo_safe_synthesizer.observability import get_logger
from nemo_safe_synthesizer.training.backend import ModelMetadata

logger = get_logger(__name__)

BOS = "<s>"
EOS = "</s>"

TOKENIZERS_DIR = Path(__file__).parent.parent / "test_data" / "tokenizers"


@pytest.fixture(scope="session")
def fixture_save_path(fixture_session_cache_dir: Path) -> Path:
    return fixture_session_cache_dir / "processors"


@pytest.fixture(
    params=["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "HuggingFaceTB/SmolLM3-3B", "mistralai/Mistral-7B-Instruct-v0.3"],
    ids=["tinyllama", "smollm3", "mistral"],
)
def fixture_over_tokenizers(request) -> tuple[str, PreTrainedTokenizer]:
    """Fixture parameterized over multiple tokenizers of interest."""
    model_name = request.param

    repo_name = model_name
    local_files_only = False
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        repo_name = str(TOKENIZERS_DIR / "tinyllama")
        local_files_only = True
    elif model_name == "HuggingFaceTB/SmolLM3-3B":
        repo_name = str(TOKENIZERS_DIR / "smollm3b")
        local_files_only = True
    elif model_name == "mistralai/Mistral-7B-Instruct-v0.3":
        repo_name = str(TOKENIZERS_DIR / "mistral7b")
        local_files_only = True
    else:
        logger.warning(f"Tokenizer for {model_name} not stored in repo, loading from HF Hub")

    tokenizer = AutoTokenizer.from_pretrained(repo_name, local_files_only=local_files_only)

    return model_name, tokenizer


# Purpose: Builds training metadata for tests, using the stub tokenizer and explicit params.
@pytest.fixture
def fixture_metadata(
    fixture_save_path,
) -> ModelMetadata:
    config = SafeSynthesizerParameters.from_params(use_unsloth=False, rope_scaling_factor=1)
    metadata = ModelMetadata.from_str_or_path(config.training.pretrained_model, save_path=fixture_save_path)
    return metadata


@pytest.fixture
def fixture_validation_config() -> ValidationParameters:
    return ValidationParameters(
        group_by_accept_no_delineator=False,
        group_by_ignore_invalid_records=False,
        group_by_fix_non_unique_value=False,
        group_by_fix_unordered_records=False,
    )


# Purpose: Validates TabularDataProcessor parses valid JSONL rows per schema.
# Data: First 5 Iris rows (JSONL string) with corresponding schema.
# Asserts: 5 valid records; 0 invalid/errors; correct prompt_number.
def test_tabular_data_processor(
    fixture_valid_iris_dataset_jsonl_and_schema, fixture_validation_config: ValidationParameters
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    response = TabularDataProcessor(schema=jsonl_schema, config=fixture_validation_config)(1, jsonl_str)
    assert len(response.valid_records) == 5
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0
    assert response.prompt_number == 1


# Purpose: GroupedDataProcessor should reject input not wrapped in BOS/EOS group blocks.
# Data: Raw JSONL without BOS/EOS.
# Asserts: 0 valid; entire input counted as 1 invalid; one error emitted.
def test_grouped_data_processor_with_no_groups(
    fixture_valid_iris_dataset_jsonl_and_schema,
    fixture_validation_config: ValidationParameters,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    response = GroupedDataProcessor(
        schema=jsonl_schema, config=fixture_validation_config, group_by="variety", bos_token=BOS, eos_token=EOS
    )(1, jsonl_str)
    assert len(response.valid_records) == 0
    assert len(response.invalid_records) == 1
    assert len(response.errors) == 1
    assert response.invalid_records == [jsonl_str]


# Purpose: Mixed validity across groups: accept the valid group and reject the invalid one.
# Data: Two groups: one valid (BOS + 5 records + EOS), one with extra invalid JSON appended.
# Asserts: 5 valid; 6 invalid/errors; final error details invalid JSON in other group.
def test_grouped_data_processor_with_invalid_json(
    fixture_valid_iris_dataset_jsonl_and_schema,
    fixture_validation_config: ValidationParameters,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    valid_group = BOS + jsonl_str + EOS
    invalid_group = BOS + jsonl_str + '{"a":1, "b":2}\n' + EOS
    groups_jsonl_str = invalid_group + " " + valid_group
    response = GroupedDataProcessor(
        schema=jsonl_schema, config=fixture_validation_config, group_by="variety", bos_token=BOS, eos_token="</s>"
    )(1, groups_jsonl_str)
    assert len(response.valid_records) == 5
    assert len(response.invalid_records) == 6
    assert len(response.errors) == 6
    assert response.errors[-1] == ("Invalid JSON in other groupby records", "groupby")


# Purpose: Group value must be unique within a group; mixing values invalidates the group.
# Data: Valid group plus one record with a different 'variety'.
# Asserts: 0 valid; 6 invalid/errors; last error indicates non-unique group value.
def test_grouped_data_processor_with_non_unique_group_by(
    fixture_valid_iris_dataset_jsonl_and_schema,
    fixture_validation_config: ValidationParameters,
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
    response = GroupedDataProcessor(
        schema=jsonl_schema, config=fixture_validation_config, group_by="variety", bos_token=BOS, eos_token=EOS
    )(1, groups_jsonl_str)
    assert len(response.valid_records) == 0
    assert len(response.invalid_records) == 6
    assert len(response.errors) == 6
    assert response.errors[-1] == ("Groupby value is not unique", "groupby")


# Purpose: Enforces ordering constraint within grouped records when order_by is set.
# Data: Group not sorted by 'sepal.length'.
# Asserts: 0 valid; 5 invalid/errors; last error indicates group not ordered.
def test_grouped_data_processor_out_of_order_records(
    fixture_valid_iris_dataset_jsonl_and_schema,
    fixture_validation_config: ValidationParameters,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    groups_jsonl_str = BOS + jsonl_str + EOS
    response = GroupedDataProcessor(
        schema=jsonl_schema,
        config=fixture_validation_config,
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
    fixture_validation_config: ValidationParameters,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    groups_jsonl_str = BOS + jsonl_str + EOS
    response = GroupedDataProcessor(
        schema=jsonl_schema,
        config=fixture_validation_config,
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
    fixture_validation_config: ValidationParameters,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    groups_jsonl_str = BOS + jsonl_str + EOS
    response = GroupedDataProcessor(
        schema=jsonl_schema,
        config=fixture_validation_config,
        group_by=["sepal.length", "variety"],
        bos_token=BOS,
        eos_token=EOS,
    )(1, groups_jsonl_str)
    assert len(response.valid_records) == 0
    assert len(response.invalid_records) == 5
    assert len(response.errors) == 5
    assert response.errors[-1] == ("Groupby value is not unique", "groupby")


# Purpose: group_by_accept_no_delineator=True treats raw JSONL (no BOS/EOS) as a single group.
# Data: Raw JSONL without BOS/EOS (same as test_grouped_data_processor_with_no_groups).
# Asserts: 5 valid; 0 invalid; 0 errors.
def test_grouped_data_processor_accept_no_delineator(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    config = ValidationParameters(
        group_by_accept_no_delineator=True,
        group_by_ignore_invalid_records=False,
        group_by_fix_non_unique_value=False,
        group_by_fix_unordered_records=False,
    )
    response = GroupedDataProcessor(
        schema=jsonl_schema, config=config, group_by="variety", bos_token=BOS, eos_token=EOS
    )(1, jsonl_str)
    assert len(response.valid_records) == 5
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0


# Purpose: group_by_ignore_invalid_records=True keeps only valid records in a group with mixed validity.
# Data: A group with one invalid record and 5 valid JSONL records.
# Asserts: 5 valid; 0 invalid; 0 errors.
def test_grouped_data_processor_ignore_invalid_records(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    group_str = BOS + '{"a":1,"b":2}\n' + jsonl_str + EOS
    config = ValidationParameters(
        group_by_accept_no_delineator=False,
        group_by_ignore_invalid_records=True,
        group_by_fix_non_unique_value=False,
        group_by_fix_unordered_records=False,
    )
    response = GroupedDataProcessor(
        schema=jsonl_schema, config=config, group_by="variety", bos_token=BOS, eos_token=EOS
    )(1, group_str)
    assert len(response.valid_records) == 5
    # Invalid records in the group are ignored and not included in either valid_records or invalid_records.
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0


# Purpose: group_by_fix_non_unique_value=True normalizes group_by to the first record's value.
# Data: One group with 5 Setosa + 1 NewSetosa (same shape as test_grouped_data_processor_with_non_unique_group_by).
# Asserts: 6 valid; all records have same variety as first; 0 invalid; 0 errors.
def test_grouped_data_processor_fix_non_unique_value(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    schema = copy.deepcopy(jsonl_schema)
    schema["properties"]["variety"]["enum"].append("NewSetosa")
    extra = '{"sepal.length":5.1,"sepal.width":3.5,"petal.length":1.4,"petal.width":0.2,"variety":"NewSetosa"}\n'
    groups_jsonl_str = BOS + jsonl_str.strip() + "\n" + extra + EOS
    config = ValidationParameters(
        group_by_accept_no_delineator=False,
        group_by_ignore_invalid_records=False,
        group_by_fix_non_unique_value=True,
        group_by_fix_unordered_records=False,
    )
    response = GroupedDataProcessor(schema=schema, config=config, group_by="variety", bos_token=BOS, eos_token=EOS)(
        1, groups_jsonl_str
    )
    assert len(response.valid_records) == 6
    assert all(r["variety"] == response.valid_records[0]["variety"] for r in response.valid_records)
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0


# Purpose: group_by_fix_unordered_records=True sorts records by order_by instead of rejecting.
# Data: One group not sorted by 'sepal.length' (same as test_grouped_data_processor_out_of_order_records).
# Asserts: 5 valid; sorted by sepal.length; 0 invalid; 0 errors.
def test_grouped_data_processor_fix_unordered_records(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    groups_jsonl_str = BOS + jsonl_str + EOS
    config = ValidationParameters(
        group_by_accept_no_delineator=False,
        group_by_ignore_invalid_records=False,
        group_by_fix_non_unique_value=False,
        group_by_fix_unordered_records=True,
    )
    response = GroupedDataProcessor(
        schema=jsonl_schema,
        config=config,
        group_by="variety",
        order_by="sepal.length",
        bos_token=BOS,
        eos_token=EOS,
    )(1, groups_jsonl_str)
    assert len(response.valid_records) == 5
    assert response.valid_records == sorted(response.valid_records, key=lambda r: r["sepal.length"])
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0


# Purpose: All validation relaxations True; no delimiters + invalid line → accept and ignore.
# Data: Raw JSONL (no BOS/EOS) with 5 valid lines and one invalid line.
# Asserts: 5 valid; 0 invalid; 0 errors.
def test_grouped_data_processor_all_relaxations_no_delineator_and_ignore_invalid(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    text = jsonl_str + '{"a":1,"b":2}\n'
    config = ValidationParameters(
        group_by_accept_no_delineator=True,
        group_by_ignore_invalid_records=True,
        group_by_fix_non_unique_value=True,
        group_by_fix_unordered_records=True,
    )
    response = GroupedDataProcessor(
        schema=jsonl_schema, config=config, group_by="variety", bos_token=BOS, eos_token=EOS
    )(1, text)
    assert len(response.valid_records) == 5
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0


# Purpose: All validation relaxations True; one group with invalid line, non-unique group_by, and wrong order.
# Data: BOS/EOS group with 5 Setosa + 1 NewSetosa (out of order by sepal.length) and one invalid line.
# Asserts: 6 valid; sorted by sepal.length; all same variety; 0 invalid; 0 errors.
def test_grouped_data_processor_all_relaxations_with_fixes(
    fixture_valid_iris_dataset_jsonl_and_schema,
):
    jsonl_str, jsonl_schema = fixture_valid_iris_dataset_jsonl_and_schema
    schema = copy.deepcopy(jsonl_schema)
    schema["properties"]["variety"]["enum"].append("NewSetosa")
    extra = '{"sepal.length":5.1,"sepal.width":3.5,"petal.length":1.4,"petal.width":0.2,"variety":"NewSetosa"}\n'
    invalid_line = '{"a":1,"b":2}\n'
    groups_jsonl_str = BOS + jsonl_str.strip() + "\n" + extra + invalid_line + EOS
    config = ValidationParameters(
        group_by_accept_no_delineator=True,
        group_by_ignore_invalid_records=True,
        group_by_fix_non_unique_value=True,
        group_by_fix_unordered_records=True,
    )
    response = GroupedDataProcessor(
        schema=schema,
        config=config,
        group_by="variety",
        order_by="sepal.length",
        bos_token=BOS,
        eos_token=EOS,
    )(1, groups_jsonl_str)
    assert len(response.valid_records) == 6
    assert response.valid_records == sorted(response.valid_records, key=lambda r: r["sepal.length"])
    assert all(r["variety"] == response.valid_records[0]["variety"] for r in response.valid_records)
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0


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


@pytest.fixture
def fixture_timeseries_schema():
    """Schema for time-series data with a timestamp column and numeric values."""
    return {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string"},
            "value": {"type": "number"},
            "metric": {"type": "string"},
        },
        "required": ["timestamp", "value", "metric"],
    }


@pytest.fixture
def fixture_valid_timeseries_jsonl():
    """Valid time-series JSONL with consistent 60-second intervals."""
    return (
        '{"timestamp": "2024-01-01 00:00:00", "value": 10.5, "metric": "cpu"}\n'
        '{"timestamp": "2024-01-01 00:01:00", "value": 12.3, "metric": "cpu"}\n'
        '{"timestamp": "2024-01-01 00:02:00", "value": 11.8, "metric": "cpu"}\n'
    )


# Purpose: TimeSeriesDataProcessor should parse valid JSONL rows with consistent time intervals.
# Data: 3 time-series records with 60-second intervals.
# Asserts: 3 valid records; 0 invalid/errors; correct prompt_number.
def test_timeseries_data_processor_valid_records(
    fixture_timeseries_schema, fixture_valid_timeseries_jsonl, fixture_validation_config: ValidationParameters
):
    processor = TimeSeriesDataProcessor(
        schema=fixture_timeseries_schema,
        config=fixture_validation_config,
        time_column="timestamp",
        interval_seconds=60,
        time_format="%Y-%m-%d %H:%M:%S",
    )
    response = processor(1, fixture_valid_timeseries_jsonl)
    assert len(response.valid_records) == 3
    assert len(response.invalid_records) == 0
    assert len(response.errors) == 0
    assert response.prompt_number == 1


# Purpose: TimeSeriesDataProcessor should reject records with inconsistent time intervals.
# Data: Records with 60-second expected interval but 120-second actual interval.
# Asserts: 1 valid record; 2 invalid records (current + remaining cascade).
def test_timeseries_data_processor_invalid_interval(
    fixture_timeseries_schema, fixture_validation_config: ValidationParameters
):
    jsonl_str = (
        '{"timestamp": "2024-01-01 00:00:00", "value": 10.5, "metric": "cpu"}\n'
        '{"timestamp": "2024-01-01 00:02:00", "value": 12.3, "metric": "cpu"}\n'  # 120s gap instead of 60s
        '{"timestamp": "2024-01-01 00:03:00", "value": 11.8, "metric": "cpu"}\n'
    )
    processor = TimeSeriesDataProcessor(
        schema=fixture_timeseries_schema,
        config=fixture_validation_config,
        time_column="timestamp",
        interval_seconds=60,
        time_format="%Y-%m-%d %H:%M:%S",
    )
    response = processor(1, jsonl_str)
    assert len(response.valid_records) == 1
    assert len(response.invalid_records) == 2
    assert len(response.errors) == 2


def _check_assembler_to_processor(
    config: SafeSynthesizerParameters,
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizer,
    cache_file_path: str | Path | None,
) -> None:
    """Test helper to run assembler and processor.

    Asserts that the processor can parse all records from the text produced
    by the assembler and extracts out the expected number of records.

    Args:
        config: The configuration for the synthesizer.
        df: The dataframe to use for the test.
        tokenizer: The tokenizer to use for the test.
    """
    schema = make_json_schema(df)

    hf_dataset = Dataset.from_pandas(df)

    metadata = ModelMetadata.from_config(config)
    assembler = TrainingExampleAssembler.from_data(
        dataset=hf_dataset,
        tokenizer=tokenizer,
        metadata=metadata,
        config=config,
        seed=125642,
        cache_file_path=cache_file_path,
    )

    training_examples = assembler.assemble_training_examples(data_fraction=1.0)
    processor = create_processor(schema=schema, metadata=metadata, config=config)

    total_parsed_records = 0
    for idx, example in enumerate(training_examples.train):
        input_ids = example["input_ids"]
        labels = example["labels"]
        assert len(input_ids) == len(labels)
        completion_input_ids = [id for id, label in zip(input_ids, labels) if label != -100]

        text = tokenizer.decode(completion_input_ids)
        response = processor(idx, text)

        assert len(response.valid_records) > 0
        assert len(response.invalid_records) == 0
        assert len(response.errors) == 0
        total_parsed_records += len(response.valid_records)

    assert total_parsed_records == len(df)


# Purpose: Ensure that the tabular examples we assemble are parsable by the processor
# with no invalid records.
@pytest.mark.parametrize("max_sequences_per_example", [1, 2])
def test_assembler_to_processor_tabular(
    fixture_over_tokenizers, fixture_session_cache_dir, max_sequences_per_example: int
):
    model_name, tokenizer = fixture_over_tokenizers
    config = SafeSynthesizerParameters.from_params(
        use_unsloth=True,
        rope_scaling_factor=1,
        max_sequences_per_example=max_sequences_per_example,
        pretrained_model=model_name,
    )
    df = pd.DataFrame(
        {
            "integer": [1, 2, 3, 5, 10],
            "float": [4.5, 5.6, 6.7, 2.3, -1.2],
            "text": ["hello", "world", "foo bar baz", "with\ttabs", "new\nline"],
            "boolean": [True, False, True, False, True],
        }
    )

    _check_assembler_to_processor(config, df, tokenizer, cache_file_path=fixture_session_cache_dir)


# Purpose: Ensure that the grouped examples we assemble are parsable by the processor
# with no invalid records.
@pytest.mark.parametrize("max_sequences_per_example", [1, 2])
def test_assembler_to_processor_grouped(fixture_over_tokenizers, fixture_session_cache_dir, max_sequences_per_example):
    model_name, tokenizer = fixture_over_tokenizers
    config = SafeSynthesizerParameters.from_params(
        use_unsloth=True,
        rope_scaling_factor=1,
        max_sequences_per_example=max_sequences_per_example,
        group_training_examples_by="group_id",
        pretrained_model=model_name,
    )
    df = pd.DataFrame(
        {
            "group_id": [1, 1, 1, 2, 2],
            "integer": [1, 2, 3, 5, 10],
            "float": [4.5, 5.6, 6.7, 2.3, -1.2],
            "text": ["hello", "world", "foo bar baz", "with\ttabs", "new\nline"],
            "boolean": [True, False, True, False, True],
        }
    )

    _check_assembler_to_processor(config, df, tokenizer, cache_file_path=fixture_session_cache_dir)


def test_assembler_to_processor_adobe(
    fixture_over_tokenizers, fixture_session_cache_dir, fixture_adobe_sampled_dataset
):
    model_name, tokenizer = fixture_over_tokenizers
    config = SafeSynthesizerParameters.from_params(
        use_unsloth=True,
        rope_scaling_factor=1,
        max_sequences_per_example=1,
        pretrained_model=model_name,
    )
    _check_assembler_to_processor(
        config=config, df=fixture_adobe_sampled_dataset, tokenizer=tokenizer, cache_file_path=fixture_session_cache_dir
    )


def test_assembler_to_processor_non_english(
    fixture_over_tokenizers, fixture_session_cache_dir, fixture_lmsys_chat_non_english_dataset
):
    model_name, tokenizer = fixture_over_tokenizers
    config = SafeSynthesizerParameters.from_params(
        use_unsloth=True,
        rope_scaling_factor=1,
        max_sequences_per_example=1,
        pretrained_model=model_name,
    )
    _check_assembler_to_processor(
        config=config,
        df=fixture_lmsys_chat_non_english_dataset,
        tokenizer=tokenizer,
        cache_file_path=fixture_session_cache_dir,
    )
