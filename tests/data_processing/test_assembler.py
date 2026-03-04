# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from datasets import Dataset
from transformers import PretrainedConfig, PreTrainedTokenizer

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing.assembler import (
    Example,
    GroupedDataExampleAssembler,
    SequentialExampleAssembler,
    TabularDataExampleAssembler,
    TrainingExampleAssembler,
    _should_flush_example,
)
from nemo_safe_synthesizer.data_processing.record_utils import (
    check_if_records_are_ordered,
    extract_records_from_jsonl_string,
)
from nemo_safe_synthesizer.defaults import PROMPT_TEMPLATE, PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.llm.metadata import DEFAULT_MAX_SEQ_LENGTH, LLMPromptConfig, ModelMetadata

STUB_PROMPT = "Test prompt"
STUB_SEQUENCE = dict(input_ids=[66, 67], attention_mask=[1, 1])


@pytest.fixture(scope="session")
def fixture_assembler_config() -> SafeSynthesizerParameters:
    config = SafeSynthesizerParameters.from_params(use_unsloth=False, rope_scaling_factor=1)
    return config


@pytest.fixture(scope="session")
def fixture_autoconfig() -> PretrainedConfig:
    """Create a PretrainedConfig for testing that passes Pydantic isinstance validation."""
    config = PretrainedConfig()
    config.max_position_embeddings = DEFAULT_MAX_SEQ_LENGTH
    return config


@pytest.fixture(scope="session")
def fixture_llm_metadata(
    fixture_session_cache_dir, fixture_assembler_config: SafeSynthesizerParameters
) -> ModelMetadata:
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_assembler_config.training.pretrained_model, save_path=fixture_session_cache_dir
    )
    assert metadata is not None
    return metadata


def test_example_with_special_tokens_in_prompt(
    fixture_llm_metadata: ModelMetadata, fixture_tokenizer: PreTrainedTokenizer
):
    fixture_llm_metadata.prompt_config.add_bos_token_to_prompt = True
    fixture_llm_metadata.prompt_config.add_eos_token_to_prompt = True
    example = Example(prompt=STUB_PROMPT, tokenizer=fixture_tokenizer, metadata=fixture_llm_metadata)
    example.add_sequence(STUB_SEQUENCE, add_special_tokens=True)
    assert example.num_tokens == 8
    assert example.input_ids == [128011, 2323, 10137, 128012, 128011, 66, 67, 128012]
    assert example.attention_mask == [1] * 8
    assert example.labels == [-100, -100, -100, -100, 128011, 66, 67, 128012]

    example.add_sequence(STUB_SEQUENCE, add_special_tokens=False)
    assert example.num_sequences == 2
    assert example.num_tokens == 10
    assert example.input_ids == [128011, 2323, 10137, 128012, 128011, 66, 67, 128012, 66, 67]
    assert example.attention_mask == [1] * 10
    assert example.labels == [-100, -100, -100, -100, 128011, 66, 67, 128012, 66, 67]
    assert set(example.to_dict().keys()) == {"input_ids", "attention_mask", "labels"}


def test_example_without_special_tokens_in_prompt(
    fixture_llm_metadata: ModelMetadata, fixture_tokenizer: PreTrainedTokenizer
):
    fixture_llm_metadata.prompt_config.add_bos_token_to_prompt = False
    fixture_llm_metadata.prompt_config.add_eos_token_to_prompt = False
    example = Example(prompt=STUB_PROMPT, tokenizer=fixture_tokenizer, metadata=fixture_llm_metadata)

    example.add_sequence(STUB_SEQUENCE, add_special_tokens=True)
    assert example.num_tokens == 6
    assert example.input_ids == [2323, 10137, 128011, 66, 67, 128012]
    assert example.attention_mask == [1] * 6
    assert example.labels == [-100, -100, 128011, 66, 67, 128012]

    example.add_sequence(STUB_SEQUENCE, add_special_tokens=False)
    assert example.num_tokens == 8
    assert example.input_ids == [2323, 10137, 128011, 66, 67, 128012, 66, 67]
    assert example.attention_mask == [1] * 8
    assert example.labels == [-100, -100, 128011, 66, 67, 128012, 66, 67]


def test_add_sequence_raising_exception(fixture_llm_metadata: ModelMetadata, fixture_tokenizer: PreTrainedTokenizer):
    fixture_llm_metadata.base_max_seq_length = 1
    example = Example(prompt=STUB_PROMPT, tokenizer=fixture_tokenizer, metadata=fixture_llm_metadata)

    with pytest.raises(
        GenerationError,
        match="The number of tokens in an example exceeds the available context length.",
    ):
        example.add_sequence(STUB_SEQUENCE)


def test_example_assembler_test_set_size_exception(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_llm_metadata: ModelMetadata,
    fixture_session_cache_dir: Path,
):
    with pytest.raises(
        ParameterError,
        match="The test set size is too large compared to the input dataset.",
    ):
        _ = TabularDataExampleAssembler(
            dataset=fixture_iris_dataset,
            tokenizer=fixture_tokenizer,
            metadata=fixture_llm_metadata,
            test_size=100,
            cache_file_path=fixture_session_cache_dir,
            seed=1,
        )


def test_tabular_data_assembler(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_assembler_config: SafeSynthesizerParameters,
    fixture_session_cache_dir: str,
):
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_assembler_config.training.pretrained_model, save_path=fixture_session_cache_dir
    )
    assembler = TabularDataExampleAssembler(
        dataset=fixture_iris_dataset,
        tokenizer=fixture_tokenizer,
        metadata=metadata,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert assembler.num_records_total == 150
    assert assembler.num_records_train == 150
    assert assembler.num_records_validation == 0

    examples = assembler.assemble_training_examples()
    assert examples.train.num_rows == 1
    assert examples.test is None


def test_tabular_data_assembler_shorter_context_with_test_split(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_llm_metadata: ModelMetadata,
    fixture_session_cache_dir,
):
    fixture_llm_metadata.base_max_seq_length = 512

    assembler = TabularDataExampleAssembler(
        dataset=fixture_iris_dataset,
        tokenizer=fixture_tokenizer,
        metadata=fixture_llm_metadata,
        test_size=0.20,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert assembler.num_records_total == 150
    assert assembler.num_records_train == 120
    assert assembler.num_records_validation == 30

    examples = assembler.assemble_training_examples()
    assert examples.test.num_rows == 3  # depends on tokenizer/model: we fill context with records for the test set
    assert examples.train.num_rows == 11


def test_tabular_data_assembler_dp(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_llm_metadata: ModelMetadata,
    fixture_session_cache_dir,
):
    # Set max_sequences_per_example=1 for DP mode (1 record per example)
    fixture_llm_metadata.max_sequences_per_example = 1
    assembler = TabularDataExampleAssembler(
        dataset=fixture_iris_dataset,
        tokenizer=fixture_tokenizer,
        metadata=fixture_llm_metadata,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    examples = assembler.assemble_training_examples()
    assert examples.stats["records_per_example"].min == 1
    assert examples.stats["records_per_example"].max == 1


def test_assembler_schema_tokenization_exception(
    fixture_pems_sf_sample_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_llm_metadata,
    fixture_session_cache_dir,
    fixture_assembler_config: SafeSynthesizerParameters,
):
    # Use a small context size so this test exercises the max-token limit (default is 12k for non-tinyllama).
    fixture_llm_metadata.base_max_seq_length = 2048
    with pytest.raises(
        GenerationError,
        match="The dataset schema requires more tokens than the max length of the model.",
    ):
        _ = TrainingExampleAssembler.from_data(
            dataset=fixture_pems_sf_sample_dataset,
            tokenizer=fixture_tokenizer,
            metadata=fixture_llm_metadata,
            config=fixture_assembler_config,
            cache_file_path=fixture_session_cache_dir,
            seed=1,
        )


def test_assembler_max_new_token_tokenization_exception(
    fixture_iris_dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_llm_metadata: ModelMetadata,
    fixture_session_cache_dir,
):
    expected_snippet = "At least one record requires more tokens than fit in the available context length."

    with pytest.raises(GenerationError, match=expected_snippet):
        # deliberately reducing max_seq_length to be very small so that records don't fit
        # even though the schema itself fits (schema is 57 tokens for iris)
        fixture_llm_metadata.base_max_seq_length = 60
        _ = TabularDataExampleAssembler(
            dataset=fixture_iris_dataset,
            metadata=fixture_llm_metadata,
            tokenizer=fixture_tokenizer,
            cache_file_path=fixture_session_cache_dir,
            seed=1,
        )


def test_grouped_data_assembler(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_assembler_config: SafeSynthesizerParameters,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
        use_unsloth=True,
        rope_scaling_factor=1,
    )
    llm_metadata = ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        save_path=Path(fixture_session_cache_dir),
    )

    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_chickweight_dataset,
        tokenizer=fixture_tokenizer,
        metadata=llm_metadata,
        config=config,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )

    assert assembler.num_records_total == 578
    assert assembler.num_records_train == 578
    assert assembler.num_records_validation == 0

    examples = assembler.assemble_training_examples()
    assert examples.train.num_rows == 7
    assert examples.test is None
    assert round(examples.stats["tokens_per_record"].mean, 4) == 19.0
    assert round(examples.stats["tokens_per_group"].mean, 4) == 219.64
    assert round(examples.stats["tokens_per_example"].mean, 4) == 1628.1429
    assert round(examples.stats["records_per_example"].mean, 4) == 82.5714
    assert round(examples.stats["groups_per_example"].mean, 4) == 7.1429


def test_grouped_data_assembler_training_examples_low_decimal(
    fixture_sample_patient_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="patient_name",
        order_training_examples_by="timestamp",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
        use_unsloth=True,
        rope_scaling_factor=1,
    )
    llm_metadata = ModelMetadata(
        model_name_or_path=fixture_tokenizer.name_or_path,
        base_max_seq_length=2048,
        autoconfig=fixture_autoconfig,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
    )
    assert llm_metadata is not None
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_sample_patient_dataset,
        tokenizer=fixture_tokenizer,
        metadata=llm_metadata,
        config=config,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert assembler.num_records_total == 200
    assert assembler.num_records_train == 200
    assert assembler.num_records_validation == 0

    examples = assembler.assemble_training_examples(data_fraction=1.01)
    assert examples.train.num_rows == 3
    assert examples.test is None
    assert round(examples.stats["tokens_per_record"].mean, 4) == 18.88
    assert round(examples.stats["tokens_per_group"].mean, 4) == 314.6667
    assert round(examples.stats["tokens_per_example"].mean, 4) == 1431.3333
    assert round(examples.stats["records_per_example"].mean, 4) == 73.3333
    assert round(examples.stats["groups_per_example"].mean, 4) == 4.3333


def test_grouped_data_assembler_training_examples_high_decimal(
    fixture_sample_patient_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="patient_name",
        order_training_examples_by="timestamp",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
        use_unsloth=True,
        rope_scaling_factor=1,
    )
    llm_metadata = ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        save_path=Path(fixture_session_cache_dir),
    )
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_sample_patient_dataset,
        metadata=llm_metadata,
        tokenizer=fixture_tokenizer,
        config=config,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert assembler.num_records_total == 200
    assert assembler.num_records_train == 200
    assert assembler.num_records_validation == 0

    examples = assembler.assemble_training_examples(data_fraction=2.999)
    assert examples.train.num_rows == 7
    assert examples.test is None
    assert round(examples.stats["tokens_per_record"].mean, 4) == 18.88
    assert round(examples.stats["tokens_per_group"].mean, 4) == 314.6667
    assert round(examples.stats["tokens_per_example"].mean, 4) == 1667.5714
    assert round(examples.stats["records_per_example"].mean, 4) == 85.7143
    assert round(examples.stats["groups_per_example"].mean, 4) == 5.1429


def test_grouped_data_assembler_shorter_context_with_test_split(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
        use_unsloth=True,
        rope_scaling_factor=1,
        validation_ratio=0.2,
    )
    llm_metadata = ModelMetadata(
        base_max_seq_length=512,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        save_path=Path(fixture_session_cache_dir),
    )
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_chickweight_dataset,
        tokenizer=fixture_tokenizer,
        metadata=llm_metadata,
        config=config,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assembler = cast(GroupedDataExampleAssembler, assembler)

    assert assembler.num_records_total == 578
    assert assembler.num_records_train == 463
    assert assembler.num_records_validation == 115
    assert assembler.num_groups_train == 40
    assert assembler.num_groups_validation == 10
    assert (
        assembler.num_groups_train + assembler.num_groups_validation
        == fixture_chickweight_dataset.to_pandas()["Chick"].nunique()
    )

    examples = assembler.assemble_training_examples()

    assert examples.train.num_rows == 37
    assert examples.test.num_rows == 9
    assert round(examples.stats["tokens_per_record"].mean, 4) == 19.0
    assert round(examples.stats["tokens_per_group"].mean, 4) == 219.64
    assert round(examples.stats["tokens_per_example"].mean, 4) == 284.9189
    assert round(examples.stats["records_per_example"].mean, 4) == 12.5135
    assert round(examples.stats["groups_per_example"].mean, 4) == 1.0811


def test_grouped_data_assembler_dp(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
        use_unsloth=True,
        rope_scaling_factor=1,
        validation_ratio=0.2,
    )
    llm_metadata = ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        save_path=Path(fixture_session_cache_dir),
        # Set max_sequences_per_example=1 for DP mode (1 group per example)
        max_sequences_per_example=1,
    )
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_chickweight_dataset,
        tokenizer=fixture_tokenizer,
        metadata=llm_metadata,
        config=config,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert isinstance(assembler, GroupedDataExampleAssembler)
    examples = assembler.assemble_training_examples()
    assert examples.stats["groups_per_example"].min == 1
    assert examples.stats["groups_per_example"].max == 1


def test_grouped_data_assembler_context_width_exception(
    fixture_dow_jones_index_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="stock",
        order_training_examples_by="date",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
        use_unsloth=True,
        rope_scaling_factor=1,
    )
    llm_metadata = ModelMetadata(
        # Use a small context so at least one group exceeds it during example generation.
        # Must be large enough for initial tokenization to pass but small enough that
        # the generator hits context limit.
        base_max_seq_length=512,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        save_path=Path(fixture_session_cache_dir),
    )
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_dow_jones_index_dataset,
        tokenizer=fixture_tokenizer,
        metadata=llm_metadata,
        config=config,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    with pytest.raises(
        GenerationError,
        match="The generator provided for dataset generation ran into errors.",
    ):
        _ = assembler.assemble_training_examples()


def test_create_tabular_example_assembler(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_assembler_config: SafeSynthesizerParameters,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    llm_metadata = ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
    )
    assert isinstance(
        TrainingExampleAssembler.from_data(
            dataset=fixture_iris_dataset,
            tokenizer=fixture_tokenizer,
            metadata=llm_metadata,
            config=fixture_assembler_config,
            cache_file_path=fixture_session_cache_dir,
        ),
        TabularDataExampleAssembler,
    )


def test_create_group_example_assembler(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="Chick",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
        use_unsloth=True,
        rope_scaling_factor=1,
    )
    llm_metadata = ModelMetadata(
        model_name_or_path=fixture_tokenizer.name_or_path,
        base_max_seq_length=2048,
        autoconfig=fixture_autoconfig,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        save_path=Path(fixture_session_cache_dir),
    )
    assert isinstance(
        TrainingExampleAssembler.from_data(
            dataset=fixture_chickweight_dataset,
            tokenizer=fixture_tokenizer,
            metadata=llm_metadata,
            config=config,
            cache_file_path=fixture_session_cache_dir,
        ),
        GroupedDataExampleAssembler,
    )


@pytest.fixture
def fixture_sequential_metadata(
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
) -> ModelMetadata:
    """Create ModelMetadata for SequentialExampleAssembler tests."""
    return ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        save_path=Path(fixture_session_cache_dir),
    )


def test_sequential_assembler_raises_for_missing_group_column(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
):
    """Test that SequentialExampleAssembler raises for missing group column."""
    with pytest.raises(ParameterError, match="Group by column.*not found in dataset"):
        SequentialExampleAssembler(
            dataset=fixture_iris_dataset,
            tokenizer=fixture_tokenizer,
            metadata=fixture_sequential_metadata,
            group_training_examples_by="nonexistent_column",
            order_training_examples_by="sepal.length",
            cache_file_path=fixture_session_cache_dir,
            seed=1,
        )


def test_sequential_assembler_raises_for_missing_order_column(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
):
    """Test that SequentialExampleAssembler raises for missing order column."""
    with pytest.raises(ParameterError, match="Order by column.*not found in dataset"):
        SequentialExampleAssembler(
            dataset=fixture_iris_dataset,
            tokenizer=fixture_tokenizer,
            metadata=fixture_sequential_metadata,
            group_training_examples_by="variety",
            order_training_examples_by="nonexistent_column",
            cache_file_path=fixture_session_cache_dir,
            seed=1,
        )


def test_sequential_assembler_reorders_columns(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
):
    """Test that SequentialExampleAssembler puts group and order columns first."""
    assembler = SequentialExampleAssembler(
        dataset=fixture_chickweight_dataset,
        tokenizer=fixture_tokenizer,
        metadata=fixture_sequential_metadata,
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert assembler.schema_prompt.index("Chick") < assembler.schema_prompt.index("Time")


def test_sequential_assembler_excludes_pseudo_group_from_schema(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
):
    """Test that SequentialExampleAssembler excludes PSEUDO_GROUP_COLUMN from schema."""
    df = fixture_iris_dataset.to_pandas()
    df[PSEUDO_GROUP_COLUMN] = 0
    dataset_with_pseudo = Dataset.from_pandas(df)

    assembler = SequentialExampleAssembler(
        dataset=dataset_with_pseudo,
        tokenizer=fixture_tokenizer,
        metadata=fixture_sequential_metadata,
        group_training_examples_by=PSEUDO_GROUP_COLUMN,
        order_training_examples_by="sepal.length",
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert PSEUDO_GROUP_COLUMN not in assembler.schema_prompt


def test_sequential_assembler_sorts_records_by_group_and_order(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
):
    """Test that SequentialExampleAssembler sorts records correctly within groups."""
    assembler = SequentialExampleAssembler(
        dataset=fixture_chickweight_dataset,
        tokenizer=fixture_tokenizer,
        metadata=fixture_sequential_metadata,
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        cache_file_path=fixture_session_cache_dir,
        seed=42,
    )

    train_df = assembler.train_dataset.to_pandas()
    for chick_id, group_df in train_df.groupby("Chick"):
        time_values = group_df["Time"].tolist()
        assert time_values == sorted(time_values), f"Time values not sorted for Chick {chick_id}"


def test_sequential_assembler_token_budget(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
):
    """Test that SequentialExampleAssembler token budget sampling works correctly."""
    import numpy as np

    assembler = SequentialExampleAssembler(
        dataset=fixture_chickweight_dataset,
        tokenizer=fixture_tokenizer,
        metadata=fixture_sequential_metadata,
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        cache_file_path=fixture_session_cache_dir,
        seed=42,
    )

    max_tokens = 1000
    assembler._window_rng = np.random.default_rng(42)

    budget_train = assembler._next_token_budget(max_tokens, is_val=False)
    assert 700 <= budget_train <= 1000  # Training: 0.7-1.0 of max

    budget_val = assembler._next_token_budget(max_tokens, is_val=True)
    assert budget_val == max_tokens  # Validation: always max


def test_sequential_assembler_initial_prefill(
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
):
    """Test that SequentialExampleAssembler returns correct prefill for each group."""

    # Create a small, controlled dataset with 2 groups and known values
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B"],
            "time": [1, 2, 3, 1, 2],
            "value": [10, 20, 30, 100, 200],
        }
    )
    dataset = Dataset.from_pandas(df)

    assembler = SequentialExampleAssembler(
        dataset=dataset,
        tokenizer=fixture_tokenizer,
        metadata=fixture_sequential_metadata,
        group_training_examples_by="group",
        order_training_examples_by="time",
        cache_file_path=fixture_session_cache_dir,
        seed=42,
    )

    prefill = assembler._get_initial_prefill()

    # Should have exactly 2 groups
    assert len(prefill) == 2
    assert "A" in prefill
    assert "B" in prefill

    # Each prefill should contain up to 3 records as JSONL (newline-separated)
    # Group A has 3 records, Group B has 2 records
    # Filter out empty lines that may appear between records
    prefill_a_lines = [line for line in prefill["A"].strip().split("\n") if line]
    prefill_b_lines = [line for line in prefill["B"].strip().split("\n") if line]

    assert len(prefill_a_lines) == 3  # All 3 records from group A
    assert len(prefill_b_lines) == 2  # Both records from group B

    # Verify the records contain expected values (order should be by time)
    assert '"value": 10' in prefill_a_lines[0] or '"value":10' in prefill_a_lines[0]
    assert '"value": 20' in prefill_a_lines[1] or '"value":20' in prefill_a_lines[1]
    assert '"value": 30' in prefill_a_lines[2] or '"value":30' in prefill_a_lines[2]
    assert '"value": 100' in prefill_b_lines[0] or '"value":100' in prefill_b_lines[0]
    assert '"value": 200' in prefill_b_lines[1] or '"value":200' in prefill_b_lines[1]


def test_should_flush_example_boundary_conditions():
    """Test _should_flush_example returns correct values for boundary conditions.

    This is a pure function with no side effects, so multiple test cases
    in one method is appropriate.
    """
    # Group boundary triggers flush
    assert (
        _should_flush_example(
            prev_row_idx=1,
            row_idx=2,
            current_group_value="A",
            record_group="B",
            num_sequences=1,
            max_sequences=10,
            token_total=50,
            record_len=10,
            token_budget=100,
        )
        is True
    )

    # Token budget triggers flush
    assert (
        _should_flush_example(
            prev_row_idx=1,
            row_idx=2,
            current_group_value="A",
            record_group="A",
            num_sequences=1,
            max_sequences=10,
            token_total=95,
            record_len=10,
            token_budget=100,
        )
        is True
    )

    # Max sequences triggers flush
    assert (
        _should_flush_example(
            prev_row_idx=1,
            row_idx=2,
            current_group_value="A",
            record_group="A",
            num_sequences=10,
            max_sequences=10,
            token_total=50,
            record_len=10,
            token_budget=100,
        )
        is True
    )

    # No flush when within limits
    assert (
        _should_flush_example(
            prev_row_idx=1,
            row_idx=2,
            current_group_value="A",
            record_group="A",
            num_sequences=1,
            max_sequences=10,
            token_total=50,
            record_len=10,
            token_budget=100,
        )
        is False
    )

    # Row index restart boundary triggers flush
    assert (
        _should_flush_example(
            prev_row_idx=5,
            row_idx=0,  # Went backwards - dataset restart
            current_group_value="A",
            record_group="A",
            num_sequences=1,
            max_sequences=10,
            token_total=50,
            record_len=10,
            token_budget=100,
        )
        is True
    )


def test_sequential_assembler_end_to_end(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    """End-to-end test: SequentialExampleAssembler creates valid examples."""
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        pretrained_model=fixture_tokenizer.name_or_path,
        num_input_records_to_sample=5000,
        use_unsloth=True,
        rope_scaling_factor=1,
    )
    config.time_series.is_timeseries = True
    max_seq_length = 256
    llm_metadata = ModelMetadata(
        base_max_seq_length=max_seq_length,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        save_path=Path(fixture_session_cache_dir),
    )

    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_chickweight_dataset,
        tokenizer=fixture_tokenizer,
        metadata=llm_metadata,
        config=config,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )

    assert isinstance(assembler, SequentialExampleAssembler)
    assert assembler.num_records_total == 578

    examples = assembler.assemble_training_examples()
    assert examples.train.num_rows > 0

    # All examples should respect max sequence length
    for i in range(examples.train.num_rows):
        num_tokens = len(examples.train[i]["input_ids"])
        assert num_tokens <= max_seq_length

    # Verify each example has records from only one group and Time values are ordered
    for i in range(examples.train.num_rows):
        input_ids = examples.train[i]["input_ids"]
        text = fixture_tokenizer.decode(input_ids, skip_special_tokens=True)
        record_strings = extract_records_from_jsonl_string(text)
        records = [json.loads(r) for r in record_strings]

        if len(records) > 0:
            # All records in an example should have the same Chick (group) value
            chick_values = [r.get("Chick") for r in records if "Chick" in r]
            if chick_values:
                assert len(set(chick_values)) == 1, f"Example {i} has records from multiple groups: {set(chick_values)}"

            # Time values should be in ascending order within each example
            records_with_time = [r for r in records if "Time" in r]
            if len(records_with_time) > 1:
                assert check_if_records_are_ordered(records_with_time, "Time"), (
                    f"Example {i} has Time values out of order"
                )


def test_sequential_assembler_single_group_with_pseudo_column(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_autoconfig: PretrainedConfig,
):
    """Test SequentialExampleAssembler with a single group using pseudo column."""

    # Add pseudo group column to simulate ungrouped time series
    # Adding pseudo group is already tested in test_timeseries_preprocessing.py
    df = fixture_iris_dataset.to_pandas()
    df[PSEUDO_GROUP_COLUMN] = 0  # All records in one group
    df["timestamp"] = range(len(df))  # Add a synthetic timestamp column
    dataset_with_pseudo = Dataset.from_pandas(df)

    max_seq_length = 512
    llm_metadata = ModelMetadata(
        base_max_seq_length=max_seq_length,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<s>",
            bos_token_id=1,
            eos_token="</s>",
            eos_token_id=2,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
        save_path=Path(fixture_session_cache_dir),
    )

    assembler = SequentialExampleAssembler(
        dataset=dataset_with_pseudo,
        tokenizer=fixture_tokenizer,
        metadata=llm_metadata,
        group_training_examples_by=PSEUDO_GROUP_COLUMN,
        order_training_examples_by="timestamp",
        cache_file_path=fixture_session_cache_dir,
        seed=42,
    )

    # Verify pseudo group column is excluded from schema prompt
    assert PSEUDO_GROUP_COLUMN not in assembler.schema_prompt

    # Verify assembler processes all records
    assert assembler.num_records_total == len(fixture_iris_dataset)

    examples = assembler.assemble_training_examples()
    assert examples.train.num_rows > 0

    # All examples should respect max sequence length
    for i in range(examples.train.num_rows):
        num_tokens = len(examples.train[i]["input_ids"])
        assert num_tokens <= max_seq_length

    # Verify timestamp ordering is maintained within each example
    for i in range(examples.train.num_rows):
        input_ids = examples.train[i]["input_ids"]
        text = fixture_tokenizer.decode(input_ids, skip_special_tokens=True)
        record_strings = extract_records_from_jsonl_string(text)
        records = [json.loads(r) for r in record_strings]

        if len(records) > 1:
            # Timestamp values should be in ascending order
            records_with_timestamp = [r for r in records if "timestamp" in r]
            if len(records_with_timestamp) > 1:
                assert check_if_records_are_ordered(records_with_timestamp, "timestamp"), (
                    f"Example {i} has timestamps out of order"
                )

            # Pseudo group column should not appear in the records (excluded from JSONL)
            for record in records:
                assert PSEUDO_GROUP_COLUMN not in record, f"Pseudo group column found in record: {record}"
