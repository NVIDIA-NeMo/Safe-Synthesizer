# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import cast

import pytest
from datasets import Dataset
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing.assembler import (
    Example,
    GroupedDataExampleAssembler,
    TabularDataExampleAssembler,
    TrainingExampleAssembler,
)
from nemo_safe_synthesizer.defaults import PROMPT_TEMPLATE
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.llm.metadata import DEFAULT_MAX_SEQ_LENGTH, LLMPromptConfig, ModelMetadata
from transformers import PretrainedConfig, PreTrainedTokenizer

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
    assert example.input_ids == [1, 4321, 9508, 2, 1, 66, 67, 2]
    assert example.attention_mask == [1] * 8
    assert example.labels == [-100, -100, -100, -100, 1, 66, 67, 2]

    example.add_sequence(STUB_SEQUENCE, add_special_tokens=False)
    assert example.num_sequences == 2
    assert example.num_tokens == 10
    assert example.input_ids == [1, 4321, 9508, 2, 1, 66, 67, 2, 66, 67]
    assert example.attention_mask == [1] * 10
    assert example.labels == [-100, -100, -100, -100, 1, 66, 67, 2, 66, 67]
    assert set(example.to_dict().keys()) == {"input_ids", "attention_mask", "labels"}


def test_example_without_special_tokens_in_prompt(
    fixture_llm_metadata: ModelMetadata, fixture_tokenizer: PreTrainedTokenizer
):
    fixture_llm_metadata.prompt_config.add_bos_token_to_prompt = False
    fixture_llm_metadata.prompt_config.add_eos_token_to_prompt = False
    example = Example(prompt=STUB_PROMPT, tokenizer=fixture_tokenizer, metadata=fixture_llm_metadata)

    example.add_sequence(STUB_SEQUENCE, add_special_tokens=True)
    assert example.num_tokens == 6
    assert example.input_ids == [4321, 9508, 1, 66, 67, 2]
    assert example.attention_mask == [1] * 6
    assert example.labels == [-100, -100, 1, 66, 67, 2]

    example.add_sequence(STUB_SEQUENCE, add_special_tokens=False)
    assert example.num_tokens == 8
    assert example.input_ids == [4321, 9508, 1, 66, 67, 2, 66, 67]
    assert example.attention_mask == [1] * 8
    assert example.labels == [-100, -100, 1, 66, 67, 2, 66, 67]


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
    assert examples.train.num_rows == 4
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
    assert (
        examples.test.num_rows == 4
    )  # changed from 30 to 4 because we are filling context with records for the test set as well
    assert examples.train.num_rows == 13


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
    assert round(examples.stats["tokens_per_record"].mean, 4) == 22.9135
    assert round(examples.stats["tokens_per_group"].mean, 4) == 264.88
    assert round(examples.stats["tokens_per_example"].mean, 4) == 1949.2857
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
    assert examples.train.num_rows == 4
    assert examples.test is None
    assert round(examples.stats["tokens_per_record"].mean, 4) == 27.755
    assert round(examples.stats["tokens_per_group"].mean, 4) == 462.5833
    assert round(examples.stats["tokens_per_example"].mean, 4) == 1571.0
    assert round(examples.stats["records_per_example"].mean, 4) == 55.0
    assert round(examples.stats["groups_per_example"].mean, 4) == 3.25


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
    assert examples.train.num_rows == 10
    assert examples.test is None
    assert round(examples.stats["tokens_per_record"].mean, 4) == 27.755
    assert round(examples.stats["tokens_per_group"].mean, 4) == 462.5833
    assert round(examples.stats["tokens_per_example"].mean, 4) == 1714.5
    assert round(examples.stats["records_per_example"].mean, 4) == 60.0
    assert round(examples.stats["groups_per_example"].mean, 4) == 3.6


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

    assert examples.train.num_rows == 39
    assert examples.test.num_rows == 10
    assert round(examples.stats["tokens_per_record"].mean, 4) == 22.9135
    assert round(examples.stats["tokens_per_group"].mean, 4) == 264.88
    assert round(examples.stats["tokens_per_example"].mean, 4) == 316.8462
    assert round(examples.stats["records_per_example"].mean, 4) == 11.8718
    assert round(examples.stats["groups_per_example"].mean, 4) == 1.0256


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
