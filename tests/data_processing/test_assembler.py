# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import cast

import numpy as np
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
from nemo_safe_synthesizer.tokenization import WorkloadKind, bind_tokenizer
from nemo_safe_synthesizer.tokenization.core import _BoundTokenization

STUB_PROMPT = "Test prompt"
STUB_SEQUENCE = dict(input_ids=[66, 67], attention_mask=[1, 1])


def _record_tokenizer(native: PreTrainedTokenizer, metadata: ModelMetadata, *, time_series: bool = False):
    workload = WorkloadKind.TIME_SERIES if time_series else WorkloadKind.TABULAR
    return bind_tokenizer(native, metadata, workload_kind=workload)


def _example(native: PreTrainedTokenizer, metadata: ModelMetadata) -> Example:
    tokenizer = _record_tokenizer(native, metadata)
    return Example(prompt=tokenizer.encode_prompt_text(STUB_PROMPT), tokenization=tokenizer, metadata=metadata)


# Purpose: Session-scoped assembler config pointing at a local SmolLM3 tokenizer directory
# to avoid HuggingFace Hub downloads during tests.
@pytest.fixture(scope="session")
def fixture_assembler_config(fixture_smollm3_tokenizer: str) -> SafeSynthesizerParameters:
    config = SafeSynthesizerParameters.from_params(rope_scaling_factor=1, pretrained_model=fixture_smollm3_tokenizer)
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
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=fixture_assembler_config.training.pretrained_model)
    assert metadata is not None
    return metadata


def test_example_with_special_tokens_in_prompt(
    fixture_llm_metadata: ModelMetadata, fixture_tokenizer: PreTrainedTokenizer
):
    fixture_llm_metadata.prompt_config.add_bos_token_to_prompt = True
    fixture_llm_metadata.prompt_config.add_eos_token_to_prompt = True
    example = _example(fixture_tokenizer, fixture_llm_metadata)
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
    example = _example(fixture_tokenizer, fixture_llm_metadata)

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


def test_example_preserves_nontrivial_sequence_attention_mask(
    fixture_llm_metadata: ModelMetadata, fixture_tokenizer: PreTrainedTokenizer
) -> None:
    fixture_llm_metadata.prompt_config.add_bos_token_to_prompt = False
    fixture_llm_metadata.prompt_config.add_eos_token_to_prompt = False
    example = _example(fixture_tokenizer, fixture_llm_metadata)

    example.add_sequence({"input_ids": [66, 67], "attention_mask": [0, 1]})

    assert example.attention_mask == [1, 1, 1, 0, 1, 1]


@pytest.mark.parametrize("attention_mask", [[1], [1, 2]])
def test_example_rejects_malformed_attention_mask_without_mutation(
    fixture_llm_metadata: ModelMetadata,
    fixture_tokenizer: PreTrainedTokenizer,
    attention_mask: list[int],
) -> None:
    example = _example(fixture_tokenizer, fixture_llm_metadata)
    num_sequences = example.num_sequences
    input_ids = list(example.input_ids)
    original_attention_mask = list(example.attention_mask)
    labels = list(example.labels)

    with pytest.raises(ParameterError) as error:
        example.add_sequence({"input_ids": [66, 67], "attention_mask": attention_mask})

    assert str(error.value) == "Each sequence attention mask must match its IDs and contain only zero or one."
    assert example.num_sequences == num_sequences
    assert example.input_ids == input_ids
    assert example.attention_mask == original_attention_mask
    assert example.labels == labels


def test_example_validates_malformed_attention_mask_before_sequence_count(
    fixture_llm_metadata: ModelMetadata,
    fixture_tokenizer: PreTrainedTokenizer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fixture_llm_metadata, "max_sequences_per_example", 1)
    example = _example(fixture_tokenizer, fixture_llm_metadata)
    example.add_sequence(STUB_SEQUENCE)
    num_sequences = example.num_sequences
    input_ids = list(example.input_ids)
    attention_mask = list(example.attention_mask)
    labels = list(example.labels)

    with pytest.raises(ParameterError) as error:
        example.add_sequence({"input_ids": [66, 67], "attention_mask": [1]})

    assert str(error.value) == "Each sequence attention mask must match its IDs and contain only zero or one."
    assert example.num_sequences == num_sequences
    assert example.input_ids == input_ids
    assert example.attention_mask == attention_mask
    assert example.labels == labels


def test_add_sequence_raising_exception(fixture_llm_metadata: ModelMetadata, fixture_tokenizer: PreTrainedTokenizer):
    fixture_llm_metadata.base_max_seq_length = 1
    example = _example(fixture_tokenizer, fixture_llm_metadata)

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
            tokenization=_record_tokenizer(fixture_tokenizer, fixture_llm_metadata),
            metadata=fixture_llm_metadata,
            test_size=100,
            cache_file_path=fixture_session_cache_dir,
            seed=1,
        )


def test_tabular_data_assembler(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_assembler_config: SafeSynthesizerParameters,
    tmp_path: Path,
):
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=fixture_assembler_config.training.pretrained_model)
    assembler = TabularDataExampleAssembler(
        dataset=fixture_iris_dataset,
        tokenization=_record_tokenizer(fixture_tokenizer, metadata),
        metadata=metadata,
        cache_file_path=tmp_path,
        seed=1,
    )
    assert assembler.num_records_total == 150
    assert assembler.num_records_train == 150
    assert assembler.num_records_validation == 0
    assert assembler.tokenization.native is fixture_tokenizer

    examples = assembler.assemble_training_examples()
    assert examples.train.num_rows == 1
    assert examples.test is None


def test_tabular_token_cache_hit_avoids_reencoding_and_replays_capacity_guard(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_assembler_config: SafeSynthesizerParameters,
    tmp_path: Path,
    monkeypatch,
):
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=fixture_assembler_config.training.pretrained_model)
    original_encode = _BoundTokenization.encode_records
    encode_calls = 0

    def recording_encode(self, records, *, exclude_columns=()):
        nonlocal encode_calls
        encode_calls += 1
        return original_encode(self, records, exclude_columns=exclude_columns)

    monkeypatch.setattr(_BoundTokenization, "encode_records", recording_encode)
    first = TrainingExampleAssembler.from_data(
        dataset=fixture_iris_dataset,
        tokenizer=fixture_tokenizer,
        metadata=metadata,
        config=fixture_assembler_config,
        cache_file_path=tmp_path,
        seed=1,
    )
    assert encode_calls > 0

    encode_calls = 0
    original_guard = TrainingExampleAssembler._validate_record_capacity
    guard_calls = 0

    def recording_guard(self, input_ids):
        nonlocal guard_calls
        guard_calls += 1
        return original_guard(self, input_ids)

    monkeypatch.setattr(TrainingExampleAssembler, "_validate_record_capacity", recording_guard)
    second = TrainingExampleAssembler.from_data(
        dataset=fixture_iris_dataset,
        tokenizer=fixture_tokenizer,
        metadata=metadata,
        config=fixture_assembler_config,
        cache_file_path=tmp_path,
        seed=1,
    )

    assert encode_calls == 0
    assert guard_calls == 1
    assert second.tokenized_records.to_dict() == first.tokenized_records.to_dict()
    assert second.stats["tokens_per_record"].mean == first.stats["tokens_per_record"].mean


def test_record_mapping_is_invariant_to_dataset_batch_boundaries(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_assembler_config: SafeSynthesizerParameters,
    tmp_path: Path,
) -> None:
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=fixture_assembler_config.training.pretrained_model)
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_iris_dataset,
        tokenizer=fixture_tokenizer,
        metadata=metadata,
        config=fixture_assembler_config,
        cache_file_path=tmp_path,
        seed=1,
    )
    source = fixture_iris_dataset.select(range(7))

    single = source.map(
        assembler._tokenize_records,
        batched=True,
        batch_size=1,
        remove_columns=source.column_names,
        load_from_cache_file=False,
        new_fingerprint="single-record-batches",
    )
    multi = source.map(
        assembler._tokenize_records,
        batched=True,
        batch_size=4,
        remove_columns=source.column_names,
        load_from_cache_file=False,
        new_fingerprint="multi-record-batches",
    )

    assert single.to_dict() == multi.to_dict()
    assert single.to_dict() == assembler.tokenized_records.select(range(7)).to_dict()


def test_dataset_content_fingerprint_changes_production_cache_key(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_assembler_config: SafeSynthesizerParameters,
    tmp_path: Path,
) -> None:
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=fixture_assembler_config.training.pretrained_model)
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_iris_dataset,
        tokenizer=fixture_tokenizer,
        metadata=metadata,
        config=fixture_assembler_config,
        cache_file_path=tmp_path,
        seed=1,
    )
    changed = fixture_iris_dataset.map(
        lambda row: {**row, "sepal.length": row["sepal.length"] + 1},
        load_from_cache_file=False,
    )

    original_key = assembler._token_cache_key(fixture_iris_dataset, ())
    changed_key = assembler._token_cache_key(changed, ())

    assert original_key.dataset_fingerprint != changed_key.dataset_fingerprint
    assert original_key.digest != changed_key.digest


def test_grouped_and_sequential_share_the_native_record_transform(
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_assembler_config: SafeSynthesizerParameters,
    tmp_path: Path,
) -> None:
    dataset = Dataset.from_dict(
        {
            "group": ["a", "a", "b", "b"],
            "order": [1, 2, 1, 2],
            "value": [10, 11, 20, 21],
        }
    )
    grouped_metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_assembler_config.training.pretrained_model
    )
    sequential_metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_assembler_config.training.pretrained_model
    )

    grouped = GroupedDataExampleAssembler(
        group_training_examples_by="group",
        order_training_examples_by="order",
        dataset=dataset,
        tokenization=_record_tokenizer(fixture_tokenizer, grouped_metadata),
        metadata=grouped_metadata,
        cache_file_path=tmp_path,
        seed=1,
    )
    sequential = SequentialExampleAssembler(
        group_training_examples_by="group",
        order_training_examples_by="order",
        dataset=dataset,
        tokenization=_record_tokenizer(fixture_tokenizer, sequential_metadata, time_series=True),
        metadata=sequential_metadata,
        cache_file_path=tmp_path,
        seed=1,
    )

    cache_files = tuple((tmp_path / "nss-record-tokens" / "v2").glob("*.arrow"))
    assert cache_files
    assert grouped.tokenized_records.column_names == ["group", "order", "text", "input_ids", "attention_mask"]
    assert sequential.tokenized_records.column_names == ["group", "order", "text", "input_ids", "attention_mask"]
    expected_text = [
        '{"group":"a","order":1,"value":10}\n',
        '{"group":"a","order":2,"value":11}\n',
        '{"group":"b","order":1,"value":20}\n',
        '{"group":"b","order":2,"value":21}\n',
    ]
    expected_ids = fixture_tokenizer(expected_text, add_special_tokens=False)["input_ids"]
    expected_masks = [[1] * len(row) for row in expected_ids]
    for assembler in (grouped, sequential):
        assert assembler.tokenized_records.to_dict() == {
            "group": ["a", "a", "b", "b"],
            "order": [1, 2, 1, 2],
            "text": expected_text,
            "input_ids": expected_ids,
            "attention_mask": expected_masks,
        }
        assert assembler.stats["tokens_per_record"].mean == sum(map(len, expected_ids)) / len(expected_ids)


def test_tabular_data_assembler_shorter_context_with_test_split(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_llm_metadata: ModelMetadata,
    fixture_session_cache_dir,
):
    fixture_llm_metadata.base_max_seq_length = 512

    assembler = TabularDataExampleAssembler(
        dataset=fixture_iris_dataset,
        tokenization=_record_tokenizer(fixture_tokenizer, fixture_llm_metadata),
        metadata=fixture_llm_metadata,
        test_size=0.20,
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert assembler.num_records_total == 150
    assert assembler.num_records_train == 120
    assert assembler.num_records_validation == 30

    examples = assembler.assemble_training_examples()
    assert examples.test is not None
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
        tokenization=_record_tokenizer(fixture_tokenizer, fixture_llm_metadata),
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
            tokenization=_record_tokenizer(fixture_tokenizer, fixture_llm_metadata),
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
        rope_scaling_factor=1,
    )
    llm_metadata = ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
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
    assert examples.train.num_rows == 6
    assert all(len(input_ids) <= llm_metadata.max_seq_length for input_ids in examples.train["input_ids"])
    assert examples.test is None
    assert round(examples.stats["tokens_per_record"].mean, 4) == 19.0
    assert round(examples.stats["tokens_per_group"].mean, 4) == 219.64
    assert round(examples.stats["tokens_per_example"].mean, 4) == 1892.0
    assert round(examples.stats["records_per_example"].mean, 4) == 96.3333
    assert round(examples.stats["groups_per_example"].mean, 4) == 8.3333


def test_grouped_data_assembler_training_examples_low_decimal(
    fixture_sample_patient_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    tmp_path: Path,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="patient_name",
        order_training_examples_by="timestamp",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
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
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
    )
    assert llm_metadata is not None
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_sample_patient_dataset,
        tokenizer=fixture_tokenizer,
        metadata=llm_metadata,
        config=config,
        cache_file_path=tmp_path,
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


def test_grouped_data_assembler_training_examples_high_decimal_with_warm_shuffle_cache(
    fixture_sample_patient_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    tmp_path: Path,
    fixture_autoconfig: PretrainedConfig,
):
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="patient_name",
        order_training_examples_by="timestamp",
        pretrained_model=fixture_tokenizer.name_or_path,
        # Provide specific values for auto params as auto param resolution
        # only happens in the skynet or jarvis implementations.
        num_input_records_to_sample=5000,
        rope_scaling_factor=1,
    )
    llm_metadata = ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
    )
    assembler = TrainingExampleAssembler.from_data(
        dataset=fixture_sample_patient_dataset,
        metadata=llm_metadata,
        tokenizer=fixture_tokenizer,
        config=config,
        cache_file_path=tmp_path,
        seed=1,
    )
    # Warm the datasets-owned shuffle cache explicitly. A cache hit returns
    # before consuming the generator, so all three passes reuse one group order.
    _ = assembler.training_dataset.shuffle(generator=np.random.default_rng(assembler.seed))
    assert assembler.num_records_total == 200
    assert assembler.num_records_train == 200
    assert assembler.num_records_validation == 0

    examples = assembler.assemble_training_examples(data_fraction=2.999)
    assert examples.train.num_rows == 6
    assert examples.test is None
    assert round(examples.stats["tokens_per_record"].mean, 4) == 18.88
    assert round(examples.stats["tokens_per_group"].mean, 4) == 314.6667
    assert round(examples.stats["tokens_per_example"].mean, 4) == 1939.0
    assert round(examples.stats["records_per_example"].mean, 4) == 100.0
    assert round(examples.stats["groups_per_example"].mean, 4) == 6.0


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
        rope_scaling_factor=1,
        validation_ratio=0.2,
    )
    llm_metadata = ModelMetadata(
        base_max_seq_length=512,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
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
        == cast(pd.DataFrame, fixture_chickweight_dataset.to_pandas())["Chick"].nunique()
    )

    examples = assembler.assemble_training_examples()

    assert examples.train.num_rows == 20
    assert examples.test is not None
    assert examples.test.num_rows == 5
    assert all(len(input_ids) <= llm_metadata.max_seq_length for input_ids in examples.train["input_ids"])
    assert all(len(input_ids) <= llm_metadata.max_seq_length for input_ids in examples.test["input_ids"])
    assert round(examples.stats["tokens_per_record"].mean, 4) == 19.0
    assert round(examples.stats["tokens_per_group"].mean, 4) == 219.925
    assert round(examples.stats["tokens_per_example"].mean, 4) == 488.85
    assert round(examples.stats["records_per_example"].mean, 4) == 23.15
    assert round(examples.stats["groups_per_example"].mean, 4) == 2.0
    # Holdout groups must not inflate the training-derived generation bound.
    assert examples.stats["records_per_group"].count == assembler.num_groups_train
    assert assembler.stats_val["records_per_group"].count == assembler.num_groups_validation


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
        rope_scaling_factor=1,
        validation_ratio=0.2,
    )
    llm_metadata = ModelMetadata(
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template=PROMPT_TEMPLATE,
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
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
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
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
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
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
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
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
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
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
        tokenization=_record_tokenizer(fixture_tokenizer, fixture_sequential_metadata, time_series=True),
        metadata=fixture_sequential_metadata,
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        cache_file_path=fixture_session_cache_dir,
        seed=1,
    )
    assert assembler.schema_prompt.index("Chick") < assembler.schema_prompt.index("Time")


@pytest.mark.parametrize(
    ("group_by", "order_by", "missing_role"),
    [
        ("nonexistent_group", "Time", "Group by"),
        ("Chick", "nonexistent_order", "Order by"),
    ],
)
def test_sequential_assembler_raises_parameter_error_for_missing_required_columns(
    fixture_chickweight_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
    group_by: str,
    order_by: str,
    missing_role: str,
):
    """Direct constructor callers should get actionable user-facing errors."""
    with pytest.raises(ParameterError, match=f"{missing_role} column 'nonexistent_"):
        SequentialExampleAssembler(
            dataset=fixture_chickweight_dataset,
            tokenization=_record_tokenizer(fixture_tokenizer, fixture_sequential_metadata, time_series=True),
            metadata=fixture_sequential_metadata,
            group_training_examples_by=group_by,
            order_training_examples_by=order_by,
            cache_file_path=fixture_session_cache_dir,
            seed=1,
        )


def test_sequential_assembler_excludes_pseudo_group_from_schema(
    fixture_iris_dataset: Dataset,
    fixture_tokenizer: PreTrainedTokenizer,
    fixture_session_cache_dir: str,
    fixture_sequential_metadata: ModelMetadata,
):
    """Test that SequentialExampleAssembler excludes PSEUDO_GROUP_COLUMN from schema."""
    df = cast(pd.DataFrame, fixture_iris_dataset.to_pandas())
    df[PSEUDO_GROUP_COLUMN] = 0
    dataset_with_pseudo = Dataset.from_pandas(df)

    assembler = SequentialExampleAssembler(
        dataset=dataset_with_pseudo,
        tokenization=_record_tokenizer(fixture_tokenizer, fixture_sequential_metadata, time_series=True),
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
        tokenization=_record_tokenizer(fixture_tokenizer, fixture_sequential_metadata, time_series=True),
        metadata=fixture_sequential_metadata,
        group_training_examples_by="Chick",
        order_training_examples_by="Time",
        cache_file_path=fixture_session_cache_dir,
        seed=42,
    )

    assert assembler.training_dataset is not None
    training_df = cast(pd.DataFrame, assembler.training_dataset.to_pandas())
    for chick_id, group_df in training_df.groupby("Chick"):
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
        tokenization=_record_tokenizer(fixture_tokenizer, fixture_sequential_metadata, time_series=True),
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
        tokenization=_record_tokenizer(fixture_tokenizer, fixture_sequential_metadata, time_series=True),
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

    # Pin the exact byte shape: a leading space, then newline-terminated
    # training-dialect (pandas to_json) records with single newlines between
    # them -- the same shape training examples use. Blank lines or Python
    # json.dumps spacing here would put the generation prompt in a dialect
    # the model never saw in training.
    assert prefill["A"] == (
        ' {"group":"A","time":1,"value":10}\n{"group":"A","time":2,"value":20}\n{"group":"A","time":3,"value":30}\n'
    )
    assert prefill["B"] == ' {"group":"B","time":1,"value":100}\n{"group":"B","time":2,"value":200}\n'


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
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
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
        assert isinstance(text, str)
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
    df = cast(pd.DataFrame, fixture_iris_dataset.to_pandas())
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
            bos_token="<|im_start|>",
            bos_token_id=128011,
            eos_token="<|im_end|>",
            eos_token_id=128012,
        ),
        model_name_or_path=fixture_tokenizer.name_or_path,
        autoconfig=fixture_autoconfig,
    )

    assembler = SequentialExampleAssembler(
        dataset=dataset_with_pseudo,
        tokenization=_record_tokenizer(fixture_tokenizer, llm_metadata, time_series=True),
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
        assert isinstance(text, str)
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
