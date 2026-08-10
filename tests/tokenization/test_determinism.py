# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-stage tokenizer identity and deterministic rendering contracts."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing import record_utils
from nemo_safe_synthesizer.defaults import DEFAULT_INSTRUCTION, PROMPT_TEMPLATE
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.llm.metadata import Mistral, ModelMetadata, SmolLM3, TinyLlama
from nemo_safe_synthesizer.tokenization import PromptEncoding, WorkloadKind, bind_tokenizer
from nemo_safe_synthesizer.tokenization.core import TrainingEncoding
from nemo_safe_synthesizer.training.huggingface_backend import HuggingFaceBackend


@dataclass(frozen=True)
class _TokenizerFixture:
    name: str
    metadata_type: Callable[..., ModelMetadata]
    template: str
    bos_token_id: int
    eos_token_id: int
    add_bos: bool
    add_eos: bool


@dataclass(frozen=True)
class _PipelineCase:
    name: str
    workload: WorkloadKind
    group_by: str | None
    order_by: str | None
    expected_columns: tuple[str, ...]
    records: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _PromptCase:
    name: str
    pipeline: _PipelineCase
    prefill: str


_INPUT_COLUMNS = ("value", "group", "time", "name")
_TIME_SERIES_COLUMNS = ("group", "time", "value", "name")
_REPRESENTATIVE_RECORD: dict[str, object] = {"value": -1.5, "group": "A", "time": 0, "name": "é/☃"}
_SECOND_RECORD: dict[str, object] = {"value": 2.0, "group": "A", "time": 60, "name": "line\nbreak"}
_TOKENIZER_FIXTURES = (
    _TokenizerFixture("tinyllama", TinyLlama, PROMPT_TEMPLATE, 1, 2, True, True),
    _TokenizerFixture(
        "mistral7b",
        Mistral,
        "[INST] {instruction} \n\n {schema} [/INST]{prefill}",
        1,
        2,
        True,
        True,
    ),
    _TokenizerFixture(
        "smollm3b",
        SmolLM3,
        "user\n {instruction} {schema} <|im_end|> \n assistant\n{prefill}",
        128011,
        128012,
        True,
        False,
    ),
)
_PIPELINE_CASES = (
    _PipelineCase("tabular", WorkloadKind.TABULAR, None, None, _INPUT_COLUMNS, (_REPRESENTATIVE_RECORD,)),
    _PipelineCase(
        "grouped",
        WorkloadKind.TABULAR,
        "group",
        "time",
        _INPUT_COLUMNS,
        (_SECOND_RECORD, _REPRESENTATIVE_RECORD),
    ),
    _PipelineCase(
        "time-series",
        WorkloadKind.TIME_SERIES,
        "group",
        "time",
        _TIME_SERIES_COLUMNS,
        (_SECOND_RECORD, _REPRESENTATIVE_RECORD),
    ),
)
_TABULAR, _GROUPED, _TIME_SERIES = _PIPELINE_CASES
_SPACE_LED_PREFILL = ' {"group":"A","time":0,"value":-1.5,"name":"é\\/☃"}\n'
_SPACELESS_PREFILL = _SPACE_LED_PREFILL[1:]
_RECORD_TEXTS: dict[tuple[tuple[str, ...], int], str] = {
    (_INPUT_COLUMNS, 0): '{"value":-1.5,"group":"A","time":0,"name":"é\\/☃"}\n',
    (_INPUT_COLUMNS, 60): '{"value":2.0,"group":"A","time":60,"name":"line\\nbreak"}\n',
    (_TIME_SERIES_COLUMNS, 0): '{"group":"A","time":0,"value":-1.5,"name":"é\\/☃"}\n',
    (_TIME_SERIES_COLUMNS, 60): '{"group":"A","time":60,"value":2.0,"name":"line\\nbreak"}\n',
}
_PROMPT_CASES = (
    _PromptCase("tabular-empty", _TABULAR, ""),
    _PromptCase("grouped-empty", _GROUPED, ""),
    _PromptCase("time-series-empty", _TIME_SERIES, ""),
    _PromptCase("time-series-space-led", _TIME_SERIES, _SPACE_LED_PREFILL),
    _PromptCase("time-series-space-less", _TIME_SERIES, _SPACELESS_PREFILL),
)


def _workdir(path: Path) -> Workdir:
    workdir = Workdir(
        base_path=path,
        dataset_name="dataset",
        config_name="config",
        run_name="run",
        _current_phase="train",
    )
    workdir.ensure_directories()
    return workdir


def _autoconfig() -> PretrainedConfig:
    config = PretrainedConfig()
    config.max_position_embeddings = 2048
    return config


def _metadata(
    fixture: _TokenizerFixture,
    native: PreTrainedTokenizerBase,
    workdir: Workdir,
) -> ModelMetadata:
    metadata = fixture.metadata_type(
        model_name_or_path=str(native.name_or_path),
        tokenizer=native,
        workdir=workdir,
        max_sequences_per_example=10,
    )
    assert metadata.instruction == DEFAULT_INSTRUCTION
    assert metadata.prompt_config.template == fixture.template
    assert metadata.prompt_config.bos_token_id == fixture.bos_token_id
    assert metadata.prompt_config.eos_token_id == fixture.eos_token_id
    assert metadata.prompt_config.add_bos_token_to_prompt is fixture.add_bos
    assert metadata.prompt_config.add_eos_token_to_prompt is fixture.add_eos
    return metadata


def _parameters(source: Path, case: _PipelineCase) -> SafeSynthesizerParameters:
    values: dict[str, object] = {
        "pretrained_model": str(source),
        "num_input_records_to_sample": len(case.records),
        "validation_ratio": 0.0,
        "rope_scaling_factor": 1,
        "group_training_examples_by": case.group_by,
        "order_training_examples_by": case.order_by,
    }
    if case.workload is WorkloadKind.TIME_SERIES:
        values.update(
            is_timeseries=True,
            timestamp_column="time",
            timestamp_format="elapsed_seconds",
        )
    return SafeSynthesizerParameters.from_params(**values)


def _training_backend(
    params: SafeSynthesizerParameters,
    metadata: ModelMetadata,
    workdir: Workdir,
    dataset: Dataset,
) -> HuggingFaceBackend:
    with patch(
        "nemo_safe_synthesizer.training.huggingface_backend.AutoConfig.from_pretrained",
        return_value=_autoconfig(),
    ):
        backend = HuggingFaceBackend(params=params, model_metadata=metadata, training_dataset=dataset, workdir=workdir)
    backend.model_loader_type = MagicMock()
    with patch.object(backend, "_normalize_rope_parameters"):
        backend._load_pretrained_model()
    return backend


def _expected_prompt(
    native: PreTrainedTokenizerBase,
    fixture: _TokenizerFixture,
    columns: tuple[str, ...],
    prefill: str,
) -> PromptEncoding:
    schema = ",".join(f'"{column}":<unk>' for column in columns)
    text = fixture.template.format(instruction=DEFAULT_INSTRUCTION, schema=schema, prefill=prefill)
    body = tuple(native.encode(text, add_special_tokens=False))
    input_ids = (
        *((fixture.bos_token_id,) if fixture.add_bos else ()),
        *body,
        *((fixture.eos_token_id,) if fixture.add_eos else ()),
    )
    return PromptEncoding(text, input_ids, (1,) * len(input_ids))


def _record_text(record: dict[str, object], columns: tuple[str, ...]) -> str:
    time = record["time"]
    assert isinstance(time, int)
    return _RECORD_TEXTS[(columns, time)]


def _expected_record_ids(
    native: PreTrainedTokenizerBase,
    case: _PipelineCase,
) -> tuple[tuple[int, ...], ...]:
    ordered = sorted(case.records, key=lambda record: record["time"]) if case.order_by else case.records
    return tuple(
        tuple(native.encode(_record_text(record, case.expected_columns), add_special_tokens=False))
        for record in ordered
    )


def _expected_training(
    prompt: PromptEncoding,
    fixture: _TokenizerFixture,
    record_sequences: tuple[tuple[int, ...], ...],
) -> TrainingEncoding:
    record_ids = tuple(token_id for sequence in record_sequences for token_id in sequence)
    framed = (fixture.bos_token_id, *record_ids, fixture.eos_token_id)
    input_ids = (*prompt.input_ids, *framed)
    attention_mask = (*prompt.attention_mask, *((1,) * len(framed)))
    labels = (*((-100,) * len(prompt.input_ids)), *framed)
    return TrainingEncoding(input_ids, attention_mask, labels)


def _actual_training(backend: HuggingFaceBackend) -> TrainingEncoding:
    rows = backend.training_examples.train
    assert len(rows) == 1
    row = rows[0]
    return TrainingEncoding(tuple(row["input_ids"]), tuple(row["attention_mask"]), tuple(row["labels"]))


def _persist_and_reload(metadata: ModelMetadata, workdir: Workdir, destination: Path) -> ModelMetadata:
    metadata.save_metadata()
    shutil.copytree(workdir.train.adapter.path, destination)
    return ModelMetadata.from_metadata_json(destination / workdir.train.adapter.metadata.name)


def test_record_encoding_uses_shared_jsonl_serialization(tokenizers_dir: Path, tmp_path: Path) -> None:
    source = tokenizers_dir / "tinyllama"
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source, local_files_only=True))
    metadata = _metadata(_TOKENIZER_FIXTURES[0], native, _workdir(tmp_path / "workdir"))
    tokenization = bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)
    authority_text = '{"serialized-by":"record-utils"}\n'

    with patch.object(record_utils, "records_to_jsonl", return_value=authority_text):
        encoded = tokenization.encode_records([_REPRESENTATIVE_RECORD])

    assert tuple(record.utf8 for record in encoded.records) == (authority_text.encode(),)
    assert encoded.input_ids == (tuple(native.encode(authority_text, add_special_tokens=False)),)


@pytest.mark.parametrize("fixture", _TOKENIZER_FIXTURES, ids=lambda fixture: fixture.name)
@pytest.mark.parametrize("case", _PIPELINE_CASES, ids=lambda case: case.name)
def test_training_pipeline_uses_complete_production_encoding(
    tokenizers_dir: Path,
    tmp_path: Path,
    fixture: _TokenizerFixture,
    case: _PipelineCase,
) -> None:
    source = tokenizers_dir / fixture.name
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source, local_files_only=True))
    workdir = _workdir(tmp_path / "workdir")
    metadata = _metadata(fixture, native, workdir)
    metadata.tokenization = bind_tokenizer(native, metadata, workload_kind=case.workload)
    metadata.tokenizer = native
    backend = _training_backend(
        _parameters(source, case),
        metadata,
        workdir,
        Dataset.from_list(list(case.records)),
    )

    assert backend.tokenization is metadata.tokenization
    assert backend.tokenizer is metadata.tokenization.native
    backend.prepare_training_data()
    schema = backend.dataset_schema
    assert schema is not None
    assert tuple(schema["properties"]) == case.expected_columns

    expected_prompt = _expected_prompt(native, fixture, case.expected_columns, "")
    record_sequences = _expected_record_ids(native, case)
    expected_training = _expected_training(expected_prompt, fixture, record_sequences)
    assert backend.training_prompt_encoding == expected_prompt
    assert _actual_training(backend) == expected_training
    if case.workload is WorkloadKind.TIME_SERIES:
        expected_prefill = " " + "".join(
            _record_text(record, case.expected_columns)
            for record in sorted(case.records, key=lambda record: record["time"])
        )
        assert metadata.initial_prefill == {"A": expected_prefill}
        assert metadata.tokenization.render_prompt(
            case.expected_columns,
            metadata.instruction,
            current_prefill=expected_prefill,
        ) == _expected_prompt(native, fixture, case.expected_columns, expected_prefill)

    repeated = metadata.tokenization.frame_training(
        expected_prompt,
        [tuple(token_id for sequence in record_sequences for token_id in sequence)],
    )
    assert repeated == expected_training
    assert (
        metadata.tokenization.frame_training(
            expected_prompt,
            [tuple(token_id for sequence in record_sequences for token_id in sequence)],
        )
        == expected_training
    )

    loaded = _persist_and_reload(metadata, workdir, tmp_path / "relocated")
    assert loaded.tokenization is not None
    ordered_records = tuple({column: record[column] for column in case.expected_columns} for record in case.records)
    persisted_records = loaded.tokenization.encode_records(ordered_records)
    expected_persisted_ids = tuple(
        tuple(native.encode(_record_text(record, case.expected_columns), add_special_tokens=False))
        for record in case.records
    )
    assert tuple(record.utf8.decode() for record in persisted_records.records) == tuple(
        _record_text(record, case.expected_columns) for record in case.records
    )
    assert persisted_records.input_ids == expected_persisted_ids
    assert loaded.tokenization.render_prompt(case.expected_columns, loaded.instruction) == expected_prompt
    sorted_persisted_ids = (
        tuple(reversed(persisted_records.input_ids)) if case.order_by else persisted_records.input_ids
    )
    assert (
        loaded.tokenization.frame_training(
            expected_prompt,
            [tuple(token_id for sequence in sorted_persisted_ids for token_id in sequence)],
        )
        == expected_training
    )
    if case.workload is WorkloadKind.TIME_SERIES:
        assert loaded.initial_prefill == metadata.initial_prefill
        assert isinstance(loaded.initial_prefill, dict)
        assert loaded.tokenization.render_prompt(
            case.expected_columns,
            loaded.instruction,
            current_prefill=loaded.initial_prefill["A"],
        ) == _expected_prompt(native, fixture, case.expected_columns, loaded.initial_prefill["A"])


@pytest.mark.parametrize("fixture", _TOKENIZER_FIXTURES, ids=lambda fixture: fixture.name)
@pytest.mark.parametrize("case", _PROMPT_CASES, ids=lambda case: case.name)
def test_production_prompt_rendering_is_exact_and_persisted(
    tokenizers_dir: Path,
    tmp_path: Path,
    fixture: _TokenizerFixture,
    case: _PromptCase,
) -> None:
    source = tokenizers_dir / fixture.name
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source, local_files_only=True))
    workdir = _workdir(tmp_path / "workdir")
    metadata = _metadata(fixture, native, workdir)
    metadata.tokenization = bind_tokenizer(native, metadata, workload_kind=case.pipeline.workload)
    metadata.tokenizer = native
    expected = _expected_prompt(native, fixture, case.pipeline.expected_columns, case.prefill)

    first = metadata.tokenization.render_prompt(
        case.pipeline.expected_columns,
        metadata.instruction,
        current_prefill=case.prefill,
    )
    second = metadata.tokenization.render_prompt(
        case.pipeline.expected_columns,
        metadata.instruction,
        current_prefill=case.prefill,
    )
    assert first == expected
    assert second == expected

    loaded = _persist_and_reload(metadata, workdir, tmp_path / "relocated")
    assert loaded.tokenization is not None
    assert (
        loaded.tokenization.render_prompt(
            case.pipeline.expected_columns,
            loaded.instruction,
            current_prefill=case.prefill,
        )
        == expected
    )


@pytest.mark.parametrize("fixture", _TOKENIZER_FIXTURES, ids=lambda fixture: fixture.name)
@pytest.mark.parametrize("case", (_TABULAR, _GROUPED), ids=lambda case: case.name)
def test_tabular_prompt_modes_reject_non_empty_prefills(
    tokenizers_dir: Path,
    tmp_path: Path,
    fixture: _TokenizerFixture,
    case: _PipelineCase,
) -> None:
    source = tokenizers_dir / fixture.name
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source, local_files_only=True))
    metadata = _metadata(fixture, native, _workdir(tmp_path / "workdir"))
    tokenization = bind_tokenizer(native, metadata, workload_kind=case.workload)

    with pytest.raises(ParameterError, match="do not support a prefill"):
        tokenization.render_prompt(
            case.expected_columns,
            metadata.instruction,
            current_prefill=_SPACE_LED_PREFILL,
        )
