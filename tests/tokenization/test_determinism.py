# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-stage tokenizer identity and deterministic rendering contracts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.generation.timeseries_backend import TimeseriesBackend
from nemo_safe_synthesizer.generation.vllm_backend import VllmBackend
from nemo_safe_synthesizer.llm.metadata import LLMPromptConfig, ModelMetadata
from nemo_safe_synthesizer.tokenization import PromptEncoding, WorkloadKind, bind_tokenizer
from nemo_safe_synthesizer.training.huggingface_backend import HuggingFaceBackend


@dataclass(frozen=True)
class _PipelineCase:
    name: str
    workload: WorkloadKind
    group_by: str | None
    order_by: str | None
    expected_columns: tuple[str, ...]


_INPUT_COLUMNS = ("value", "group", "time", "name")
_CASES = (
    _PipelineCase("tabular", WorkloadKind.TABULAR, None, None, _INPUT_COLUMNS),
    _PipelineCase("grouped", WorkloadKind.TABULAR, "group", "time", _INPUT_COLUMNS),
    _PipelineCase("time-series", WorkloadKind.TIME_SERIES, "group", "time", ("group", "time", "value", "name")),
)
_RECORDS = (
    {"value": -1.5, "group": "A", "time": 0, "name": "é/☃"},
    {"value": 2.0, "group": "A", "time": 60, "name": "line\nbreak"},
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


def _metadata(native: PreTrainedTokenizerBase, workdir: Workdir) -> ModelMetadata:
    bos_token = native.bos_token or "<|im_start|>"
    bos_token_id = native.convert_tokens_to_ids(bos_token)
    assert isinstance(bos_token_id, int)
    assert native.eos_token is not None
    assert native.eos_token_id is not None
    return ModelMetadata(
        model_name_or_path=str(native.name_or_path),
        autoconfig=_autoconfig(),
        base_max_seq_length=2048,
        prompt_config=LLMPromptConfig(
            template="I:{instruction}|S:{schema}|P:{prefill}",
            add_bos_token_to_prompt=True,
            add_eos_token_to_prompt=True,
            bos_token=bos_token,
            bos_token_id=bos_token_id,
            eos_token=native.eos_token,
            eos_token_id=native.eos_token_id,
        ),
        workdir=workdir,
    )


def _parameters(source: Path, case: _PipelineCase) -> SafeSynthesizerParameters:
    values: dict[str, object] = {
        "pretrained_model": str(source),
        "num_input_records_to_sample": len(_RECORDS),
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
    return SafeSynthesizerParameters.from_params(
        **values,
    )


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


def _dispatch_prompt(
    metadata: ModelMetadata,
    case: _PipelineCase,
    columns: tuple[str, ...],
    prefill: str,
) -> tuple[PromptEncoding, dict[str, object]]:
    backend_type = TimeseriesBackend if case.workload is WorkloadKind.TIME_SERIES else VllmBackend
    backend = object.__new__(backend_type)
    backend.model_metadata = metadata
    backend.columns = list(columns)
    backend._torn_down = True
    captured: dict[str, object] = {}

    def generate(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return []

    backend._gen_method = partial(generate, sampling_params=object())
    prompt = backend._render_token_prompt(prefill)
    assert prompt is not None
    backend._generate(input_ids=list(prompt.input_ids))
    return prompt, captured


def _prepare_training(backend: HuggingFaceBackend, case: _PipelineCase) -> tuple[str, ...]:
    backend.prepare_training_data()
    schema = backend.dataset_schema
    assert schema is not None
    columns = tuple(schema["properties"])
    assert columns == case.expected_columns
    assert all(
        tuple(row[: len(backend.training_prompt_encoding.input_ids)]) == backend.training_prompt_encoding.input_ids
        for row in backend.training_examples.train["input_ids"]
    )
    return columns


def _contains_tokens(container: list[int], tokens: tuple[int, ...]) -> bool:
    return any(
        tuple(container[index : index + len(tokens)]) == tokens for index in range(len(container) - len(tokens) + 1)
    )


@pytest.mark.parametrize("fixture_name", ["tinyllama", "mistral7b", "smollm3b"])
@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.name)
def test_pipeline_uses_identical_tokenization(
    tokenizers_dir: Path,
    tmp_path: Path,
    fixture_name: str,
    case: _PipelineCase,
) -> None:
    source = tokenizers_dir / fixture_name
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source, local_files_only=True))
    workdir = _workdir(tmp_path / "workdir")
    metadata = _metadata(native, workdir)
    metadata.tokenization = bind_tokenizer(native, metadata, workload_kind=case.workload)
    metadata.tokenizer = native
    backend = _training_backend(_parameters(source, case), metadata, workdir, Dataset.from_list(list(_RECORDS)))

    assert backend.tokenization is metadata.tokenization
    assert backend.tokenizer is metadata.tokenization.native
    columns = _prepare_training(backend, case)

    ordered_records = tuple({column: record[column] for column in case.expected_columns} for record in _RECORDS)
    expected_records = metadata.tokenization.encode_records(ordered_records)
    assert all(
        any(_contains_tokens(row, record.input_ids) for row in backend.training_examples.train["input_ids"])
        for record in expected_records.records
    )
    metadata.save_metadata()
    relocated = tmp_path / "relocated"
    shutil.copytree(workdir.train.adapter.path, relocated)
    loaded = ModelMetadata.from_metadata_json(relocated / workdir.train.adapter.metadata.name)
    assert loaded.tokenization is not None
    assert loaded.tokenization.encode_records(ordered_records) == expected_records

    prefill = '{"group":"A","time":0,"value":-1.5,"name":"é/☃"}\n' if case.workload is WorkloadKind.TIME_SERIES else ""
    expected_prompt = metadata.tokenization.render_prompt(
        case.expected_columns,
        metadata.instruction,
        current_prefill=prefill,
    )
    generated_prompt, captured = _dispatch_prompt(loaded, case, columns, prefill)
    assert generated_prompt == expected_prompt
    assert captured["prompts"] == {"prompt_token_ids": list(expected_prompt.input_ids)}
