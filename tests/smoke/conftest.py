# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke-test fixtures: tiny models, stub tokenizers, minimal datasets."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerBase

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer


@pytest.fixture(scope="session")
def fixture_stub_tokenizer_path(stub_tokenizer_dir) -> str:
    """Path to the Llama stub tokenizer in tests/stub_tokenizer/."""
    return str(stub_tokenizer_dir)


@pytest.fixture(scope="session")
def fixture_tiny_llama_config(fixture_stub_tokenizer) -> LlamaConfig:
    """LlamaConfig with minimal dimensions for fast smoke testing."""
    return LlamaConfig(
        vocab_size=fixture_stub_tokenizer.vocab_size,  # 32000 -- must match stub tokenizer
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )


@pytest.fixture
def fixture_tiny_model(fixture_tiny_llama_config) -> LlamaForCausalLM:
    """Randomly initialized LlamaForCausalLM. Tiny (~few KB), no download."""
    return LlamaForCausalLM(fixture_tiny_llama_config)


@pytest.fixture(scope="session")
def fixture_stub_tokenizer(fixture_stub_tokenizer_path) -> PreTrainedTokenizerBase:
    """Load the Llama stub tokenizer from tests/stub_tokenizer/."""
    return AutoTokenizer.from_pretrained(fixture_stub_tokenizer_path)


@pytest.fixture(scope="session")
def fixture_tiny_training_dataset(fixture_stub_tokenizer) -> Dataset:
    """~8 tokenized training examples as a datasets.Dataset."""
    texts = [
        '{"col1":"a","col2":"1"}',
        '{"col1":"b","col2":"2"}',
        '{"col1":"c","col2":"3"}',
        '{"col1":"d","col2":"4"}',
        '{"col1":"e","col2":"5"}',
        '{"col1":"f","col2":"6"}',
        '{"col1":"g","col2":"7"}',
        '{"col1":"h","col2":"8"}',
    ]
    tokenized = fixture_stub_tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="np")
    return Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"].tolist(),
            "attention_mask": tokenized["attention_mask"].tolist(),
            "labels": tokenized["input_ids"].tolist(),  # labels = input_ids for causal LM
        }
    )


@pytest.fixture(scope="session")
def fixture_tiny_training_dataset_with_position_ids(fixture_tiny_training_dataset) -> Dataset:
    """Training dataset with position_ids column, required by DataCollatorForPrivateTokenClassification."""
    seq_len = len(fixture_tiny_training_dataset[0]["input_ids"])
    position_ids = [list(range(seq_len))] * len(fixture_tiny_training_dataset)
    return fixture_tiny_training_dataset.add_column("position_ids", position_ids)


@pytest.fixture(scope="session")
def fixture_local_tinyllama_dir(tmp_path_factory, fixture_tiny_llama_config, fixture_stub_tokenizer) -> Path:
    """Save tiny model + tokenizer to a local dir named with 'tinyllama' for NSS compatibility."""
    local_dir = tmp_path_factory.mktemp("smoke-tinyllama-model")
    model = LlamaForCausalLM(fixture_tiny_llama_config)
    model.save_pretrained(local_dir)
    fixture_stub_tokenizer.save_pretrained(local_dir)
    return local_dir


@pytest.fixture(scope="session")
def fixture_iris_df(stub_datasets_dir) -> pd.DataFrame:
    """Load iris.csv from stub_datasets."""
    return pd.read_csv(stub_datasets_dir / "iris.csv")


@pytest.fixture(scope="session")
def fixture_gpu_smoke_df(stub_datasets_dir) -> pd.DataFrame:
    """Small tabular dataset sized for GPU smoke tests that run preflight."""
    return pd.read_csv(stub_datasets_dir / "clinc_oos.csv", nrows=210)


@pytest.fixture(scope="session")
def fixture_timeseries_df() -> pd.DataFrame:
    """Minimal timeseries stub: 2 groups, 5 rows each, 60s intervals."""
    return pd.DataFrame(
        {
            "group_id": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "timestamp": [
                "2024-01-01 00:00:00",
                "2024-01-01 00:01:00",
                "2024-01-01 00:02:00",
                "2024-01-01 00:03:00",
                "2024-01-01 00:04:00",
                "2024-01-01 00:00:00",
                "2024-01-01 00:01:00",
                "2024-01-01 00:02:00",
                "2024-01-01 00:03:00",
                "2024-01-01 00:04:00",
            ],
            "value": [10, 20, 30, 40, 50, 100, 110, 120, 130, 140],
        }
    )


@pytest.fixture(scope="session")
def fixture_preflight_timeseries_df() -> pd.DataFrame:
    """Timeseries stub with 200 rows for GPU paths that run preflight."""
    start = pd.Timestamp("2024-01-01 00:00:00")
    rows = []
    for group, offset in (("A", 0), ("B", 1000)):
        for i in range(100):
            rows.append(
                {
                    "group_id": group,
                    "timestamp": start + pd.Timedelta(seconds=60 * i),
                    "value": offset + i,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def fixture_smoke_save_path(tmp_path_factory) -> Path:
    """Shared temp directory for Tier B (SmolLM2) train -> generate flow."""
    return tmp_path_factory.mktemp("smoke-tier-b")


@pytest.fixture(scope="session")
def fixture_base_smoke_config(fixture_local_tinyllama_dir) -> SafeSynthesizerParameters:
    """Base SafeSynthesizerParameters shared by all GPU smoke tests with local tiny model.

    Session-scoped because the config is immutable (Pydantic frozen model).
    Tests that need different settings create their own via SafeSynthesizerParameters.from_params().
    """
    return SafeSynthesizerParameters.from_params(
        replace_pii=None,
        pretrained_model=str(fixture_local_tinyllama_dir),
        num_input_records_to_sample=10,
        num_records=5,
        lora_r=8,
        holdout=0,
        max_holdout=0,
    )


def assert_adapter_saved(workdir: Workdir) -> None:
    """Verify adapter files exist after training.

    Reusable assertion helper for any test that trains via the SDK.
    """
    adapter_dir = workdir.train.adapter.path
    assert (adapter_dir / "adapter_config.json").exists(), "adapter_config.json missing"
    assert any(adapter_dir.glob("*.safetensors")), "No safetensors files found"


def train_with_sdk(config: SafeSynthesizerParameters, data_df: pd.DataFrame, save_path: Path) -> SafeSynthesizer:
    """Run SafeSynthesizer.process_data().train() and return the instance."""
    nss = SafeSynthesizer(config=config, save_path=save_path)
    nss.with_data_source(data_df).process_data().train()
    return nss


@pytest.fixture(scope="session")
def _patch_attn_eager() -> Generator[None, None, None]:
    """Override attn_implementation from 'flashinfer' (not a valid HF option) to 'sdpa'.

    Session-scoped so class-scoped and function-scoped fixtures can depend on it.
    The HuggingFaceBackend defaults to 'flashinfer' which is not supported by
    HuggingFace's from_pretrained. PyTorch SDPA is universally compatible.
    """
    from nemo_safe_synthesizer.training.huggingface_backend import HuggingFaceBackend

    original_build = HuggingFaceBackend._build_base_framework_params

    def patched_build(self, model_kwargs):
        model_kwargs.setdefault("attn_implementation", "sdpa")
        return original_build(self, model_kwargs)

    HuggingFaceBackend._build_base_framework_params = patched_build  # ty: ignore[invalid-assignment] -- test monkey-patch
    yield
    HuggingFaceBackend._build_base_framework_params = original_build
