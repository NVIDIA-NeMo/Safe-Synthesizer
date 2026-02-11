# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest
from datasets import Dataset
from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.defaults import DEFAULT_INSTRUCTION
from nemo_safe_synthesizer.llm.metadata import LLMPromptConfig, ModelMetadata
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from transformers import AutoConfig, AutoTokenizer, LlamaConfig, LlamaForCausalLM


class SmolLM2(ModelMetadata):
    """Test-only metadata for HuggingFaceTB/SmolLM2-135M.

    Moved out of production code because SmolLM2-135M is only used in smoke tests.
    Uses the tokenizer's native BOS token (not the Qwen2 <|im_start|> override
    that was previously hardcoded -- that token ID is out of range for SmolLM2's
    49K vocab and causes CUDA device-side asserts).
    """

    def __init__(
        self, model_name_or_path: str, tokenizer=None, rope_scaling_factor: float | None = None, **kwargs
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config = AutoConfig.from_pretrained(model_name_or_path)

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                template="user\n {instruction} {schema} \n assistant\n{prefill}",
                add_bos_token_to_prompt=False,
                add_eos_token_to_prompt=False,
                tokenizer=tokenizer,
                name=model_name_or_path,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=None,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


@pytest.fixture
def _register_smollm2(monkeypatch):
    """Patch ModelMetadata resolution so the SDK can find SmolLM2 (test-only class)."""
    original = ModelMetadata.from_str_or_path.__func__

    def patched(cls, model_name_or_path, **kwargs):
        if "smollm2" in str(model_name_or_path).lower():
            return SmolLM2(model_name_or_path=str(model_name_or_path), **kwargs)
        return original(cls, model_name_or_path, **kwargs)

    monkeypatch.setattr(ModelMetadata, "from_str_or_path", classmethod(patched))


@pytest.fixture(scope="session")
def fixture_stub_tokenizer_path() -> str:
    """Session-scoped override of the function-scoped fixture in tests/conftest.py."""
    return str(Path(__file__).parent.parent / "stub_tokenizer")


@pytest.fixture(scope="session")
def tiny_llama_config(fixture_stub_tokenizer_path):
    """LlamaConfig with minimal dimensions for fast smoke testing."""
    tokenizer = AutoTokenizer.from_pretrained(fixture_stub_tokenizer_path)
    return LlamaConfig(
        vocab_size=tokenizer.vocab_size,  # 32000 -- must match stub tokenizer
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )


@pytest.fixture
def tiny_model(tiny_llama_config):
    """Randomly initialized LlamaForCausalLM. Tiny (~few KB), no download."""
    return LlamaForCausalLM(tiny_llama_config)


@pytest.fixture(scope="session")
def stub_tokenizer(fixture_stub_tokenizer_path):
    """Load the Llama stub tokenizer from tests/stub_tokenizer/."""
    return AutoTokenizer.from_pretrained(fixture_stub_tokenizer_path)


@pytest.fixture(scope="session")
def tiny_training_dataset(stub_tokenizer):
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
    tokenized = stub_tokenizer(texts, padding="max_length", truncation=True, max_length=64, return_tensors="np")
    return Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"].tolist(),
            "attention_mask": tokenized["attention_mask"].tolist(),
            "labels": tokenized["input_ids"].tolist(),  # labels = input_ids for causal LM
        }
    )


@pytest.fixture(scope="session")
def tiny_training_dataset_with_position_ids(tiny_training_dataset):
    """Training dataset with position_ids column, required by DataCollatorForPrivateTokenClassification."""
    seq_len = len(tiny_training_dataset[0]["input_ids"])
    position_ids = [list(range(seq_len))] * len(tiny_training_dataset)
    return tiny_training_dataset.add_column("position_ids", position_ids)


@pytest.fixture(scope="session")
def local_tinyllama_dir(tmp_path_factory, tiny_llama_config, stub_tokenizer):
    """Save tiny model + tokenizer to a local dir named with 'tinyllama' for NSS compatibility."""
    local_dir = tmp_path_factory.mktemp("smoke-tinyllama-model")
    model = LlamaForCausalLM(tiny_llama_config)
    model.save_pretrained(local_dir)
    stub_tokenizer.save_pretrained(local_dir)
    return local_dir


@pytest.fixture(scope="session")
def iris_df():
    """Load iris.csv from stub_datasets."""
    from tests.conftest import load_test_dataframe

    return load_test_dataframe("iris.csv").copy()


@pytest.fixture(scope="session")
def timeseries_df():
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
    ).copy()


@pytest.fixture(scope="session")
def smoke_save_path(tmp_path_factory):
    """Shared temp directory for Tier B (SmolLM2) train -> generate flow."""
    return tmp_path_factory.mktemp("smoke-tier-b")


@pytest.fixture
def base_smoke_config(local_tinyllama_dir):
    """Base SafeSynthesizerParameters shared by all GPU smoke tests with local tiny model.

    Individual tests override specific fields via SafeSynthesizerParameters.from_params(**overrides).
    """
    return SafeSynthesizerParameters.from_params(
        enable_synthesis=True,
        enable_replace_pii=False,
        pretrained_model=str(local_tinyllama_dir),
        use_unsloth=False,
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


@pytest.fixture
def _patch_attn_eager(monkeypatch):
    """Override attn_implementation from 'flashinfer' (not a valid HF option) to 'sdpa'.

    The HuggingFaceBackend defaults to 'flashinfer' which is not supported by
    HuggingFace's from_pretrained. PyTorch SDPA is universally compatible.
    """
    from nemo_safe_synthesizer.training.huggingface_backend import HuggingFaceBackend

    original_build = HuggingFaceBackend._build_base_framework_params

    def patched_build(self, model_kwargs):
        model_kwargs.setdefault("attn_implementation", "sdpa")
        return original_build(self, model_kwargs)

    monkeypatch.setattr(HuggingFaceBackend, "_build_base_framework_params", patched_build)
