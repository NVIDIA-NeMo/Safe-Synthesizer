# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for NVIDIA Nemotron 3 Nano 4B BF16 support."""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import torch
from transformers import NemotronHConfig, PretrainedConfig, PreTrainedTokenizerBase

from nemo_safe_synthesizer.config.autoconfig import AutoConfigResolver
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.data_processing.assembler import Example
from nemo_safe_synthesizer.data_processing.budget import compute_max_new_tokens
from nemo_safe_synthesizer.generation.processors import create_processor
from nemo_safe_synthesizer.generation.regex_manager import build_json_based_regex
from nemo_safe_synthesizer.generation.timeseries_backend import TimeseriesBackend
from nemo_safe_synthesizer.llm import model_policy
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.llm.model_policy import NEMOTRON3_NANO_LAYER_BLOCK_TYPES
from nemo_safe_synthesizer.llm.utils import ModelRef
from nemo_safe_synthesizer.preflight.checks.environment import param_count_from_empty_model

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
LAYER_TYPES = list(NEMOTRON3_NANO_LAYER_BLOCK_TYPES)


def _write_local_model_config(path: Path, **overrides) -> Path:
    config = {
        "architectures": ["NemotronHForCausalLM"],
        "hidden_size": 3136,
        "layers_block_type": LAYER_TYPES,
        "mamba_head_dim": 80,
        "mamba_num_heads": 96,
        "model_type": "nemotron_h",
        "ssm_state_size": 128,
        "dtype": "bfloat16",
        "vocab_size": 131072,
        **overrides,
    }
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    return path


class StubNemotronTokenizer:
    """Small tokenizer double that preserves the official chat boundaries."""

    bos_token = "<s>"
    bos_token_id = 1
    eos_token = "<|im_end|>"
    eos_token_id = 11
    name_or_path = MODEL_ID

    _pieces = {
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "<think>": 12,
        "</think>": 13,
        "\n": 1010,
    }

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=True,
        **_kwargs,
    ):
        assert tokenize is False
        system = messages[0]["content"]
        user = messages[1]["content"]
        prefix = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        if add_generation_prompt:
            return prefix + ("<think>\n" if enable_thinking else "<think></think>")
        if len(messages) == 2:
            return prefix.removesuffix("<|im_start|>assistant\n")
        assistant = messages[2]["content"]
        return prefix + f"<think></think>{assistant}<|im_end|>\n"

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        for marker, token_id in sorted(self._pieces.items(), key=lambda item: -len(item[0])):
            text = text.replace(marker, chr(0xE000 + token_id))
        return [ord(char) - 0xE000 if ord(char) >= 0xE000 else 2000 + ord(char) for char in text]


def _metadata() -> ModelMetadata:
    stub = StubNemotronTokenizer()
    tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    tokenizer.bos_token = stub.bos_token
    tokenizer.bos_token_id = stub.bos_token_id
    tokenizer.eos_token = stub.eos_token
    tokenizer.eos_token_id = stub.eos_token_id
    tokenizer.name_or_path = stub.name_or_path
    tokenizer.apply_chat_template.side_effect = stub.apply_chat_template
    tokenizer.encode.side_effect = stub.encode
    config = PretrainedConfig()
    setattr(config, "max_position_embeddings", 262_144)
    with patch.object(ModelMetadata, "_load_config_and_tokenizer", return_value=(config, tokenizer)):
        return ModelMetadata.from_str_or_path(MODEL_ID)


def test_exact_bf16_id_resolves_native_nemotron3_policy() -> None:
    metadata = _metadata()

    assert type(metadata).__name__ == "Nemotron3Nano"
    assert metadata.uses_rope is False
    assert metadata.base_max_seq_length == 12_288
    assert ModelRef.parse(MODEL_ID).trust_remote_code is False


def test_local_bf16_checkpoint_config_resolves_native_nemotron3_policy(tmp_path: Path) -> None:
    model_path = _write_local_model_config(tmp_path / "renamed-local-checkpoint")

    assert ModelMetadata._resolve_model_class(model_path).__name__ == "Nemotron3Nano"
    assert model_policy.model_policy_for_local_path(model_path) is model_policy.NEMOTRON3_NANO_POLICY


def test_transformers_saved_local_checkpoint_resolves_native_nemotron3_policy(tmp_path: Path) -> None:
    model_path = tmp_path / "transformers-saved-checkpoint"
    config = NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        hidden_size=3136,
        layers_block_type=LAYER_TYPES,
        mamba_head_dim=80,
        mamba_num_heads=96,
        ssm_state_size=128,
        dtype="bfloat16",
        vocab_size=131072,
    )

    config.save_pretrained(model_path)

    assert model_policy.model_policy_for_local_path(model_path) is model_policy.NEMOTRON3_NANO_POLICY
    assert ModelMetadata._resolve_model_class(model_path).__name__ == "Nemotron3Nano"


def test_legacy_local_checkpoint_schema_resolves_native_nemotron3_policy(tmp_path: Path) -> None:
    model_path = _write_local_model_config(tmp_path / "legacy-checkpoint")
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    config["num_hidden_layers"] = len(config.pop("layers_block_type"))
    config["torch_dtype"] = config.pop("dtype")
    config_path.write_text(json.dumps(config))

    assert model_policy.model_policy_for_local_path(model_path) is model_policy.NEMOTRON3_NANO_POLICY


def test_local_quantized_checkpoint_does_not_match_bf16_policy(tmp_path: Path) -> None:
    model_path = _write_local_model_config(
        tmp_path / "quantized-local-checkpoint",
        quantization_config={"quant_method": "fp8"},
    )

    assert model_policy.model_policy_for_local_path(model_path) is None


def test_reasoning_off_prompt_places_timeseries_prefill_inside_assistant_turn() -> None:
    metadata = _metadata()

    prompt = metadata.render_prompt(["device", "timestamp", "value"], prefill='{"device":"A"')

    assert prompt.endswith('<think></think>{"device":"A"')
    assert "</think><s>" not in prompt
    assert metadata.response_prefix_ids[-2:] == [12, 13]
    assert metadata.response_suffix_ids == [11, 1010]


def test_chat_prompt_renderer_caches_a_reloaded_tokenizer() -> None:
    metadata = _metadata()
    tokenizer = metadata.tokenizer
    assert tokenizer is not None
    metadata.tokenizer = None

    with patch("nemo_safe_synthesizer.llm.metadata.load_fast_tokenizer", return_value=tokenizer) as load:
        metadata.render_prompt(["device"])
        metadata.render_prompt(["device"], prefill='{"device":"A"')

    load.assert_called_once()
    assert metadata.tokenizer is tokenizer


def test_nemotron3_autoconfig_disables_rope_and_selects_hybrid_lora_targets() -> None:
    params = SafeSynthesizerParameters.from_params(pretrained_model=MODEL_ID)
    data = pd.DataFrame({"device": ["A", "A"], "timestamp": [0, 60], "value": [1.0, 2.0]})

    with patch("nemo_safe_synthesizer.config.autoconfig.choose_rope_scaling_factor", return_value=4):
        resolved = AutoConfigResolver(data=data, config=params).resolve()

    assert resolved.training.rope_scaling_factor == 1
    assert resolved.training.lora_target_modules == [
        "in_proj",
        "up_proj",
        "down_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def test_nemotron3_rejects_explicit_rope_scaling() -> None:
    params = SafeSynthesizerParameters.from_params(pretrained_model=MODEL_ID, rope_scaling_factor=2)

    with pytest.raises(ValueError, match="does not use RoPE"):
        AutoConfigResolver(data=pd.DataFrame({"value": [1]}), config=params).resolve()


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"quantize_model": True}, "does not support quantized training"),
        ({"privacy.dp_enabled": True}, "does not support differential-privacy training"),
    ],
)
def test_nemotron3_rejects_unsupported_training_modes(parameters, message) -> None:
    params = SafeSynthesizerParameters.from_params(pretrained_model=MODEL_ID, **parameters)

    with pytest.raises(ValueError, match=message):
        AutoConfigResolver(data=pd.DataFrame({"value": [1]}), config=params).resolve()


def test_training_example_masks_chat_prefix_and_labels_json_and_assistant_suffix() -> None:
    metadata = _metadata()
    tokenizer = metadata.tokenizer
    assert tokenizer is not None
    prompt = metadata.render_prompt(["device", "timestamp", "value"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    record_ids = [3001, 3002, 3003]
    example = Example(
        prompt=prompt,
        tokenizer=tokenizer,  # ty: ignore[invalid-argument-type] -- tokenizer mock implements the required backend.
        metadata=metadata,
    )

    example.add_sequence({"input_ids": record_ids, "attention_mask": [1, 1, 1]})

    expected_response = [*record_ids, 11, 1010]
    assert example.input_ids == [*prompt_ids, *expected_response]
    assert example.labels == [-100] * len(prompt_ids) + expected_response
    assert metadata.prompt_config.bos_token_id not in example.input_ids[len(prompt_ids) :]


def test_grouped_training_reopens_an_assistant_turn_between_sequences() -> None:
    metadata = _metadata()
    tokenizer = metadata.tokenizer
    assert tokenizer is not None
    prompt = metadata.render_prompt(["device", "value"])
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    example = Example(
        prompt=prompt,
        tokenizer=tokenizer,  # ty: ignore[invalid-argument-type] -- tokenizer mock implements the required backend.
        metadata=metadata,
    )

    example.add_sequence({"input_ids": [3001], "attention_mask": [1]})
    example.add_sequence({"input_ids": [3002], "attention_mask": [1]})

    expected_response = [
        3001,
        *metadata.response_suffix_ids,
        *metadata.response_prefix_ids,
        3002,
        *metadata.response_suffix_ids,
    ]
    assert example.input_ids == [*prompt_ids, *expected_response]
    assert example.labels == [-100] * len(prompt_ids) + expected_response


def test_grouped_generation_uses_the_same_chat_boundaries_as_training() -> None:
    metadata = _metadata()
    framing = metadata.response_framing
    schema = {
        "type": "object",
        "properties": {
            "device": {"type": "string"},
            "value": {"type": "integer"},
        },
        "required": ["device", "value"],
    }
    config = SafeSynthesizerParameters.from_params(
        group_training_examples_by="device",
        max_sequences_per_example=2,
    )
    completion = (
        f'{{"device":"A","value":1}}\n{framing.suffix}'
        f'{framing.subsequent_prefix}{{"device":"B","value":2}}\n{framing.suffix}'
    )

    regex = build_json_based_regex(
        schema,
        config,
        bos_token=metadata.prompt_config.bos_token,
        eos_token=metadata.prompt_config.eos_token,
        response_framing=framing,
    )
    response = create_processor(schema, metadata, config)(0, completion)

    assert re.fullmatch(regex, completion) is not None
    assert response.invalid_records == []
    assert response.valid_records == [
        {"device": "A", "value": 1},
        {"device": "B", "value": 2},
    ]


def test_token_budget_uses_chat_response_suffix_instead_of_four_generic_tokens() -> None:
    metadata = _metadata()
    prompt_ids = list(range(100))

    assert compute_max_new_tokens(prompt_ids, 1000, metadata=metadata) == 898


def test_timeseries_backend_uses_model_renderer_for_sliding_prefill() -> None:
    metadata = _metadata()
    backend = object.__new__(TimeseriesBackend)
    backend.model_metadata = metadata
    backend.schema = {"properties": {"device": {}, "timestamp": {}, "value": {}}}

    prompt = backend._format_prompt('{"device":"A","timestamp":120')

    assert prompt.endswith('<think></think>{"device":"A","timestamp":120')


def test_lora_target_validation_reports_every_suffix_and_rejects_missing_targets() -> None:
    model = torch.nn.Module()
    model.in_proj = torch.nn.Linear(4, 8, bias=False)
    model.q_proj = torch.nn.Linear(4, 4, bias=False)
    validate = getattr(model_policy, "validate_lora_targets")

    assert validate(model, ["in_proj", "q_proj"]) == {"in_proj": 1, "q_proj": 1}
    with pytest.raises(ValueError, match="gate_proj"):
        validate(model, ["in_proj", "gate_proj"])


def test_meta_parameter_count_disables_mamba_kernels_on_a_config_copy() -> None:
    config = PretrainedConfig()
    setattr(config, "use_mamba_kernels", True)
    object.__setattr__(config, "model_type", "nemotron_h")
    observed = {}
    fake_model = MagicMock()
    fake_model.parameters.return_value = [torch.nn.Parameter(torch.empty(7, device="meta"))]

    def from_config(candidate):
        observed["candidate"] = candidate
        return fake_model

    with patch("transformers.AutoModelForCausalLM.from_config", side_effect=from_config):
        assert param_count_from_empty_model(config) == 7

    assert observed["candidate"] is not config
    assert observed["candidate"].use_mamba_kernels is False
    assert config.use_mamba_kernels is True
