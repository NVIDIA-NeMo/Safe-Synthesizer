# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU training smoke tests -- Trainer, LoRA, DP, Assembler.

All tests run on CPU with max_steps=1. The point is catching dep breakage
(torch + transformers + peft + opacus) and exercising the NSS data pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pytest
import torch
from datasets import Dataset
from packaging.version import Version
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForTokenClassification, LlamaConfig, LlamaForCausalLM, Trainer, TrainingArguments

from nemo_safe_synthesizer.data_processing.assembler import TrainingExampleAssembler
from nemo_safe_synthesizer.defaults import DEFAULT_INSTRUCTION, PROMPT_TEMPLATE
from nemo_safe_synthesizer.privacy.dp_transformers import dp_utils
from nemo_safe_synthesizer.privacy.dp_transformers.dp_utils import (
    DataCollatorForPrivateTokenClassification,
    OpacusDPTrainer,
)
from nemo_safe_synthesizer.privacy.dp_transformers.privacy_args import PrivacyArguments


def _cpu_training_args(tmp_path, **overrides):
    """Build TrainingArguments for CPU smoke tests with sensible defaults."""
    defaults: dict[str, Any] = dict(
        output_dir=str(tmp_path),
        max_steps=1,
        use_cpu=True,
        bf16=False,
        optim="adamw_torch",
        per_device_train_batch_size=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)


def _privacy_args(*, noise_multiplier: float | None = None, per_sample_max_grad_norm: float = 1.0) -> PrivacyArguments:
    if noise_multiplier is not None:
        return PrivacyArguments(
            noise_multiplier=noise_multiplier,
            target_delta=1e-5,
            per_sample_max_grad_norm=per_sample_max_grad_norm,
            use_prv=False,
        )
    return PrivacyArguments(
        target_epsilon=100.0,
        target_delta=1e-5,
        per_sample_max_grad_norm=per_sample_max_grad_norm,
    )


def _private_data_collator(fixture_stub_tokenizer) -> DataCollatorForPrivateTokenClassification:
    return DataCollatorForPrivateTokenClassification(tokenizer=fixture_stub_tokenizer)


def _dp_trainer(
    *,
    model,
    tokenizer,
    train_dataset,
    tmp_path,
    grad_sample_mode: Literal["hooks", "ghost"] = "hooks",
    privacy_args: PrivacyArguments | None = None,
    secure_mode: bool = True,
    true_dataset_size: int = 8,
    **training_arg_overrides,
) -> OpacusDPTrainer:
    return OpacusDPTrainer(
        model=model,
        args=_cpu_training_args(
            tmp_path,
            remove_unused_columns=False,
            max_grad_norm=0.0,
            **training_arg_overrides,
        ),
        train_dataset=train_dataset,
        data_collator=_private_data_collator(tokenizer),
        privacy_args=privacy_args or _privacy_args(),
        grad_sample_mode=grad_sample_mode,
        secure_mode=secure_mode,
        true_dataset_size=true_dataset_size,
        data_fraction=1.0,
    )


def _tiny_lora_model(base_model):
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    return get_peft_model(base_model, lora_config)


def _trainable_parameter_state(model) -> dict[str, torch.Tensor]:
    """Return detached trainable parameters with Opacus wrapper prefixes normalized."""
    return {
        name.removeprefix("_module."): parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


@dataclass
class _StubPromptConfig:
    """Minimal picklable prompt config for assembler tests."""

    template: str = PROMPT_TEMPLATE
    add_bos_token_to_prompt: bool = False
    add_eos_token_to_prompt: bool = False
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class _StubModelMetadata:
    """Minimal picklable model metadata for assembler tests."""

    instruction: str = DEFAULT_INSTRUCTION
    max_seq_length: int = 128
    rope_scaling_factor: float = 1.0
    max_sequences_per_example: int | None = None
    prompt_config: _StubPromptConfig = field(default_factory=_StubPromptConfig)


def test_hf_trainer_one_step(fixture_tiny_model, fixture_stub_tokenizer, fixture_tiny_training_dataset, tmp_path):
    """Exercises: transformers.Trainer forward + backward pass."""
    trainer = Trainer(
        model=fixture_tiny_model,
        args=_cpu_training_args(tmp_path),
        train_dataset=fixture_tiny_training_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=fixture_stub_tokenizer),
    )
    trainer.train()
    assert len(trainer.state.log_history) > 0
    last_log = trainer.state.log_history[-1]
    assert "loss" in last_log or "train_loss" in last_log


def test_lora_training_one_step(fixture_tiny_model, fixture_stub_tokenizer, fixture_tiny_training_dataset, tmp_path):
    """Exercises: peft.get_peft_model + LoraConfig + Trainer."""
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(fixture_tiny_model, lora_config)
    model.enable_input_require_grads()
    trainer = Trainer(
        model=model,
        args=_cpu_training_args(tmp_path),
        train_dataset=fixture_tiny_training_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=fixture_stub_tokenizer),
    )
    trainer.train()
    assert len(trainer.state.log_history) > 0
    last_log = trainer.state.log_history[-1]
    assert "loss" in last_log or "train_loss" in last_log


def test_dp_training_one_step(
    fixture_tiny_model, fixture_stub_tokenizer, fixture_tiny_training_dataset_with_position_ids, tmp_path
):
    """Exercises: OpacusDPTrainer + PrivacyArguments + DataCollatorForPrivateTokenClassification."""
    trainer = _dp_trainer(
        model=fixture_tiny_model,
        tokenizer=fixture_stub_tokenizer,
        train_dataset=fixture_tiny_training_dataset_with_position_ids,
        tmp_path=tmp_path,
    )
    trainer.train()
    assert len(trainer.state.log_history) > 0


def test_dp_ghost_clipping_requires_opacus_1_6(
    fixture_tiny_model,
    fixture_stub_tokenizer,
    fixture_tiny_training_dataset_with_position_ids,
    tmp_path,
    monkeypatch,
):
    """Ghost clipping is gated on the Opacus release with causal-LM ignore-index fixes."""
    monkeypatch.setattr(dp_utils, "_get_opacus_version", lambda: Version("1.5.4"))

    with pytest.raises(RuntimeError, match="opacus>=1.6.0"):
        _dp_trainer(
            model=fixture_tiny_model,
            tokenizer=fixture_stub_tokenizer,
            train_dataset=fixture_tiny_training_dataset_with_position_ids,
            tmp_path=tmp_path,
            grad_sample_mode="ghost",
        )


def test_dp_ghost_clipping_training_one_step(
    fixture_tiny_model,
    fixture_stub_tokenizer,
    fixture_tiny_training_dataset_with_position_ids,
    tmp_path,
    monkeypatch,
):
    """Exercises: OpacusDPTrainer Fast/Ghost Gradient Clipping path."""
    monkeypatch.setattr(dp_utils, "_get_opacus_version", lambda: Version("1.6.0"))
    trainer = _dp_trainer(
        model=fixture_tiny_model,
        tokenizer=fixture_stub_tokenizer,
        train_dataset=fixture_tiny_training_dataset_with_position_ids,
        tmp_path=tmp_path,
        grad_sample_mode="ghost",
    )
    trainer.train()
    assert len(trainer.state.log_history) > 0


@pytest.mark.parametrize("grad_sample_mode", ["hooks", "ghost"])
def test_dp_clipping_training_with_gradient_accumulation(
    fixture_tiny_model,
    fixture_stub_tokenizer,
    fixture_tiny_training_dataset_with_position_ids,
    tmp_path,
    monkeypatch,
    grad_sample_mode,
):
    """DP clipping must survive gradient accumulation (the max_physical_batch_size path).

    With ``gradient_accumulation_steps > 1`` the Trainer runs multiple
    micro-batch backward passes per optimizer step. ``DPCallback.on_substep_end``
    signals Opacus to accumulate per-sample-clipped gradients into ``summed_grad``
    and only the final (non-skipped) step adds noise. For ghost clipping this also
    exercises that the wrapper's internal ``optimizer.zero_grad()`` (between its two
    backward passes) preserves ``summed_grad`` across skipped substeps.
    """
    if grad_sample_mode == "ghost":
        monkeypatch.setattr(dp_utils, "_get_opacus_version", lambda: Version("1.6.0"))
    trainer = _dp_trainer(
        model=fixture_tiny_model,
        tokenizer=fixture_stub_tokenizer,
        train_dataset=fixture_tiny_training_dataset_with_position_ids,
        tmp_path=tmp_path,
        grad_sample_mode=grad_sample_mode,
        gradient_accumulation_steps=2,
    )
    trainer.train()

    assert trainer.dp_callback._on_substep_end_was_called
    assert len(trainer.state.log_history) > 0


def _uneven_mask_training_dataset() -> Dataset:
    input_ids = [[1 + (row + position) % 14 for position in range(6)] for row in range(4)]
    labels = []
    for supervised_count, row_ids in enumerate(input_ids, start=1):
        row_labels = [-100] * len(row_ids)
        row_labels[1 : supervised_count + 1] = row_ids[1 : supervised_count + 1]
        labels.append(row_labels)
    return Dataset.from_dict(
        {
            "input_ids": input_ids,
            "attention_mask": [[1] * 6 for _ in input_ids],
            "position_ids": [list(range(6)) for _ in input_ids],
            "labels": labels,
        }
    )


def _per_sample_trainable_gradients(model: torch.nn.Module, dataset: Dataset) -> list[dict[str, torch.Tensor]]:
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    gradients = []
    for example in dataset:
        inputs = {
            name: torch.tensor(example[name]).unsqueeze(0) for name in ("input_ids", "attention_mask", "position_ids")
        }
        labels = torch.tensor(example["labels"]).unsqueeze(0)
        logits = model(**inputs).logits
        shift_labels = torch.nn.functional.pad(labels, (0, 1), value=-100)[..., 1:]
        keep = shift_labels != -100
        loss = torch.nn.functional.cross_entropy(logits[keep], shift_labels[keep])
        values = torch.autograd.grad(loss, [parameter for _, parameter in named_parameters])
        gradients.append(
            {
                name.removeprefix("_module."): value.detach().clone()
                for (name, _), value in zip(named_parameters, values)
            }
        )
    return gradients


def _clipped_sgd_expected_state(
    initial_state: dict[str, torch.Tensor], gradients: list[dict[str, torch.Tensor]], learning_rate: float
) -> tuple[dict[str, torch.Tensor], float]:
    norms = [
        torch.sqrt(torch.stack([gradient.square().sum() for gradient in sample.values()]).sum()) for sample in gradients
    ]
    sorted_norms = sorted(norm.item() for norm in norms)
    clip_norm = (sorted_norms[1] + sorted_norms[2]) / 2
    assert sum(norm.item() > clip_norm for norm in norms) == 2

    expected = {}
    for name, initial in initial_state.items():
        clipped_mean = torch.stack(
            [sample[name] * min(1.0, clip_norm / norm.item()) for sample, norm in zip(gradients, norms, strict=True)]
        ).mean(dim=0)
        expected[name] = initial - learning_rate * clipped_mean
    return expected, clip_norm


@pytest.mark.parametrize(
    ("per_device_batch_size", "gradient_accumulation_steps"),
    [(4, 1), (2, 2)],
)
def test_dp_ghost_clipping_matches_hooks_and_clipped_sgd_oracle(
    fixture_stub_tokenizer,
    tmp_path,
    per_device_batch_size,
    gradient_accumulation_steps,
):
    """Hooks and ghost should match separable clipped DP-SGD with uneven masks."""
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    dataset = _uneven_mask_training_dataset()

    def make_model() -> torch.nn.Module:
        torch.manual_seed(2026)
        model = _tiny_lora_model(LlamaForCausalLM(config))
        model.enable_input_require_grads()
        return model

    oracle_model = make_model()
    initial_state = _trainable_parameter_state(oracle_model)
    gradients = _per_sample_trainable_gradients(oracle_model, dataset)
    learning_rate = 0.1
    expected_state, clip_norm = _clipped_sgd_expected_state(initial_state, gradients, learning_rate)

    states = {}
    for grad_sample_mode in ("hooks", "ghost"):
        trainer = _dp_trainer(
            model=make_model(),
            tokenizer=fixture_stub_tokenizer,
            train_dataset=dataset,
            tmp_path=tmp_path / f"{grad_sample_mode}-{gradient_accumulation_steps}",
            grad_sample_mode=grad_sample_mode,
            privacy_args=_privacy_args(noise_multiplier=0.0, per_sample_max_grad_norm=clip_norm),
            secure_mode=False,
            true_dataset_size=4,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            lr_scheduler_type="constant",
            optim="sgd",
            weight_decay=0.0,
            seed=1729,
            data_seed=1729,
        )
        trainer.train()
        states[grad_sample_mode] = _trainable_parameter_state(trainer.model)

    for name, expected in expected_state.items():
        torch.testing.assert_close(states["hooks"][name], expected, rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(states["ghost"][name], expected, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("grad_sample_mode", ["hooks", "ghost"])
def test_dp_clipping_save_writes_peft_adapter(
    fixture_tiny_model,
    fixture_stub_tokenizer,
    fixture_tiny_training_dataset_with_position_ids,
    tmp_path,
    monkeypatch,
    grad_sample_mode,
):
    """DP clipping wrappers should save PEFT adapters, not wrapper weights."""
    if grad_sample_mode == "ghost":
        monkeypatch.setattr(dp_utils, "_get_opacus_version", lambda: Version("1.6.0"))
    trainer = _dp_trainer(
        model=_tiny_lora_model(fixture_tiny_model),
        tokenizer=fixture_stub_tokenizer,
        train_dataset=fixture_tiny_training_dataset_with_position_ids,
        tmp_path=tmp_path,
        grad_sample_mode=grad_sample_mode,
    )

    output_dir = tmp_path / "adapter"
    trainer._save(str(output_dir))

    assert (output_dir / "adapter_model.safetensors").exists()
    assert not (output_dir / "model.safetensors").exists()


def test_training_example_assembler(fixture_iris_df, fixture_stub_tokenizer, tmp_path):
    """Exercises: NSS data preparation pipeline (TrainingExampleAssembler)."""
    from nemo_safe_synthesizer.config import SafeSynthesizerParameters

    config = SafeSynthesizerParameters.from_params(
        num_input_records_to_sample=10,
    )
    hf_dataset = Dataset.from_pandas(fixture_iris_df, preserve_index=False)

    # Build a minimal picklable metadata stub (MagicMock can't be pickled by datasets).
    stub_metadata = _StubModelMetadata()

    assembler = TrainingExampleAssembler.from_data(
        dataset=hf_dataset,
        tokenizer=fixture_stub_tokenizer,
        metadata=stub_metadata,  # ty: ignore[invalid-argument-type] -- deliberate test stub
        config=config,
        seed=42,
        cache_file_path=str(tmp_path / "cache"),
    )
    training_examples = assembler.assemble_training_examples()

    assert training_examples is not None
    assert assembler.num_records_train > 0
