# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU training smoke tests -- Trainer, LoRA, DP, Assembler.

All tests run on CPU with max_steps=1. The point is catching dep breakage
(torch + transformers + peft + opacus) and exercising the NSS data pipeline.
"""

from dataclasses import dataclass, field

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

from nemo_safe_synthesizer.data_processing.assembler import TrainingExampleAssembler
from nemo_safe_synthesizer.defaults import DEFAULT_INSTRUCTION, PROMPT_TEMPLATE
from nemo_safe_synthesizer.privacy.dp_transformers.dp_utils import (
    DataCollatorForPrivateTokenClassification,
    OpacusDPTrainer,
)
from nemo_safe_synthesizer.privacy.dp_transformers.privacy_args import PrivacyArguments


def _cpu_training_args(tmp_path, **overrides):
    """Build TrainingArguments for CPU smoke tests with sensible defaults."""
    defaults = dict(
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


def test_hf_trainer_one_step(tiny_model, stub_tokenizer, tiny_training_dataset, tmp_path):
    """Exercises: transformers.Trainer forward + backward pass."""
    trainer = Trainer(
        model=tiny_model,
        args=_cpu_training_args(tmp_path),
        train_dataset=tiny_training_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=stub_tokenizer),
    )
    trainer.train()
    assert len(trainer.state.log_history) > 0
    last_log = trainer.state.log_history[-1]
    assert "loss" in last_log or "train_loss" in last_log


def test_lora_training_one_step(tiny_model, stub_tokenizer, tiny_training_dataset, tmp_path):
    """Exercises: peft.get_peft_model + LoraConfig + Trainer."""
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(tiny_model, lora_config)
    model.enable_input_require_grads()
    trainer = Trainer(
        model=model,
        args=_cpu_training_args(tmp_path),
        train_dataset=tiny_training_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=stub_tokenizer),
    )
    trainer.train()
    assert len(trainer.state.log_history) > 0
    last_log = trainer.state.log_history[-1]
    assert "loss" in last_log or "train_loss" in last_log


def test_dp_training_one_step(tiny_model, stub_tokenizer, tiny_training_dataset_with_position_ids, tmp_path):
    """Exercises: OpacusDPTrainer + PrivacyArguments + DataCollatorForPrivateTokenClassification."""
    privacy_args = PrivacyArguments(
        target_epsilon=100.0,
        target_delta=1e-5,
        per_sample_max_grad_norm=1.0,
    )
    args = _cpu_training_args(tmp_path, remove_unused_columns=False, max_grad_norm=0.0)
    data_collator = DataCollatorForPrivateTokenClassification(tokenizer=stub_tokenizer)
    trainer = OpacusDPTrainer(
        model=tiny_model,
        args=args,
        train_dataset=tiny_training_dataset_with_position_ids,
        data_collator=data_collator,
        privacy_args=privacy_args,
        true_dataset_size=8,
        data_fraction=1.0,
    )
    trainer.train()
    assert len(trainer.state.log_history) > 0


def test_training_example_assembler(iris_df, stub_tokenizer, tmp_path):
    """Exercises: NSS data preparation pipeline (TrainingExampleAssembler)."""
    from nemo_safe_synthesizer.config import SafeSynthesizerParameters

    config = SafeSynthesizerParameters.from_params(
        num_input_records_to_sample=10,
        use_unsloth=False,
    )
    hf_dataset = Dataset.from_pandas(iris_df, preserve_index=False)

    # Build a minimal picklable metadata stub (MagicMock can't be pickled by datasets).
    stub_metadata = _StubModelMetadata()

    assembler = TrainingExampleAssembler.from_data(
        dataset=hf_dataset,
        tokenizer=stub_tokenizer,
        metadata=stub_metadata,
        config=config,
        seed=42,
        cache_file_path=str(tmp_path / "cache"),
    )
    training_examples = assembler.assemble_training_examples()

    assert training_examples is not None
    assert assembler.num_records_train > 0
