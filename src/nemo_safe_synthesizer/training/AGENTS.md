<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# training

Fine-tuning LLMs with HuggingFace Trainer, optional differential privacy (Opacus), and optional Unsloth acceleration.

## Purpose

Train LoRA adapters on tabular or time-series data for synthetic generation. Supports full-precision and quantized PEFT (LoRA), DP-SGD via Opacus, and Unsloth-optimized loading/training.

## Backend Hierarchy

- **TrainingBackend** (ABC) — defines `prepare_training_data`, `prepare_config`, `prepare_params`, `maybe_quantize`, `load_model`, `train`, `save_model`. Subclasshook checks for these methods.
- **HuggingFaceBackend** — uses `AutoModelForCausalLM`, HuggingFace `Trainer`, LoRA via PEFT.
- **UnslothTrainer** — extends `HuggingFaceBackend`. Uses `FastLanguageModel` for model loading and `FastLanguageModel.get_peft_model` for PEFT. Requires CUDA; incompatible with DP.

## FIXED_RUNTIME_TRAINING_ARGS

Training duration is controlled by `num_input_records_to_sample`, not epochs. These args are hard-coded:

- `num_train_epochs=1`, `save_strategy=EPOCH`, `logging_steps=1`, `optim="paged_adamw_32bit"`, `bf16=True`, `group_by_length=False`, `ddp_find_unused_parameters=False`
- `per_device_eval_batch_size=2`, `disable_tqdm=True`

`data_fraction = num_input_records_to_sample / assembler.num_records_train` drives how many examples are assembled per run.

## DP Integration

Enabled via `privacy.dp_enabled`. `_configure_dp_training()`:

- Requires `max_sequences_per_example=1` (enforced by config validator when DP is on; Opacus expects one sample per microbatch for gradient clipping).
- Sets `max_grad_norm=0.0` — Opacus handles per-sample clipping instead.
- Uses `DataCollatorForPrivateTokenClassification` and `OpacusDPTrainer`.
- Requires `data_fraction` and `true_dataset_size` (raises `ParameterError` if missing).
- Key `PrivacyArguments`: `target_epsilon`, `target_delta`, `per_sample_max_grad_norm`.
- Disables gradient checkpointing (not compatible with Opacus optimizer wrapping).

## Quantization

- **4-bit** — `nf4`, double quantization, bfloat16 compute (BitsAndBytesConfig).
- **8-bit** — `nf8`, double quantization, bfloat16 compute.
- **LoftQ** — `peft_implementation == "loftq"` uses `LoftQConfig(loftq_bits=...)` for alternative LoRA init.
- If quantized: `prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)` before applying LoRA. Otherwise: `gradient_checkpointing_enable()`, `enable_input_require_grads()`, `use_cache=False`.

## Callbacks

- **InferenceEvalCallback** — during eval, generates records and validates against schema; early-stops on invalid fraction (`invalid_fraction_threshold` + `patience`).
- **ProgressBarCallback** — tqdm progress bar (used when logging level is not INFO/DEBUG).
- **SafeSynthesizerWorkerCallback** — structured logging with progress, epoch, step, loss; triggers logs at least every `log_interval` seconds and at epoch end.

Trainer removes `PrinterCallback`; adds SafeSynthesizerWorkerCallback or ProgressBarCallback based on log level. `InferenceEvalCallback` added only when `generation_eval=True`.

## Gotchas

- **`rope_parameters_location`** — `"autoconfig"` sets rope scaling on `autoconfig.rope_scaling`; `"automodel"` passes it via `framework_params["rope_scaling"]`. Wrong choice can break rope scaling.
- **`_trust_remote_code_for_model()`** — returns `True` only for `nvidia/` models. Used for `AutoConfig.from_pretrained(..., trust_remote_code=...)`.
- **`_apply_eval_dataset_overrides()`** — when `eval_dataset` is provided, overrides `eval_steps`, `eval_strategy="steps"`, `do_eval=True`, `include_for_metrics`, `eval_accumulation_steps`.
- **Timeseries preprocessing** — `process_timeseries_data()` adds `PSEUDO_GROUP_COLUMN` (`__nss_sequence_id`) when no group column is specified; treats the whole dataset as one sequence.
- **Unsloth** — uses `model_name` instead of `pretrained_model_name_or_path`; uses `max_seq_length` instead of `max_position_embeddings`. Disables `SUPPORTS_LLAMA32` to avoid HF Hub requests.

## Extension Points

- **New backend** — subclass `TrainingBackend`, implement the abstract methods.
- **Custom callbacks** — extend `TrainerCallback`, add via `callbacks` constructor arg; they are appended after the default callbacks.
- **Custom data collator** — pass `data_collator` in training args; for DP use `DataCollatorForPrivateTokenClassification` or a compatible variant.

## Read First

- `backend.py` — TrainingBackend ABC, NSSTrainerResult, _trust_remote_code_for_model
- `huggingface_backend.py` — HuggingFaceBackend, FIXED_RUNTIME_TRAINING_ARGS, DP config, LoRA/quantization, _apply_rope_scaling
- `unsloth_backend.py` — UnslothTrainer, _update_for_unsloth, maybe_quantize (unsloth path)
- `callbacks.py` — InferenceEvalCallback, ProgressBarCallback, SafeSynthesizerWorkerCallback
- `timeseries_preprocessing.py` — process_timeseries_data, PSEUDO_GROUP_COLUMN
