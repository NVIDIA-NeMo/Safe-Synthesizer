<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# llm

Model metadata, prompt templates, RoPE scaling, memory management, and quantization config for LLM loading and training.

## Purpose

- ModelMetadata — Stores model-family-specific info: prompt formats, BOS/EOS tokens, RoPE scaling, base max seq length
- LLMPromptConfig — Template string and token settings; built via `from_tokenizer()`
- RopeScaling — Context-window extension (linear, dynamic, yarn, llama3); capped by `MAX_ROPE_SCALING_FACTOR`
- Memory — `cleanup_memory()` and `get_max_vram()` for GPU lifecycle
- Quantization — `get_quantization_config(4|8)` returns `BitsAndBytesConfig` (nf4/nf8, bfloat16 compute)

## Model Subclass Pattern

`ModelMetadata` base → 8 subclasses: TinyLlama, Qwen, Llama32, Mistral, SmolLM2, SmolLM3, Granite, Nemotron. Factory `from_str_or_path(model_name_or_path)` uses case-insensitive substring matching on class name (e.g. "Llama32" in path). Raises `ValueError` if no subclass matches; no fallback to base. `_resolve_model_class(model_name_or_path)` returns the matching subclass type without instantiation; used by `AutoConfigResolver` for learning rate resolution.

Each subclass inherits `default_learning_rate: ClassVar[float] = 0.0005` from the base. Mistral overrides to `0.0001`. When `training.learning_rate` is `"auto"`, `AutoConfigResolver` calls `_resolve_model_class()` and reads this class variable.

## Prompt Config

Each model sets its own template and BOS/EOS via `LLMPromptConfig.from_tokenizer()`. Examples:

- Llama32 — `user\n {instruction} {schema} \n assistant\n{prefill}`, `<|im_start|>` (151644), add_bos=False, add_eos=False
- Mistral — `[INST] {instruction} \n\n {schema} [/INST]{prefill}`
- SmolLM3 — `... <|im_end|> \n <|im_start|>assistant\n{prefill}`, explicit bos/eos
- Qwen/Granite — `user\n {instruction} {schema} \n assistant\n{prefill}` variants

## Rope Scaling

- RopeScaling — `rope_type` (linear, dynamic, default, yarn, llama3), `factor`, `theta`
- GLOBAL_MAX_SEQ_LENGTH = 2048 * 6 — hard cap; `get_base_max_seq_length()` clamps `max_position_embeddings`
- resolve_rope_scaling_factor(factor, autoconfig) — accepts `None`, `int`, `float`, `dict`, `RopeScaling`. Int/float require `autoconfig` to read theta/rope_type. Returns `None` for factor 1.0
- Models that ignore rope scaling — Mistral, SmolLM2, SmolLM3 (log warning and pass `rope_scaling=None`)
- Module docstring notes this logic is problematic; prefer future `context_window_size`-style API

## Memory Management

- cleanup_memory() — `gc.collect()` + `torch.cuda.empty_cache()` under `torch.no_grad()`
- get_max_vram(memory_fraction, as_string, as_fraction) — leaves 2GB buffer (`free - 2*1024**3`), applies `memory_fraction` (default 0.8), returns per-device dict `{device_id: GiB_str|float}`

## Gotchas

- `rope_scaling` defaults to `None`; `populate_derived_fields` resolves any input (float, int, dict, RopeScaling, None) to `RopeScaling | None`
- trust_remote_code — `trust_remote_code_for_model()` returns True only for `nvidia/` and `gretel/` prefixes
- initial_prefill — can be `dict[str, str]` (grouped) or `str` (single column)
- rope_parameters_location — `"autoconfig"` vs `"automodel"`; all current subclasses use `"autoconfig"`
- from_str_or_path — raises on unknown model; no base `ModelMetadata` fallback

## Extension Recipe

1. Create subclass of `ModelMetadata`
2. Override `__init__()`: load tokenizer + AutoConfig, call `super().__init__()` with `prompt_config=LLMPromptConfig.from_tokenizer(...)`, `rope_scaling`, `rope_parameters_location`
3. Override `default_learning_rate` class variable if the model family needs a non-default learning rate
4. Add class to `classes` tuple in `_resolve_model_class()` and `from_str_or_path()` (match order: substring check uses class name)

## Read First

- `metadata.py` — core (567 lines): ModelMetadata, RopeScaling, LLMPromptConfig, subclasses, `from_str_or_path`, `resolve_rope_scaling_factor`
- `utils.py` — helpers (239 lines): `cleanup_memory`, `get_max_vram`, `trust_remote_code_for_model`, `get_quantization_config`, `get_device_map`, `add_bos_eos_tokens_to_tokenizer`
