<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuration Reference

Parameter tables for all synthesis configuration sections. For how to use
each stage with examples, see [Running Safe Synthesizer](running.md). For
environment variables, see [Environment Variables](environment.md).

---

## Training

Safe Synthesizer fine-tunes a pretrained language model on your tabular data
using LoRA (Low-Rank Adaptation). See
[`TrainingHyperparams`][nemo_safe_synthesizer.config.training.TrainingHyperparams]
for the full field list.

| Field | Default | Description |
|-------|---------|-------------|
| `training.learning_rate` | `0.0005` | Initial learning rate for the AdamW optimizer |
| `training.batch_size` | `1` | Per-device batch size |
| `training.gradient_accumulation_steps` | `8` | Steps to accumulate before a backward pass; effective batch size = `batch_size` x this value |
| `training.num_input_records_to_sample` | `"auto"` | Records the model sees during training -- proxy for training time (`"auto"` or int) |
| `training.lora_r` | `32` | LoRA rank; lower values produce fewer trainable parameters |
| `training.lora_alpha_over_r` | `1.0` | LoRA scaling ratio (alpha / rank) |
| `training.pretrained_model` | `"HuggingFaceTB/SmolLM3-3B"` | HuggingFace model ID or local path |
| `training.use_unsloth` | `"auto"` | Use the Unsloth backend. Set to `false` or leave at `"auto"` when using DP (`"auto"` resolves to `false` when DP is enabled) |
| `training.quantize_model` | `false` | Enable quantization to reduce VRAM usage |
| `training.quantization_bits` | `8` | Bit width (4 or 8) when `training.quantize_model` is `true` |
| `training.attn_implementation` | `"kernels-community/vllm-flash-attn3"` | Attention backend for model loading |
| `training.rope_scaling_factor` | `"auto"` | Scale the base model's context window via RoPE (`"auto"` or int) |
| `training.validation_ratio` | `0.0` | Fraction of training data held out for validation loss monitoring |

!!! note "validation_ratio vs holdout"
    `training.validation_ratio` splits the training data to monitor
    validation loss during fine-tuning. `data.holdout` splits the full
    dataset to create a test set used by the evaluation stage. They serve
    different purposes and are applied at different stages.
    `training.validation_ratio` is primarily for internal development use;
    leave it at `0.0` unless you have a specific reason to monitor
    validation loss during fine-tuning.

Safe Synthesizer has explicit support (prompt templates, RoPE scaling,
tokenizer handling) for these model families. Models outside this list
will raise a `ValueError` at startup.

| Family | HuggingFace ID |
|--------|----------------|
| SmolLM3 (default) | `HuggingFaceTB/SmolLM3-3B` |
| SmolLM2 | `HuggingFaceTB/SmolLM2-135M` |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Mistral | `mistralai/Mistral-7B-v0.1` |
| Llama 3.2 | detection requires `Llama32` in the path; standard `meta-llama/Llama-3.2-*` IDs do not match |
| Qwen | `Qwen/Qwen2.5-7B` |
| Nemotron | `nvidia/Nemotron-Mini-4B-Instruct` |
| Granite | `ibm-granite/granite-3.3-2b-instruct` |

Within each family, any size variant on HuggingFace Hub should work,
though not all have been tested -- larger models generally produce
better results but require more VRAM and training time.

---

## Generation

The generation stage controls how the fine-tuned model produces synthetic
records. See
[`GenerateParameters`][nemo_safe_synthesizer.config.generate.GenerateParameters]
for the full API reference.

| Field | Default | Description |
|-------|---------|-------------|
| `generation.num_records` | `1000` | Number of synthetic records to generate |
| `generation.temperature` | `0.9` | Sampling temperature; lower values produce more conservative output |
| `generation.top_p` | `1.0` | Nucleus sampling probability |
| `generation.repetition_penalty` | `1.0` | Penalty for repeated tokens; increase slightly if generation produces repetitive output |
| `generation.patience` | `3` | Consecutive bad batches before stopping |
| `generation.invalid_fraction_threshold` | `0.8` | Invalid record fraction that triggers the patience counter |
| `generation.use_structured_generation` | `false` | Enable structured output to constrain record format (typically at the cost of reducing the quality of generated records and increasing generation time; use when the pipeline struggles to produce valid records) |
| `generation.structured_generation_backend` | `"auto"` | vLLM guided-decoding backend |
| `generation.structured_generation_schema_method` | `"regex"` | Schema method (`"regex"` or `"json_schema"`) |
| `generation.structured_generation_use_single_sequence` | `false` | Match exactly one sequence when `max_sequences_per_example` is 1 |
| `generation.enforce_timeseries_fidelity` | `false` | Enforce time series order, intervals, and timestamps |
| `generation.attention_backend` | `"auto"` | vLLM attention backend |

Advanced group-by validation knobs live under `generation.validation`:
`group_by_accept_no_delineator`, `group_by_ignore_invalid_records`,
`group_by_fix_non_unique_value`, `group_by_fix_unordered_records`. All
default to `false`. See
[`GenerateParameters`][nemo_safe_synthesizer.config.generate.GenerateParameters]
for details.

---

## PII Replacement

PII replacement detects and replaces personally identifiable information in
your dataset before synthesis. It is on by default (`enable_replace_pii: true`).
The `replace_pii` block is only needed when customizing entity types or
classification via the SDK.

Key config structure:

- `replace_pii.globals.classify`: column classification -- which columns contain PII
- `replace_pii.globals.classify.enable_classify`: enable LLM-based column classification
- `replace_pii.globals.classify.entities`: entity types to classify (e.g. `["email", "phone_number", "ssn"]`)
- `replace_pii.globals.ner`: row-level NER via GLiNER
- `replace_pii.globals.ner.ner_threshold`: confidence threshold (default `0.3`)

See [`PiiReplacerConfig`][nemo_safe_synthesizer.config.replace_pii.PiiReplacerConfig]
for the full schema.

---

## Differential Privacy

Differential privacy (DP) provides a formal bound on what an adversary can
learn about any individual record. Safe Synthesizer implements DP-SGD via
Opacus.

| Field | Default | Description |
|-------|---------|-------------|
| `privacy.dp_enabled` | `false` | Enable DP-SGD training |
| `privacy.epsilon` | `8.0` | Privacy budget -- lower values give stronger privacy (4.0--12.0 typical) |
| `privacy.delta` | `"auto"` | Privacy failure probability (`"auto"` or float) |
| `privacy.per_sample_max_grad_norm` | `1.0` | Max L2 norm for per-sample gradients |

Compatibility constraints:

- Set `training.use_unsloth` to `false` or leave it at `"auto"` -- `"auto"` resolves to `false` when DP is enabled
- `data.max_sequences_per_example` must be `1` (or `"auto"`, which resolves to `1` when DP is enabled)
- Gradient checkpointing is disabled (incompatible with Opacus)

See [`DifferentialPrivacyHyperparams`][nemo_safe_synthesizer.config.differential_privacy.DifferentialPrivacyHyperparams]
for the full field list. For DP error diagnostics, see
[Evaluating Output Data -- Common DP Errors](evaluating-data.md#common-dp-errors).

---

## Data

| Field | Default | Description |
|-------|---------|-------------|
| `data.holdout` | `0.05` | Fraction (0--1) or absolute count (>1) for the holdout test set for evaluation |
| `data.max_holdout` | `2000` | Upper cap on holdout size |
| `data.random_state` | `null` | Random seed -- auto-generated if `null`; set an explicit integer for reproducible splits |
| `data.group_training_examples_by` | `null` | Column to group records by |
| `data.order_training_examples_by` | `null` | Column to order within groups (requires group_by) |
| `data.max_sequences_per_example` | `"auto"` | Max sequences per example (`1` for DP, defaults to `10` otherwise) |

See [`DataParameters`][nemo_safe_synthesizer.config.data.DataParameters]
for the full field list.

---

## Time Series

!!! warning "Experimental"
    Time series synthesis is an experimental feature. APIs and behavior may
    change between releases.

| Field | Default | Description |
|-------|---------|-------------|
| `time_series.is_timeseries` | `false` | Enable time series mode |
| `time_series.timestamp_column` | `null` | Timestamp column name |
| `time_series.timestamp_interval_seconds` | `null` | Fixed interval between timestamps |
| `time_series.timestamp_format` | `null` | strftime format or `"elapsed_seconds"` |
| `time_series.start_timestamp` | `null` | Override start timestamp for all groups (inferred from data if `null`) |
| `time_series.stop_timestamp` | `null` | Override stop timestamp for all groups (inferred from data if `null`) |

See [`TimeSeriesParameters`][nemo_safe_synthesizer.config.time_series.TimeSeriesParameters]
for the full schema. For detailed descriptions and constraints, see the
[Time Series README](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/src/nemo_safe_synthesizer/TIMESERIES_README.md).

---

## Evaluation

| Field | Default | Description |
|-------|---------|-------------|
| `evaluation.enabled` | `true` | Master switch for evaluation |
| `evaluation.mia_enabled` | `true` | Membership inference attack |
| `evaluation.aia_enabled` | `true` | Attribute inference attack |
| `evaluation.pii_replay_enabled` | `true` | PII replay detection |
| `evaluation.sqs_report_columns` | `250` | Max columns in SQS report |
| `evaluation.sqs_report_rows` | `5000` | Max rows in SQS report |
| `evaluation.quasi_identifier_count` | `3` | Number of quasi-identifiers sampled for AIA (auto-reduced for small datasets) |
| `evaluation.mandatory_columns` | `null` | Number of mandatory columns that must be used in evaluation |

See [`EvaluationParameters`][nemo_safe_synthesizer.config.evaluate.EvaluationParameters]
for the full API reference.

---

## Validate and Modify Configuration

### `config validate`

Check your config for errors and display the merged parameters:

```bash
safe-synthesizer config validate --config config.yaml
```

Fields set to `"auto"` remain as `"auto"` in the output -- auto-resolution
happens at runtime during `process_data()`, not at validation time. To see
resolved values, check `safe-synthesizer-config.json` in the run directory
after a pipeline run.

!!! note "Override parsing limitation"
    `config validate` accepts synthesis parameter overrides (`--training__learning_rate`,
    etc.) but due to a known parsing issue, overrides are not applied to the validated
    output. Use `config modify` to test how overrides merge with a config file.

### `config modify`

Modify a configuration and optionally save the result:

```bash
safe-synthesizer config modify --config config.yaml --training__learning_rate 0.001 --output modified.yaml
```

| Option | Description |
|--------|-------------|
| `--config` | Path to YAML config file (optional -- omit to build from overrides only) |
| `--output` | Path to write modified YAML config (prints JSON to stdout if omitted) |

### `config create`

Create a new configuration from defaults:

```bash
safe-synthesizer config create --output config.yaml
safe-synthesizer config create --training__pretrained_model "HuggingFaceTB/SmolLM3-3B" --output config.yaml
```

| Option | Description |
|--------|-------------|
| `--output` / `-o` | Path to write YAML config (prints JSON to stdout if omitted) |

### CLI Override Syntax

Use double underscores to address nested fields:

```bash
safe-synthesizer run --config config.yaml --url data.csv \
  --training__learning_rate 0.001 \
  --data__holdout 0.1 \
  --generation__num_records 5000
```

!!! note "Override precedence"
    CLI overrides > dataset registry overrides > YAML config file > model
    defaults. Parameters that accept `"auto"` cannot be set to `"auto"` via
    CLI flags -- omit the flag to use the default, or set it in YAML.

---

## Configuration Sections

| YAML Key | SDK Method | API Reference |
|----------|-----------|---------------|
| `data` | `with_data()` | [`DataParameters`][nemo_safe_synthesizer.config.data.DataParameters] |
| `training` | `with_train()` | [`TrainingHyperparams`][nemo_safe_synthesizer.config.training.TrainingHyperparams] |
| `generation` | `with_generate()` | [`GenerateParameters`][nemo_safe_synthesizer.config.generate.GenerateParameters] |
| `evaluation` | `with_evaluate()` | [`EvaluationParameters`][nemo_safe_synthesizer.config.evaluate.EvaluationParameters] |
| `replace_pii` | `with_replace_pii()` | [`PiiReplacerConfig`][nemo_safe_synthesizer.config.replace_pii.PiiReplacerConfig] |
| `privacy` | `with_differential_privacy()` | [`DifferentialPrivacyHyperparams`][nemo_safe_synthesizer.config.differential_privacy.DifferentialPrivacyHyperparams] |
| `time_series` | `with_time_series()` | [`TimeSeriesParameters`][nemo_safe_synthesizer.config.time_series.TimeSeriesParameters] |
