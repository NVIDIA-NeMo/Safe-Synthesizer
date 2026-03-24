<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuration Reference

Parameter tables for all NeMo Safe Synthesizer configuration sections. For how to use
each stage with examples, see [Running Safe Synthesizer](running.md). For
environment variables, see [Environment Variables](environment.md).

---

## Configuration Precedence

Exactly what avenues of configuration are available, and thus how precedence is resolved, depends on how you run the pipeline. Settings are resolved in this order, from highest (first) to lowest priority (last):

- CLI: `CLI flags` > `dataset registry overrides` > `YAML config file` > `model defaults`
- SDK: `SDK builder calls` > `YAML config file` > `model defaults`

Each layer only overrides what it explicitly sets -- everything else falls
through to the next layer.

### Examples

Start from model defaults, override one field via CLI:

```bash
safe-synthesizer run --data-source data.csv --generation__num_records 2000
```

Use a YAML base for most settings, tune one field per run without editing the file:

```bash
safe-synthesizer run --config config.yaml --data-source data.csv \
  --training__learning_rate 0.001
```

Load a YAML base from Python, override a section with the builder:

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.config import SafeSynthesizerParameters

config = SafeSynthesizerParameters.from_yaml("config.yaml")
synthesizer = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .with_generate(num_records=2000, temperature=0.8)  # overrides config.yaml values
)
synthesizer.run()
```

!!! note
    Parameters that accept `"auto"` cannot be set to `"auto"` via CLI flags --
    omit the flag to use the default, or set it in YAML.

See [Using YAML Config Files with the CLI and SDK](running.md#using-yaml-config-files)
for more detail on combining config files with runtime overrides.

---

## Training

NeMo Safe Synthesizer fine-tunes a pretrained language model on your tabular data
using LoRA (Low-Rank Adaptation). See
[`TrainingHyperparams`][nemo_safe_synthesizer.config.training.TrainingHyperparams]
for the full field list.

| Field | Default | Description | Guidance |
|-------|---------|-------------|----------|
| `training.learning_rate` | `"auto"` | Initial learning rate for the `AdamW` optimizer. `"auto"` selects a model-specific default (Mistral: 1e-4, others: 5e-4) | Leave at `"auto"` for most cases; override with a float in (0, 1) to tune manually |
| `training.batch_size` | `1` | Per-device batch size | Leave at 1; increase `gradient_accumulation_steps` for a larger effective batch |
| `training.gradient_accumulation_steps` | `8` | Steps to accumulate before a backward pass; effective batch size = `batch_size` x this value | 8--32 typical |
| `training.num_input_records_to_sample` | `"auto"` | Records the model sees during training -- proxy for training time (`"auto"` or int) | First knob to increase if quality is low |
| `training.lora_r` | `32` | LoRA rank; lower values produce fewer trainable parameters | 16--64 typical; 32 is a reasonable default |
| `training.lora_alpha_over_r` | `1.0` | LoRA scaling ratio (alpha / rank) | Leave at 1.0 |
| `training.pretrained_model` | `"HuggingFaceTB/SmolLM3-3B"` | HuggingFace model ID or local path | See supported families below; `TinyLlama/TinyLlama-1.1B-Chat-v1.0` for fast CPU/low-VRAM iteration |
| `training.use_unsloth` | `"auto"` | Use the Unsloth backend. Set to `false` or leave at `"auto"` when using DP (`"auto"` resolves to `false` when DP is enabled) | Leave at `"auto"` |
| `training.quantize_model` | `false` | Enable quantization to reduce VRAM usage | Enable if VRAM is limited; 8-bit has lower quality impact than 4-bit |
| `training.quantization_bits` | `8` | Bit width (4 or 8) when `training.quantize_model` is `true` | Prefer 8 over 4 for quality |
| `training.attn_implementation` | `"kernels-community/vllm-flash-attn3"` | Attention backend for model loading | Leave at default |
| `training.rope_scaling_factor` | `"auto"` | Scale the base model's context window via RoPE (`"auto"` or int) | Leave at `"auto"` |
| `training.validation_ratio` | `0.0` | Fraction of training data held out for validation loss monitoring | Leave at 0.0 unless you specifically want to monitor validation loss |
| `training.max_vram_fraction` | `0.8` | Fraction of total GPU VRAM to allocate for training. Must be in [0, 1] | Lower if other GPU consumers are active on the same device |

!!! note "validation_ratio vs holdout"
    `training.validation_ratio` splits the training data to monitor
    validation loss during fine-tuning. `data.holdout` splits the full
    dataset to create a test set used by the evaluation stage. They serve
    different purposes and are applied at different stages.

Safe Synthesizer has explicit support (prompt templates, RoPE scaling,
tokenizer handling) for the model families listed below. Models outside this
list will raise a `ValueError` at startup.

We have extensively tested the following models for synthetic data use in NSS, and encourage you to start with `SmolLM3-3B` (the default).


| Family | HuggingFace ID |
|--------|----------------|
| SmolLM3 (default) | `HuggingFaceTB/SmolLM3-3B` |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |

Benchmarking data for additional models will be added as they are
validated. To understand the trade-offs with model selection, see [Training](running.md#training).

Models configured via `training.pretrained_model` are downloaded from Hugging Face Hub. See [Pre-Caching Models](environment.md#pre-caching-models) for details.

!!! warning "Security Note: Pretrained models from Hugging Face Hub"
    Downloading pretrained models from Hugging Face Hub (or any public source) can expose your environment to significant risks, including arbitrary code execution (ACE) or remote code execution (RCE) vulnerabilities. Only use models you have reviewed yourself or from organizations and authors you explicitly trust. Malicious or modified models may contain embedded code, backdoors, or privacy-leaking mechanisms.

---

## Generation

The generation stage controls how the fine-tuned model produces synthetic
records. See
[`GenerateParameters`][nemo_safe_synthesizer.config.generate.GenerateParameters]
for the full API reference.

| Field | Default | Description | Guidance |
|-------|---------|-------------|----------|
| `generation.num_records` | `1000` | Number of synthetic records to generate | Match or exceed input dataset size for best quality |
| `generation.temperature` | `0.9` | Sampling temperature; lower values produce more predictable, less varied output | 0.7--1.1 typical; lower if output is noisy, higher if too repetitive |
| `generation.top_p` | `1.0` | Nucleus sampling probability | Leave at 1.0; lower (e.g. 0.9) to reduce tail tokens |
| `generation.repetition_penalty` | `1.0` | Penalty for repeated tokens; increase slightly if generation produces repetitive output | 1.0--1.15 typical; start at 1.05 if repetition is a problem |
| `generation.patience` | `3` | Consecutive bad batches before stopping | Leave at default |
| `generation.invalid_fraction_threshold` | `0.8` | Invalid record fraction that triggers the patience counter | Leave at default |
| `generation.use_structured_generation` | `false` | Enable structured output to constrain record format (typically at the cost of reducing the quality of generated records and increasing generation time; use when the pipeline struggles to produce valid records) | Leave off unless the pipeline cannot produce valid records |
| `generation.structured_generation_backend` | `"auto"` | vLLM guided-decoding backend | Leave at `"auto"` |
| `generation.structured_generation_schema_method` | `"regex"` | Schema method (`"regex"` or `"json_schema"`) | Leave at `"regex"` |
| `generation.structured_generation_use_single_sequence` | `false` | Match exactly one sequence when `max_sequences_per_example` is 1 | Leave at default |
| `generation.enforce_timeseries_fidelity` | `false` | Enforce time series order, intervals, and timestamps | Enable for time series data |
| `generation.attention_backend` | `"auto"` | vLLM attention backend | Leave at `"auto"` |

Advanced group-by validation knobs live under `generation.validation`:

| Knob | Default | Effect |
|------|---------|--------|
| `group_by_accept_no_delineator` | `false` | Treat raw JSONL without BOS/EOS markers as a single group instead of rejecting |
| `group_by_ignore_invalid_records` | `false` | Drop invalid records from a group and keep the rest, rather than discarding the whole group |
| `group_by_fix_non_unique_value` | `false` | Normalize the group-by column to the first record's value when records disagree |
| `group_by_fix_unordered_records` | `false` | Re-sort records instead of rejecting out-of-order groups |

See [Example Generation -- Validation](../developer-guide/example-generation.md#grouped-generation-validation-knobs)
for guidance on when to enable each knob, and
[`GenerateParameters`][nemo_safe_synthesizer.config.generate.GenerateParameters]
for the full API reference.

---

## Replacing PII

PII replacement detects and replaces personally identifiable information (PII) in
your dataset before synthesis. It is on by default -- set `replace_pii: null`
in YAML (or use `--no-replace-pii` on the CLI) to disable it.
The `replace_pii` block is only needed when customizing entity types or
classification via the SDK.

Key config parameters:

| Field | Default | Description | Guidance |
|-------|---------|-------------|----------|
| `replace_pii.globals.classify.enable_classify` | `true` | Enable LLM-based PII column classification | When using the CLI, set `NSS_INFERENCE_KEY` (and optionally `NSS_INFERENCE_ENDPOINT`); set to `false` if no LLM endpoint is available |
| `replace_pii.globals.classify.entities` | (see default list) | Entity types used for LLM-based column classification. Defaults to 15 types covering names, addresses, phone numbers, emails, SSN, national/tax IDs, and credit/debit cards -- see [PII Replacement](../product-overview/pii_replacement.md) and [`PiiReplacerConfig`][nemo_safe_synthesizer.config.replace_pii.PiiReplacerConfig] | Override to add or remove entity types from classification |
| `replace_pii.globals.ner.ner_threshold` | `0.3` | GLiNER confidence threshold for NER detection | Lower to catch more entities (more false positives); raise to reduce false positives |

See [`PiiReplacerConfig`][nemo_safe_synthesizer.config.replace_pii.PiiReplacerConfig]
for the full schema.

---

## Differential Privacy

Differential privacy (DP) provides a formal bound on what an adversary can
learn about any individual record. Safe Synthesizer implements DP-SGD
(Differentially Private Stochastic Gradient Descent) via [Opacus](https://opacus.ai/).

| Field | Default | Description | Guidance |
|-------|---------|-------------|----------|
| `privacy.dp_enabled` | `false` | Enable DP-SGD training | Enable for formal privacy guarantees |
| `privacy.epsilon` | `8.0` | Privacy budget -- lower values give stronger privacy | 4.0--12.0 typical; values below 4.0 may make convergence difficult |
| `privacy.delta` | `"auto"` | Privacy failure probability (`"auto"` or float) | Leave at `"auto"` |
| `privacy.per_sample_max_grad_norm` | `1.0` | Max L2 norm for per-sample gradients | Leave at 1.0 |

Compatibility constraints:

- Set `training.use_unsloth` to `false` or leave it at `"auto"` -- `"auto"` resolves to `false` when DP is enabled (Unsloth is incompatible with Opacus's per-sample gradient hooks)
- `data.max_sequences_per_example` must be `1` (or `"auto"`, which resolves to `1` when DP is enabled) -- must be 1 to limit each example's contribution to the gradient, which DP requires
- Safe Synthesizer disables gradient checkpointing automatically when DP is enabled -- no user action required (gradient checkpointing is incompatible with Opacus)

See [`DifferentialPrivacyHyperparams`][nemo_safe_synthesizer.config.differential_privacy.DifferentialPrivacyHyperparams]
for the full field list. For DP error diagnostics, see
[Synthetic Data Quality](evaluating-data.md#common-dp-errors).

---

## Data

| Field | Default | Description | Guidance |
|-------|---------|-------------|----------|
| `data.holdout` | `0.05` | Fraction (0--1) or absolute count (>1) for the holdout test set for evaluation | 0.05--0.15 typical |
| `data.max_holdout` | `2000` | Upper cap on holdout size | Leave at default for most datasets |
| `data.random_state` | `null` | Random seed -- auto-generated if `null`; set an explicit integer for reproducible splits | Set to a fixed integer for reproducibility |
| `data.group_training_examples_by` | `null` | Column to group records by | Use for multi-row entities (e.g. patient ID, session ID) |
| `data.order_training_examples_by` | `null` | Column to order within groups (requires `data.group_training_examples_by`) | Use with a timestamp column for time series data |
| `data.max_sequences_per_example` | `"auto"` | Max sequences per example (`1` for DP, defaults to `10` otherwise) | Leave at `"auto"` |

See [`DataParameters`][nemo_safe_synthesizer.config.data.DataParameters]
for the full field list.

---

## Time Series

!!! warning "Experimental"
    Time series synthesis is an experimental feature. APIs and behavior may
    change between releases.

| Field | Default | Description | Guidance |
|-------|---------|-------------|----------|
| `time_series.is_timeseries` | `false` | Enable time series mode | Enable for datasets with sequential time-ordered records |
| `time_series.timestamp_column` | `null` | Timestamp column name | Required when `is_timeseries: true` |
| `time_series.timestamp_interval_seconds` | `null` | Fixed interval between timestamps | Set if your data has a regular sampling interval |
| `time_series.timestamp_format` | `null` | strftime format or `"elapsed_seconds"` | Required when `is_timeseries: true` |
| `time_series.start_timestamp` | `null` | Override start timestamp for all groups (inferred from data if `null`) | Leave `null` to infer from data |
| `time_series.stop_timestamp` | `null` | Override stop timestamp for all groups (inferred from data if `null`) | Leave `null` to infer from data |

See [`TimeSeriesParameters`][nemo_safe_synthesizer.config.time_series.TimeSeriesParameters]
for the full schema. For detailed descriptions and constraints, see the
[Time Series README](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/src/nemo_safe_synthesizer/TIMESERIES_README.md).

---

## Evaluation

| Field | Default | Description | Guidance |
|-------|---------|-------------|----------|
| `evaluation.enabled` | `true` | Master switch for evaluation | Leave enabled |
| `evaluation.mia_enabled` | `true` | Membership Inference Attack (MIA) -- privacy risk assessment | Disable to speed up evaluation if privacy assessment is not needed |
| `evaluation.aia_enabled` | `true` | Attribute Inference Attack (AIA) -- measures whether an attacker can infer a sensitive attribute from quasi-identifiers in the synthetic data | Disable to speed up evaluation if AIA is not needed |
| `evaluation.pii_replay_enabled` | `true` | PII replay detection -- checks whether PII from training appears in synthetic data | Leave enabled if PII replacement is used |
| `evaluation.sqs_report_columns` | `250` | Max columns in the Synthetic Quality Score (SQS) report | Increase if your dataset has more columns |
| `evaluation.sqs_report_rows` | `5000` | Max rows in the SQS report | Increase for larger datasets (impacts report generation time) |
| `evaluation.quasi_identifier_count` | `3` | Number of quasi-identifiers sampled for AIA (auto-reduced for small datasets) | Leave at default |
| `evaluation.mandatory_columns` | `null` | Number of mandatory columns that must be used in evaluation | Leave at default |

See [`EvaluationParameters`][nemo_safe_synthesizer.config.evaluate.EvaluationParameters]
for the full API reference.

---

## Validate and Modify Configuration

### `config validate`

Check your config for errors and display the merged parameters:

```bash
safe-synthesizer config validate --config config.yaml
safe-synthesizer config validate --config config.yaml --training__learning_rate 0.001
```

Fields set to `"auto"` remain as `"auto"` in the output -- auto-resolution
happens at runtime during `process_data()`, not at validation time. To see
resolved values, check `safe-synthesizer-config.json` in the run directory
after a pipeline run.

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
safe-synthesizer run --config config.yaml --data-source data.csv \
  --training__learning_rate 0.001 \
  --data__holdout 0.1 \
  --generation__num_records 5000
```

!!! note "Override precedence"
    CLI overrides > dataset registry overrides > YAML config file > model
    defaults. See [Configuration Precedence](#configuration-precedence) for
    examples. Parameters that accept `"auto"` cannot be set to `"auto"` via
    CLI flags -- omit the flag to use the default, or set it in YAML.

---

## Configuration Sections

| YAML Key | SDK Method | API Reference |
|----------|-----------|---------------|
| `data` | `with_data()` | [`DataParameters`][nemo_safe_synthesizer.config.data.DataParameters] |
| `training` | `with_train()` | [`TrainingHyperparams`][nemo_safe_synthesizer.config.training.TrainingHyperparams] |
| `generation` | `with_generate()` | [`GenerateParameters`][nemo_safe_synthesizer.config.generate.GenerateParameters] |
| `evaluation` | `with_evaluate()` | [`EvaluationParameters`][nemo_safe_synthesizer.config.evaluate.EvaluationParameters] |
| `replace_pii` (`null` to disable) | `with_replace_pii()` / `with_replace_pii(enable=False)` | [`PiiReplacerConfig`][nemo_safe_synthesizer.config.replace_pii.PiiReplacerConfig] |
| `privacy` (`null` to disable) | `with_differential_privacy()` | [`DifferentialPrivacyHyperparams`][nemo_safe_synthesizer.config.differential_privacy.DifferentialPrivacyHyperparams] |
| `time_series` | `with_time_series()` | [`TimeSeriesParameters`][nemo_safe_synthesizer.config.time_series.TimeSeriesParameters] |

---

- [Running Safe Synthesizer](running.md) -- pipeline execution and examples
- [Environment Variables](environment.md) -- infrastructure and cache settings
- [Program Runtime](troubleshooting.md) -- runtime errors and OOM fixes
