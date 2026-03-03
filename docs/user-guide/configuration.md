<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuration

Synthesis parameters control what the pipeline produces -- model choice,
learning rate, number of records, privacy settings. Configure them via YAML,
CLI flags, or the SDK. For how to run the pipeline and what each stage does,
see [Pipeline Stages](pipeline-stages.md). For environment variables, logging,
and CLI commands, see [Reference](cli.md).

!!! note "SDK file format limitation"
    The SDK currently loads string paths via `pd.read_csv`, so only CSV and
    TXT files work directly. For JSON, JSONL, or Parquet, load into a
    DataFrame first (e.g., `pd.read_json(...)`, `pd.read_parquet(...)`).
    A future release will unify SDK format support with the CLI.

---

## Training

Safe Synthesizer fine-tunes a pretrained language model on your tabular data
using LoRA (Low-Rank Adaptation). See
[`TrainingHyperparams`][nemo_safe_synthesizer.config.training.TrainingHyperparams]
for the full field list.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `training.learning_rate` | float | `0.0005` | Initial learning rate for the AdamW optimizer |
| `training.batch_size` | int | `1` | Per-device batch size |
| `training.gradient_accumulation_steps` | int | `8` | Steps to accumulate before a backward pass; effective batch size = `batch_size` x this value |
| `training.num_input_records_to_sample` | `"auto"` or int | `"auto"` | Records the model sees during training -- proxy for training time |
| `training.lora_r` | int | `32` | LoRA rank; lower values produce fewer trainable parameters |
| `training.lora_alpha_over_r` | float | `1.0` | LoRA scaling ratio (alpha / rank) |
| `training.pretrained_model` | str | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` | HuggingFace model ID or local path (see note below) |
| `training.use_unsloth` | `"auto"` or bool | `"auto"` | Use the Unsloth backend. Must be `false` for DP training |
| `training.quantize_model` | bool | `false` | Enable quantization to reduce VRAM usage |
| `training.quantization_bits` | 4 or 8 | `8` | Bit width when `training.quantize_model` is `true` |
| `training.attn_implementation` | str | `"kernels-community/vllm-flash-attn3"` | Attention backend for model loading |
| `training.rope_scaling_factor` | `"auto"` or int | `"auto"` | Scale the base model's context window via RoPE |
| `training.validation_ratio` | float | `0.0` | Fraction of training data held out for validation loss monitoring |

!!! note "validation_ratio vs holdout"
    `training.validation_ratio` splits the training data to monitor
    validation loss during fine-tuning. `data.holdout` splits the full
    dataset to create a test set used by the evaluation stage. They serve
    different purposes and are applied at different stages.

!!! tip "Recommended models"
    For quick experiments, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (default)
    trains in minutes on a single GPU. For higher-quality output, try
    `HuggingFaceTB/SmolLM3-3B` or `mistralai/Mistral-7B-v0.1`.
    Any causal LM on HuggingFace Hub works -- larger models produce
    better results but require more VRAM and training time.

=== "YAML"

    ```yaml
    training:
      learning_rate: 0.001
      batch_size: 4
    ```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --training__learning_rate 0.001 \
      --training__batch_size 4 \
      --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_train(learning_rate=0.001, batch_size=4)
    )
    ```

### Quantization

Enabling quantization reduces VRAM consumption at the cost of some numerical
precision. Set `training.quantize_model` to `true` and choose a bit width with
`training.quantization_bits`.

=== "YAML"

    ```yaml
    training:
      quantize_model: true
      quantization_bits: 4
    ```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --training__quantize_model true \
      --training__quantization_bits 4 \
      --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_train(quantize_model=True, quantization_bits=4)
    )
    ```

### Attention Backends

`training.attn_implementation` controls which attention kernel is used when
loading the model. The default uses Flash Attention 3 via the HuggingFace
Kernels Hub and falls back to `sdpa` when the `kernels` package is not
installed.

Common values:

- `kernels-community/vllm-flash-attn3`: Flash Attention 3 (default, requires `kernels` package)
- `flash_attention_2`: Flash Attention 2 (requires `flash-attn` package)
- `sdpa`: PyTorch scaled dot-product attention -- broadest compatibility
- `eager`: standard PyTorch attention -- useful for debugging

!!! note
    The training attention backend (`training.attn_implementation`) and the
    generation attention backend (`generation.attention_backend` /
    `VLLM_ATTENTION_BACKEND`) are independent settings.

---

## Generation

The generation stage controls how the fine-tuned model produces synthetic
records. See
[`GenerateParameters`][nemo_safe_synthesizer.config.generate.GenerateParameters]
for the full API reference.

| Parameter | Default | Notes |
|-----------|---------|-------|
| `generation.num_records` | `1000` | Number of synthetic records to generate |
| `generation.temperature` | `0.9` | Sampling temperature (0.7--1.0 typical) |
| `generation.top_p` | `1.0` | Nucleus sampling probability (0.9--1.0 typical) |
| `generation.repetition_penalty` | `1.0` | Penalty for repeated tokens (1.0--1.2 typical) |
| `generation.patience` | `3` | Consecutive bad batches before stopping |
| `generation.invalid_fraction_threshold` | `0.8` | Invalid record fraction that triggers the patience counter |
| `generation.use_structured_generation` | `false` | Enable structured output to constrain record format (typically at the cost of reducing the quality of generated records) |
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

=== "YAML"

    ```yaml
    generation:
      num_records: 5000
      temperature: 0.7
    ```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --generation__num_records 5000 \
      --generation__temperature 0.7 \
      --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_generate(num_records=5000, temperature=0.7)
    )
    ```

### Structured Generation

Set `generation.use_structured_generation` to `true` to constrain the model's
output so every record matches the dataset schema. This reduces the fraction of
invalid records.

- `"regex"`: constructs a custom regex from the dataset schema. More comprehensive but slower.
- `"json_schema"`: passes a JSON Schema to the backend. Faster, but may miss edge cases.

### Stopping Conditions

Generation stops early when too many consecutive batches produce mostly invalid
records. `generation.patience` controls how many bad batches to tolerate;
`generation.invalid_fraction_threshold` defines what counts as "bad." If the
pipeline stops early, check the generation logs for the invalid record
fraction per batch.

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

=== "YAML"

    ```yaml
    enable_replace_pii: true
    ```

    PII replacement is on by default. To customize entity types or
    classification, use the SDK builder -- the `replace_pii` config block
    requires the full `steps` field which is verbose in YAML.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --enable_replace_pii true \
      --url data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig

    pii_config = PiiReplacerConfig.get_default_config()
    pii_config.globals.classify.enable_classify = True
    pii_config.globals.classify.entities = ["email", "phone_number", "ssn"]

    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_replace_pii(config=pii_config)
        .with_train()
        .with_generate(num_records=5000)
    )
    ```

    The SDK builder merges partial overrides with
    `PiiReplacerConfig.get_default_config()`, so you don't need to
    provide the full `steps` list.

#### LLM Column Classification

To enable LLM-based column classification (optional), set the endpoint
before running the pipeline. Any OpenAI-compatible inference endpoint
works -- not just NVIDIA NIM:

```bash
export NIM_ENDPOINT_URL="https://your-inference-endpoint"
export NIM_API_KEY="your-api-key"  # pragma: allowlist secret  (optional -- only needed for direct endpoints, not inference gateways)
```

When `NIM_ENDPOINT_URL` is unset, the classification step is attempted but
falls back to NER-only detection (with an error log). No environment
variables are required for NER-only PII replacement; column classification
requires `NIM_ENDPOINT_URL`.

See [`PiiReplacerConfig`][nemo_safe_synthesizer.config.replace_pii.PiiReplacerConfig]
for the full schema.

---

## Differential Privacy

Differential privacy (DP) provides a formal bound on what an adversary can
learn about any individual record. Safe Synthesizer implements DP-SGD via
Opacus.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `privacy.dp_enabled` | bool | `false` | Enable DP-SGD training |
| `privacy.epsilon` | float | `8.0` | Privacy budget -- lower values give stronger privacy (1.0--10.0 typical) |
| `privacy.delta` | `"auto"` or float | `"auto"` | Privacy failure probability |
| `privacy.per_sample_max_grad_norm` | float | `1.0` | Max L2 norm for per-sample gradients |

=== "YAML"

    ```yaml
    privacy:
      dp_enabled: true
      epsilon: 8.0
    ```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --privacy__dp_enabled true \
      --privacy__epsilon 8.0 \
      --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_differential_privacy(dp_enabled=True, epsilon=8.0)
    )
    ```

### Compatibility Constraints

- Unsloth must be disabled (`training.use_unsloth: false` or `"auto"`)
- `data.max_sequences_per_example` must be `1`
- Gradient checkpointing is disabled (incompatible with Opacus)

See [`DifferentialPrivacyHyperparams`][nemo_safe_synthesizer.config.differential_privacy.DifferentialPrivacyHyperparams]
for the full field list.

---

## Data

| Field | Default | Notes |
|-------|---------|-------|
| `data.holdout` | `0.05` | Fraction (0--1) or absolute count (>1) for the holdout test set used by evaluation |
| `data.max_holdout` | `2000` | Upper cap on holdout size |
| `data.random_state` | `null` | Random seed (auto-generated if null) |
| `data.group_training_examples_by` | `null` | Column to group records by |
| `data.order_training_examples_by` | `null` | Column to order within groups (requires group_by) |
| `data.max_sequences_per_example` | `"auto"` | Max sequences per example (`1` for DP, defaults to `10` otherwise) |

=== "YAML"

    ```yaml
    data:
      group_training_examples_by: "customer_id"
      order_training_examples_by: "transaction_date"
      holdout: 0.1
      random_state: 42
    ```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --data__group_training_examples_by customer_id \
      --data__order_training_examples_by transaction_date \
      --data__holdout 0.1 \
      --url transactions.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("transactions.csv")
        .with_data(
            group_training_examples_by="customer_id",
            order_training_examples_by="transaction_date",
            holdout=0.1,
        )
    )
    ```

### Dataset Registry

Define named datasets in a YAML file to reference them by name:

```yaml
base_url: "/data/datasets"
datasets:
  - name: "customer_transactions"
    url: "customers/transactions.csv"
    overrides:
      data:
        group_training_examples_by: "customer_id"
```

```bash
safe-synthesizer run --dataset-registry registry.yaml --url customer_transactions
```

See [`DataParameters`][nemo_safe_synthesizer.config.data.DataParameters]
for the full field list.

---

## Time Series

!!! warning "Experimental"
    Time series synthesis is an experimental feature. APIs and behavior may
    change between releases.

| Field | Default | Notes |
|-------|---------|-------|
| `time_series.is_timeseries` | `false` | Enable time series mode |
| `time_series.timestamp_column` | `null` | Timestamp column name |
| `time_series.timestamp_interval_seconds` | `null` | Fixed interval between timestamps |
| `time_series.timestamp_format` | `null` | strftime format or `"elapsed_seconds"` |

=== "YAML"

    ```yaml
    time_series:
      is_timeseries: true
      timestamp_column: "timestamp"
      timestamp_interval_seconds: 60
    data:
      group_training_examples_by: "sensor_id"
      order_training_examples_by: "timestamp"
    ```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --time_series__is_timeseries true \
      --time_series__timestamp_column timestamp \
      --time_series__timestamp_interval_seconds 60 \
      --data__group_training_examples_by sensor_id \
      --url sensor_data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("sensor_data.csv")
        .with_time_series(
            is_timeseries=True,
            timestamp_column="timestamp",
            timestamp_interval_seconds=60,
        )
        .with_data(
            group_training_examples_by="sensor_id",
            order_training_examples_by="timestamp",
        )
    )
    ```

See [`TimeSeriesParameters`][nemo_safe_synthesizer.config.time_series.TimeSeriesParameters]
for the full schema. For detailed descriptions and constraints, see the
[Time Series README](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/src/nemo_safe_synthesizer/TIMESERIES_README.md).

---

## Evaluation

| Parameter | Default | Notes |
|-----------|---------|-------|
| `evaluation.enabled` | `true` | Master switch for evaluation |
| `evaluation.mia_enabled` | `true` | Membership inference attack |
| `evaluation.aia_enabled` | `true` | Attribute inference attack |
| `evaluation.pii_replay_enabled` | `true` | PII replay detection |
| `evaluation.sqs_report_columns` | `250` | Max columns in SQS report |
| `evaluation.sqs_report_rows` | `5000` | Max rows in SQS report |

=== "YAML"

    ```yaml
    evaluation:
      mia_enabled: false
      aia_enabled: false
    ```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --evaluation__mia_enabled false \
      --evaluation__aia_enabled false \
      --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_evaluate(mia_enabled=False, aia_enabled=False)
    )
    ```

See [`EvaluationParameters`][nemo_safe_synthesizer.config.evaluate.EvaluationParameters]
for the full API reference.

---

## Validate Configuration

Check your config for errors before running:

```bash
safe-synthesizer config validate --config config.yaml
```

Fields set to `"auto"` remain as `"auto"` in the output -- auto-resolution
happens at runtime during `process_data()`, not at validation time. To see
resolved values, check `safe-synthesizer-config.json` in the run directory
after a pipeline run.

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
