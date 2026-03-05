<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

Runtime errors, OOM issues, and configuration problems for NeMo Safe
Synthesizer. Sections are organized by pipeline phase. For output quality
and evaluation metrics, see [Data Quality](data-quality.md). For attention
backends, downloads, and offline setup, see
[Environment and Runtime](environment.md).

## Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "kernels package not installed" | No network for Kernels Hub | Set `attn_implementation: sdpa` ([details](environment.md#attention-backends)) |
| `ConnectionError` during startup | No internet / model not cached | [Pre-cache models](environment.md#pre-caching-models) |
| OOM in training | VRAM exhausted | [Reduce batch size, quantize](#out-of-memory-during-training) |
| OOM in generation | VRAM exhausted | [Verify training cleanup](#out-of-memory-during-generation) |
| OOM in evaluation | Large dataset + PCA | [Reduce columns or disable eval](#out-of-memory-during-evaluation) |
| "Cannot use unsloth without GPU" | No CUDA device | [Switch to HuggingFace backend](#no-gpu-detected) |
| "max_sequences_per_example must be 1" | Incompatible DP config | [Set `max_sequences_per_example: 1`](data-quality.md#requirements) |
| "Unsloth not compatible with DP" | Mutual exclusion | [Set `use_unsloth: false`](data-quality.md#requirements) |
| "Unable to determine noise multiplier" | Epsilon too low | [Increase epsilon or add records](data-quality.md#common-dp-errors) |
| "no valid records" in generation | Underfitting / schema mismatch | [See GenerationError](#generationerror) |
| "exceeds context length" | Records too long | [Reduce record size](#context-length-and-record-fitting) |
| "fraction of invalid records" | Generation quality too low | [Lower threshold or retrain](#generationerror) |
| Metrics show UNAVAILABLE | Too few records / columns | [Ensure >= 200 records](data-quality.md#minimum-data-requirements) |
| Low SQS scores | Underfit or too few records | [Review distributions](data-quality.md#low-sqs-scores) |
| PII uses default entities | Classifier failed | [Set entities explicitly](data-quality.md#pii-uses-unexpected-entity-types) |
| "timestamp_column has missing values" | Dirty time series data | Clean NaN/nulls from timestamp column |
| "groups must have same start" | Inconsistent groups | Align group timestamps |

---

## Training

### Out of Memory During Training

Training OOM errors appear during the "Training" phase with HuggingFace Trainer
stack traces. If you see `torch.cuda.OutOfMemoryError`:

1. Reduce `training.batch_size` (default is `1`, effective batch size includes
   `gradient_accumulation_steps` which defaults to `8`)
2. Reduce the context window -- see
   [Context Length and Record Fitting](#context-length-and-record-fitting) for
   how to lower `training.rope_scaling_factor`, truncate records, or simplify
   grouped examples
3. Enable quantization by setting `training.quantize_model: true` and choosing
   `training.quantization_bits: 4` (4-bit) or `8` (8-bit)

### No GPU Detected

If Safe Synthesizer fails to find a GPU, the Unsloth backend raises immediately:

```text
RuntimeError: Cannot use unsloth without GPU
```

The HuggingFace backend will not error but will attempt to use CPU (very slow).

To diagnose:

1. Verify NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure you installed the `cu128` extras, not `cpu`:

    ```bash
    uv sync --extra cu128 --extra engine
    # or: make bootstrap-nss cuda
    ```

The Unsloth backend requires a GPU and raises immediately if none is found.
Switch to the HuggingFace backend for CPU-only environments (useful for
development, not recommended for production training).

### Context Length and Record Fitting

The effective context window (`max_seq_length`) is a computed property on
[`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata] --
`base_max_seq_length * rope_scaling_factor`. Every training example must
fit within this window. If it doesn't, data assembly fails with a
`GenerationError` before training even starts.

#### When Records Don't Fit

Two error messages indicate context-length problems during data assembly:

```text
The number of tokens in an example exceeds the available context length
```

A single training example (schema prompt + records) exceeds
`max_seq_length`.

```text
The dataset schema requires more tokens than the max length of the model
```

The schema prompt alone is wider than `max_seq_length` -- typically
because the table has too many columns for the model's context window.

#### How to Fix

1. Increase `training.rope_scaling_factor` to extend the context window.
   When set to `"auto"`, it is estimated from dataset token counts using
   heuristics (4 chars per token for text, 1 token per digit) -- this can
   underestimate for complex or multilingual data.
2. Reduce record size -- shorten text fields, drop unnecessary columns,
   or simplify the schema.
3. When using `data.group_training_examples_by`, multiple records must fit
   in context together, making the limit tighter. Consider reducing
   `data.max_sequences_per_example` or simplifying the grouped records.

!!! note
    These errors are typed as `GenerationError` in the codebase even though
    they fire during data assembly, not during generation proper. They appear
    in the pipeline before any training or generation occurs.

Context-length issues can also surface as OOM during training (the model
attempts to process sequences near the limit). See
[Out of Memory During Training](#out-of-memory-during-training) for
memory-specific fixes like quantization and batch size reduction.

---

## Generation

### Out of Memory During Generation

Generation OOM errors appear during the "Generation" phase with vLLM.
See [VRAM Management](environment.md#vram-management) for how memory is allocated.

1. Ensure no other processes hold GPU memory -- training cleanup should release
   it, but verify with `nvidia-smi`
2. If the GPU has less memory than expected, check that the training teardown
   completed before generation started

### GenerationError

Generation failures during synthetic data production. The two most common:

```text
Generation stopped prematurely due to no valid records
```

: The first batch produced zero valid records. The model may be underfitting
  or the schema may not match the training data. Increase
  `training.num_input_records_to_sample` to give the model more context,
  and check training logs for quality issues.

```text
Generation stopped prematurely because the average fraction of invalid records was higher than...
```

: Too many invalid records across `patience` consecutive batches. Consider
  lowering `generation.invalid_fraction_threshold`, retraining with more data,
  or increasing `training.rope_scaling_factor` if records are being truncated.

For context-length errors during data assembly (`"The number of tokens in an
example exceeds the available context length"`), see
[Context Length and Record Fitting](#context-length-and-record-fitting).

---

## Evaluation

### Out of Memory During Evaluation

If evaluation OOMs, reduce the evaluation scope or dataset size:

1. For wide datasets, PCA computation in deep structure analysis can OOM.
   Reduce the number of columns included in evaluation by lowering
   `evaluation.sqs_report_columns` or by subsetting the input data. If
   evaluation is not required for your run, disable it entirely with
   `evaluation.enabled: false`.
2. Histogram binning uses the `doane` method to reduce memory, but very large
   datasets may still cause issues. Reduce `evaluation.sqs_report_columns` or
   `evaluation.sqs_report_rows` to limit the evaluation scope.

!!! tip "Evaluation and Data Quality"
    SQS scores, UNAVAILABLE metrics, report limits, and low-quality
    diagnostics are covered in [Data Quality > Evaluation](data-quality.md#evaluation).

---

## Configuration

### Surprising Defaults

Several defaults may not match your expectations:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `training.batch_size` | `1` | Effective batch = `batch_size` x `gradient_accumulation_steps` (8) |
| `training.validation_ratio` | `0.0` | No validation split by default |
| `data.holdout` | `0.05` | 5% of records held out for evaluation; capped by `data.max_holdout` (2000) |
| `data.random_state` | `None` | Auto-generates a random seed -- set explicitly for reproducibility |
| `generation.num_records` | `1000` | May be too small for production use |

### Auto-Resolved Parameters

Many parameters accept `"auto"` and are resolved at runtime by the
[`AutoConfigResolver`][nemo_safe_synthesizer.config.autoconfig.AutoConfigResolver].
See the [Parameters Reference](parameters.md) for the full list.

- `training.rope_scaling_factor` -- auto-estimated from dataset token counts;
  see [Context Length and Record Fitting](#context-length-and-record-fitting)
  for details and caveats
- `training.num_input_records_to_sample` -- derived from `rope_scaling_factor * 25000`
- `training.use_unsloth` -- resolves to `true` unless DP is enabled.
  DP uses Opacus per-sample gradients (`GradSampleModule`), which require
  standard model layers and disable gradient checkpointing -- Unsloth's
  custom layers and checkpointing are incompatible
- `privacy.delta` -- computed from record count

!!! warning
    If you encounter issues when using Unsloth with Mistral models, set
    `training.use_unsloth: false` explicitly. There is no automatic
    detection for this incompatibility.

Use `safe-synthesizer config validate` to see how `"auto"` values resolve for
your configuration:

=== "CLI"

    ```bash
    safe-synthesizer config validate --config config.yaml
    ```

### Common Validation Errors

`order_training_examples_by` requires `group_training_examples_by`:

: If you set `data.order_training_examples_by` without also setting
  `data.group_training_examples_by`, config validation will fail. Ordering only
  makes sense within groups.

Unsupported file extensions:

: The `url` parameter accepts `.csv`, `.txt`, `.json`, `.jsonl`, and `.parquet`
  files. Other formats raise a `ValueError`.

Incompatible DP settings:

: If `privacy.dp_enabled` is `true` but `use_unsloth` is `true` or
  `data.max_sequences_per_example` is not `1`, config validation will fail
  with a clear error message. Set these to `"auto"` and they will resolve
  correctly.

!!! tip "Differential Privacy"
    DP errors and privacy budget troubleshooting are covered in
    [Data Quality > Differential Privacy](data-quality.md#differential-privacy).

---

## PII and NER

### GLiNER Download Fails

The PII replacer downloads the GLiNER NER model on first use. If the download
fails, it raises an exception immediately.

Fix: pre-download the model by running PII replacement once in an environment
with internet access, or set `LOCAL_FILES_ONLY=true` after the model is cached.

### NER Processing Timeouts

NER uses an internal `max_runtime_seconds` timeout. If processing a chunk takes
too long, it is dropped with a warning in the logs.

Fix: check the logs for timeout warnings. The timeout is not currently
configurable; for large datasets, reduce the amount of text processed per
chunk (for example, shorten text fields or split them into smaller pieces) and
optionally reduce CPU parallelism so each worker has more resources.

---

## Time Series

!!! warning "Experimental"
    Time series synthesis is an experimental feature. APIs and behavior may
    change between releases.

Time series synthesis has additional validation and generation requirements.

### Configuration Requirements

=== "YAML"

    ```yaml
    time_series:
      is_timeseries: true
      timestamp_column: "timestamp"
      timestamp_interval_seconds: 60
      timestamp_format: "%Y-%m-%d %H:%M:%S"
    ```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --time_series__is_timeseries true \
      --time_series__timestamp_column timestamp \
      --time_series__timestamp_interval_seconds 60 \
      --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_time_series(
            is_timeseries=True,
            timestamp_column="timestamp",
            timestamp_interval_seconds=60,
            timestamp_format="%Y-%m-%d %H:%M:%S",
        )
    )
    ```

- Set `time_series.is_timeseries: true` and provide at least one of
  `timestamp_column` or `timestamp_interval_seconds`
- `timestamp_format` must be a valid strftime string or `"elapsed_seconds"` --
  invalid formats are caught at config validation time
- All groups must share the same start and stop timestamps, or preprocessing
  raises a `DataError`

### Common Issues

Missing timestamp values:

: Any `NaN` or `null` values in the timestamp column raise a `DataError`.
  Clean your data before running the pipeline:

    ```python
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(by=["group_column", "timestamp"])
    ```

Interval mismatch:

: If `timestamp_interval_seconds` does not match the actual intervals in your
  data, a warning is logged but the pipeline continues. Verify your interval
  setting matches the data.

Groups skipped during generation:

: If a group consistently produces invalid records (exceeding `patience`
  consecutive batches above `invalid_fraction_threshold`), that group is
  skipped entirely. Check your training data quality for those groups.

Out-of-order records:

: During generation, records are validated for chronological order. Records
  that arrive out of order are marked invalid.

---

## Error Classes

Safe Synthesizer uses a structured error hierarchy. Understanding which error
class you received helps narrow down the cause.

### DataError

Bad input data -- NaNs, unsupported types, empty DataFrames, missing values
in group or timestamp columns.

Checklist:

1. Verify your CSV loads cleanly with `pd.read_csv()`
2. Check for mixed types in columns
3. Check that column names in your config match the actual data
4. For time series, ensure no nulls in timestamp or group columns

Context-length errors (records too long for the model) raise `GenerationError`,
not `DataError` -- see [Context Length and Record Fitting](#context-length-and-record-fitting).

### ParameterError

Invalid configuration -- missing columns referenced in config, incompatible
option combinations, or missing required parameters. The stacktrace should hopefully be informative.

Checklist:

1. Run `safe-synthesizer config validate --config config.yaml`
2. Verify column names in `group_training_examples_by` and
   `order_training_examples_by` exist in your data
3. For DP, ensure all required privacy parameters are set

### InternalError

Library bugs. If you encounter this error through documented interfaces,
please [file an issue on GitHub](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues).

---

## Reference

See [Environment and Runtime](environment.md) for attention backends, model
downloads, offline setup, VRAM allocation, and NER parallelism.
