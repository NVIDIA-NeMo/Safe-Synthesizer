<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Program Runtime

Runtime errors, OOM issues, and configuration problems for NeMo Safe
Synthesizer. Sections are organized by pipeline phase. For output quality
and evaluation metrics, see [Synthetic Data Quality](evaluating-data.md). For environment variables, model caching, offline setup, NIM endpoint
configuration, and NER parallelism, see [Environment Variables](environment.md).

---

## Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "kernels package not installed" | No network for Kernels Hub | Set `training.attn_implementation: sdpa` |
| `ConnectionError` during startup | No internet / model not cached | [Pre-cache models](environment.md#pre-caching-models) |
| OOM in training | VRAM exhausted | [Reduce batch size, quantize](#out-of-memory-during-training) |
| OOM in generation | VRAM exhausted | [Verify training cleanup](#out-of-memory-during-generation) |
| OOM in evaluation | Large dataset + PCA | [Reduce columns or disable eval](#out-of-memory-during-evaluation) |
| "Cannot use unsloth without GPU" | No CUDA device | [Switch to HuggingFace backend](#no-gpu-detected) |
| "max_sequences_per_example must be 1" | Incompatible DP config | [Configuration Reference -- Differential Privacy](configuration.md#differential-privacy) |
| "Unsloth not compatible with DP" | Mutual exclusion | [Configuration Reference -- Differential Privacy](configuration.md#differential-privacy) |
| "Unable to automatically determine a noise multiplier" | Epsilon too low | [Increase epsilon or add records](evaluating-data.md#common-dp-errors) |
| "no valid records" in generation | Underfitting / schema mismatch | [See GenerationError](#generationerror) |
| "exceeds context length" | Records too long | [Reduce record size](#context-length-and-record-fitting) |
| "fraction of invalid records" | Generation quality too low | [Lower threshold or retrain](#generationerror) |
| Metrics show UNAVAILABLE | Too few records / columns | [Ensure >= 200 records](evaluating-data.md#minimum-data-requirements) |
| Low SQS scores | Underfit or too few records | [Review distributions](evaluating-data.md#low-sqs-scores) |
| PII uses default entities | Classifier failed | [Set entities explicitly](evaluating-data.md#pii-uses-unexpected-entity-types) |
| "timestamp_column has missing values" | Dirty time series data | Clean NaN/nulls from timestamp column |
| "groups must have same start" | Inconsistent groups | [Align group start timestamps](#groups-must-have-same-start) |

---

## Training

GPU memory, context length, and backend issues during fine-tuning.

### Out of Memory During Training

Training OOM errors appear during the "Training" phase with HuggingFace Trainer
stack traces. If you see `torch.cuda.OutOfMemoryError`:

1. Enable 4-bit quantization -- the single largest memory saver. Set
   `training.quantize_model: true` and `training.quantization_bits: 4`. QLoRA
   stores the frozen base model in 4-bit NF4 while training LoRA adapters in
   full precision, cutting model weight memory by ~4x. Quantization reduces
   precision in the frozen weights; in practice QLoRA typically produces
   results close to full-precision LoRA, but verify with your evaluation report
2. Reduce the context window -- see
   [Context Length and Record Fitting](#context-length-and-record-fitting) for
   how to lower `training.rope_scaling_factor`, truncate records, or simplify
   grouped examples. Longer sequences require more activation memory even with
   gradient checkpointing enabled
3. Verify `training.batch_size` is `1` (the default). The effective batch
   size is `batch_size * gradient_accumulation_steps` (default 1 x 8 = 8).
   Peak memory is set by the forward/backward pass on one micro-batch --
   `gradient_accumulation_steps` controls how many micro-batches accumulate
   before each optimizer step but does not affect peak memory
4. Lower `training.max_vram_fraction` (default `0.8`) to leave headroom for
   other GPU consumers on the same device

GPU memory during LoRA SFT breaks down into three components:

- Base model weights (dominant) -- ~14 GiB for a 7B model in fp16, ~3.5 GiB
  in 4-bit. Quantization targets this component
- Activations (proportional to sequence length and batch size) --
  self-attention computes an n x n score matrix, so activation memory scales
  [quadratically with sequence length](https://huggingface.co/docs/transformers/en/model_memory_anatomy).
  Gradient checkpointing, which Safe Synthesizer enables by default, reduces
  this by recomputing activations during the backward pass instead of storing
  them. Context length and batch size target this component
- LoRA adapter gradients and optimizer states (small) -- typically < 1 GiB
  for standard LoRA ranks

For deeper coverage, see
[Methods and tools for efficient training on a single GPU](https://huggingface.co/docs/transformers/en/perf_train_gpu_one)
in the HuggingFace documentation.

### No GPU Detected

If Safe Synthesizer fails to find a GPU, the Unsloth backend raises immediately:

```text
RuntimeError: Cannot use unsloth without GPU.
```

The HuggingFace backend will not error but will attempt to use CPU (extremely slow).

To diagnose:

1. Verify NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure you installed the CUDA extras, not the CPU-only package:

    ```bash
    pip install "nemo-safe-synthesizer[cu128,engine]"
    ```

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

!!! note "Error type clarification"
    These errors are typed as `GenerationError` in the codebase even though
    they fire during data assembly, not during generation proper. They appear
    in the pipeline before any training or generation occurs.

Context-length issues can also surface as OOM during training (the model
attempts to process sequences near the limit). See
[Out of Memory During Training](#out-of-memory-during-training) for
memory-specific fixes like quantization and batch size reduction.

---

## Generation

VRAM, invalid records, and early stopping during synthetic data production.

### Out of Memory During Generation

Generation OOM errors appear during the "Generation" phase with vLLM.
GPU allocation defaults to 80% of available VRAM. Training exposes
`training.max_vram_fraction` to override this; generation does not yet have
an equivalent config field.

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

: Too many invalid records across `generation.patience` consecutive batches.
  Consider lowering `generation.invalid_fraction_threshold`, retraining with
  more data, or increasing `training.rope_scaling_factor` if records are being
  truncated.

For context-length errors during data assembly (`"The number of tokens in an
example exceeds the available context length"`), see
[Context Length and Record Fitting](#context-length-and-record-fitting).

---

## Evaluation

Memory and scope issues during quality scoring and report generation.

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
    diagnostics are covered in [Synthetic Data Quality](evaluating-data.md#evaluation).

---

## Configuration

Defaults, auto-resolution, and validation errors for pipeline parameters.

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
See [Configuration Reference](configuration.md) for the full list.

- `training.rope_scaling_factor` -- auto-estimated from dataset token counts;
  see [Context Length and Record Fitting](#context-length-and-record-fitting)
  for details and caveats
- `training.num_input_records_to_sample` -- derived from `rope_scaling_factor * 25000`
- `training.use_unsloth` -- resolves to `true` unless DP is enabled.
  DP uses Opacus per-sample gradients (`GradSampleModule`), which require
  standard model layers and disable gradient checkpointing -- Unsloth's
  custom layers and checkpointing are incompatible
- `training.learning_rate` -- model-specific default from `ModelMetadata`:
  Mistral uses 0.0001, all other supported model families use 0.0005
- `data.max_sequences_per_example` -- resolves to `1` when differential
  privacy is enabled (required to limit per-example gradient contribution),
  `10` otherwise for best performance
- `privacy.delta` -- computed from record count

!!! warning "Unsloth and Mistral compatibility"
    If you encounter issues when using Unsloth with Mistral models, set
    `training.use_unsloth: false` explicitly. There is no automatic
    detection for this incompatibility.

Use `safe-synthesizer config validate` to see how `"auto"` and default values resolve for
your configuration. Note that some `"auto"` fields (such as
`training.rope_scaling_factor` and `training.num_input_records_to_sample`)
require a dataset to resolve -- they will remain `"auto"` in the validate
output and only resolve during an actual run:

```bash
safe-synthesizer config validate --config config.yaml
```

### Common Validation Errors

`order_training_examples_by` requires `group_training_examples_by`:

: If you set `data.order_training_examples_by` without also setting
  `data.group_training_examples_by`, config validation will fail. Ordering only
  makes sense within groups.

Unsupported file extensions:

: The `url` parameter accepts `.csv`, `.json`, `.jsonl`, `.parquet`, and `.txt`
  files. Other formats raise a `ValueError`.

Incompatible DP settings:

: If `privacy.dp_enabled` is `true` but `use_unsloth` is `true` or
  `data.max_sequences_per_example` is not `1`, config validation will fail
  with a clear error message. Set these to `"auto"` and they will resolve
  correctly.

!!! tip "Differential Privacy"
    DP errors and privacy budget troubleshooting are covered in
    [Synthetic Data Quality](evaluating-data.md#differential-privacy).

---

## PII and NER

Model downloads and processing timeouts for PII detection.

### GLiNER Download Fails

The PII replacer downloads the GLiNER NER model on first use. If the download
fails, it raises an exception immediately.

Pre-download the model by running PII replacement once in an environment
with internet access, or set `LOCAL_FILES_ONLY=true` after the model is cached.

### NER Processing Timeouts

NER uses an internal `max_runtime_seconds` timeout. If processing a chunk takes
too long, it is dropped with a warning in the logs.

Check the logs for timeout warnings. The timeout is not currently
configurable; for large datasets, reduce the amount of text processed per
chunk (for example, shorten text fields or split them into smaller pieces) and
optionally reduce CPU parallelism so each worker has more resources.

---

## WandB

### Authentication Failures

WandB requires an API key when running in `online` mode. If the key is missing
or invalid, training will fail when the WandB run is initialized.

```text
wandb: ERROR api_key not configured (no-auth)
```

Set the API key before running:

```bash
export WANDB_API_KEY="your-api-key"  # pragma: allowlist secret
```

Or switch to offline mode to avoid network access entirely:

```bash
safe-synthesizer run --wandb-mode disabled --config config.yaml --data-source data.csv
```

See [Running Safe Synthesizer -- WandB Integration](running.md#wandb-integration) for the full WandB setup.

### Resume Errors

If a WandB run fails to resume (e.g., the run ID no longer exists on the WandB server),
pass `--wandb-resume-job-id` with a valid run ID from the same WandB project, or
remove the argument to start a fresh WandB run.

---

## Time Series

!!! warning "Experimental"
    Time series synthesis is an experimental feature. APIs and behavior may
    change between releases.

Time series synthesis has additional validation and generation requirements.
For configuration examples, see [Configuration -- Time Series](configuration.md#time-series).

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

: If a group consistently produces invalid records (exceeding
  `generation.patience` consecutive batches above
  `generation.invalid_fraction_threshold`), that group is skipped entirely.
  Check your training data quality for those groups.

Out-of-order records:

: During generation, records are validated for chronological order. Records
  that arrive out of order are marked invalid.

#### Groups must have same start

All groups in the dataset must begin at the same timestamp when
`time_series.start_timestamp` is `null` (inferred from data). If group
start timestamps differ, the pipeline raises a `DataError`. Either align
all group start timestamps in your data, or set
`time_series.start_timestamp` to an explicit value that applies to all
groups.

---

## Error Classes

Safe Synthesizer uses a structured error hierarchy. Understanding which error
class you received helps narrow down the cause and write targeted `except` clauses.

Inheritance:

```text
SafeSynthesizerError
├── InternalError (also RuntimeError)
└── UserError
    ├── DataError (also ValueError)
    ├── ParameterError (also ValueError)
    └── GenerationError (also RuntimeError)
```

SDK callers can catch [`UserError`][nemo_safe_synthesizer.errors.UserError] to handle all user-facing errors, or
[`SafeSynthesizerError`][nemo_safe_synthesizer.errors.SafeSynthesizerError] to also catch internal errors. Catching the built-in
base (`ValueError`, `RuntimeError`) also works since each class inherits from
both.

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
option combinations, or missing required parameters. The stacktrace will indicate which parameter is invalid.

Checklist:

1. Run `safe-synthesizer config validate --config config.yaml`
2. Verify column names in `group_training_examples_by` and
   `order_training_examples_by` exist in your data
3. For DP, ensure all required privacy parameters are set

### GenerationError

Errors during generation or data assembly. Two common cases:

- Sampling failures (no valid records, patience exceeded) -- see [GenerationError](#generationerror) in the Generation section
- Context-length errors during data assembly (records too long for the model) -- see [Context Length and Record Fitting](#context-length-and-record-fitting)

### InternalError

Library bugs. If you encounter this error through documented interfaces,
please [file an issue on GitHub](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues).

---

- [Running Safe Synthesizer](running.md) -- pipeline execution and CLI commands
- [Configuration Reference](configuration.md) -- parameter tables
- [Synthetic Data Quality](evaluating-data.md) -- quality and privacy score diagnostics
