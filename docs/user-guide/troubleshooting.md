<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

This guide covers common issues, error messages, and configuration pitfalls
you may encounter when using NeMo Safe Synthesizer. Problem-to-solution
sections come first; [Reference](#reference) material is at the end.

## Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "kernels package not installed" warning | `kernels` pip package missing | `pip install kernels` or set `training.attn_implementation: sdpa` |
| `ConnectionError` during startup | No internet, model not cached | [Pre-cache models](#pre-caching-models), set `HF_HUB_OFFLINE=1` after caching |
| `torch.cuda.OutOfMemoryError` in training | VRAM exhausted | [Reduce batch size, enable quantization](#out-of-memory-during-training) |
| `torch.cuda.OutOfMemoryError` in generation | VRAM exhausted | [Check training cleanup, verify free VRAM](#out-of-memory-during-generation) |
| `torch.cuda.OutOfMemoryError` in evaluation | Large dataset + PCA/binning | [Disable expensive components or reduce columns](#out-of-memory-during-evaluation) |
| "Cannot use unsloth without GPU" | No CUDA device | [Use HuggingFace backend or install CUDA drivers](#no-gpu-detected) |
| "max_sequences_per_example must be set to 1 or 'auto'" | Incompatible DP config | Set `data.max_sequences_per_example: 1` or `"auto"` |
| "Unsloth is currently not compatible with DP" | Mutual exclusion | Set `use_unsloth: false` for DP training |
| "Unable to determine a noise multiplier" | Epsilon too low for dataset | [Increase epsilon or add records](#common-dp-errors) |
| "Generation stopped prematurely due to no valid records" | Model underfitting or schema mismatch | [Increase `num_input_records_to_sample`, check training logs](#generationerror) |
| "Number of tokens exceeds context length" | Records too long for model | [Reduce record size or increase context window](#context-length-and-record-fitting) |
| "fraction of invalid records was higher than..." | Generation quality too low | [Lower `invalid_fraction_threshold` or retrain](#generationerror) |
| Evaluation metrics show UNAVAILABLE | Too few records or columns | [Ensure >= 200 records, >= 3 columns](#minimum-data-requirements) |
| Low SQS quality scores | Model underfit or too few records | [Review distributions, increase records/epochs](#low-sqs-scores) |
| PII uses default entities unexpectedly | Column classifier failed | [Check logs, set entities explicitly](#pii-uses-unexpected-entity-types) |
| "timestamp_column has missing values" | Dirty time series data | Clean NaN/null values from the timestamp column |
| "groups must have same start timestamp" | Inconsistent time series groups | Align group start/stop timestamps in preprocessing |

---

## GPU and OOM

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

!!! note
    There is a known memory leak in the HuggingFace Trainer evaluation loop
    that can cause OOM on large datasets. The codebase includes a workaround,
    but extremely large evaluation sets may still cause issues.

### Out of Memory During Generation

Generation OOM errors appear during the "Generation" phase with vLLM stack
traces. vLLM manages its own memory pool using `get_max_vram()` (defaults to
80% of available VRAM). If generation OOMs:

1. Ensure no other processes hold GPU memory -- training cleanup should release
   it, but verify with `nvidia-smi`
2. If the GPU has less memory than expected, check that the training teardown
   completed before generation started

### Out of Memory During Evaluation

If evaluation OOMs, disable the expensive components or reduce dataset size:

1. For wide datasets, PCA computation in deep structure analysis can OOM.
   Reduce the number of columns or disable the deep structure component.
2. Histogram binning uses `doane` method to reduce memory, but very large
   datasets may still cause issues. Reduce `sqs_report_columns` or
   `sqs_report_rows` to limit the evaluation scope.

### No GPU Detected

If Safe Synthesizer fails to find a GPU, you will see one of these errors:

- Unsloth backend: `RuntimeError: Cannot use unsloth without GPU`
- HuggingFace backend: training will attempt to use CPU (very slow)

To diagnose:

1. Verify NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`
3. Ensure you installed the `cu128` extras, not `cpu`

The Unsloth backend requires a GPU and raises immediately if none is found.
Switch to the HuggingFace backend for CPU-only environments (useful for
development, not recommended for production training).

---

## Context Length and Record Fitting

The effective context window (`max_seq_length`) is a computed property on
[`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata] --
`base_max_seq_length * rope_scaling_factor`. Every training example must
fit within this window. If it doesn't, data assembly fails with a
`GenerationError` before training even starts.

### When Records Don't Fit

Two error messages indicate context-length problems during data assembly:

"The number of tokens in an example exceeds the available context length":

: A single training example (schema prompt + records) exceeds
  `max_seq_length`.

"The dataset schema requires more tokens than the max length of the model":

: The schema prompt alone is wider than `max_seq_length` -- typically
  because the table has too many columns for the model's context window.

### How to Fix

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

## Configuration Gotchas

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
- `training.use_unsloth` -- resolves to `true` unless DP is enabled. Also needs
  to be set to `false` for Mistral models until we fix issues with Mistral and Unsloth.
- `privacy.delta` -- computed from record count

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

---

## Differential Privacy

DP training has strict requirements. Violating them produces errors that may
not immediately point to the root cause.

### Requirements

- DP and Unsloth are mutually exclusive. If `privacy.dp_enabled` is `true`,
  `use_unsloth` must be `false` or `"auto"` (which resolves to `false`).
- `data.max_sequences_per_example` must be `1` when DP is enabled.
  Set it to `"auto"` and it will resolve correctly.
- `data_fraction` and `true_dataset_size` must be available at runtime --
  these are normally set automatically when running the full pipeline.
- Gradient checkpointing is disabled when using DP (incompatible with Opacus).

### Common DP Errors

"Unable to automatically determine a noise multiplier":

: The privacy budget (epsilon) is too low for your dataset size. Either increase
  `privacy.epsilon` or add more training records.

"Discrete mean differs" warning:

: The PRV accountant failed and the system is falling back to the Opacus RDP
  accountant. This is handled automatically but may produce slightly different
  privacy guarantees.

"Number of entities in dataset is low":

: Small datasets cause poor privacy budget utilization. Consider lowering
  `training.batch_size` or adding more records.

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
  Clean your data before running the pipeline.

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

## PII Replacement

### GLiNER Download Fails

The PII replacer downloads the GLiNER NER model on first use. This respects
the `LOCAL_FILES_ONLY` environment variable, but has no retry logic -- if the
download fails, it raises an exception.

Fix: pre-download the model by running PII replacement once in an environment
with internet access, or set `LOCAL_FILES_ONLY=true` after the model is cached.

### PII Uses Unexpected Entity Types

If PII replacement is not detecting the entity types you expect, the column
classifier may have failed silently. When the classifier fails to initialize
or classify, it falls back to default entity types. Check logs for classification
errors if PII replacement seems to use unexpected entity types.

Fix: set entity types explicitly in your config, or check that `NIM_ENDPOINT_URL`
is reachable:

=== "YAML"

    ```yaml
    replace_pii:
      globals:
        classify:
          enable_classify: true
          entities: ["person", "email", "phone_number"]
    ```

=== "CLI"

    ```bash
    # PII classify config is deeply nested; YAML is recommended.
    safe-synthesizer run \
      --replace_pii__globals__classify__enable_classify true \
      --url data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig

    pii_config = PiiReplacerConfig.get_default_config()
    pii_config.globals.classify.enable_classify = True
    pii_config.globals.classify.entities = ["person", "email", "phone_number"]

    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_replace_pii(config=pii_config)
    )
    ```

### NER Processing Timeouts

NER uses an internal `max_runtime_seconds` timeout. If processing a chunk takes
too long, it is dropped with a warning in the logs.

Fix: check the logs for timeout warnings. The timeout is not currently
configurable; for large datasets, reduce the amount of text processed per
chunk (for example, shorten text fields or split them into smaller pieces) and
optionally reduce CPU parallelism so each worker has more resources.

---

## Evaluation

### Minimum Data Requirements

Several evaluation metrics have minimum data requirements:

| Metric | Minimum | Behavior if Unmet |
|--------|---------|-------------------|
| Holdout split | 200 records | Raises `ValueError` (pipeline stops) |
| Text semantic similarity | 200 records | Silently skipped, warning logged |
| Attribute Inference Attack | 3+ columns | Silently skipped, warning logged |
| PCA (deep structure) | 2x2 matrix | Silently skipped, warning logged |

### Silent Failures

Many evaluation components catch errors and return `UNAVAILABLE` grades instead
of failing the pipeline. If your evaluation report shows missing or `UNAVAILABLE`
metrics:

1. Check the logs for warnings and exceptions
2. Verify you have enough records (>= 200) and columns (>= 3)
3. Verify the SentenceTransformer model downloaded successfully

### Report Truncation

SQS reports are limited to `sqs_report_columns=250` columns and
`sqs_report_rows=5000` rows by default. Larger datasets are silently
truncated in the HTML report. Adjust these in `evaluation` config if needed.

### Low SQS Scores

If the SQS (Synthetic Quality Score) report shows low quality scores:

1. Review column distributions in the HTML report -- large divergences
   indicate the model did not learn the data patterns well
2. Check that training data is representative and not too small
3. Consider increasing `generation.num_records` for a larger sample
4. Increase `training.num_input_records_to_sample` to give the model
   more context during generation
5. Verify the model trained for enough epochs (`training.num_epochs`)

### FAISS

FAISS is included in the `cpu` and `cu128` install extras (`faiss-cpu` and
`faiss-gpu-cu12` respectively). If you installed without extras, membership
inference protection may silently degrade.

---

## Common Errors

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

### ParameterError

Invalid configuration -- missing columns referenced in config, incompatible
option combinations, or missing required parameters.

Checklist:

1. Run `safe-synthesizer config validate --config config.yaml`
2. Verify column names in `group_training_examples_by` and
   `order_training_examples_by` exist in your data
3. For DP, ensure all required privacy parameters are set

### GenerationError

Generation failures during synthetic data production. The two most common:

"Generation stopped prematurely due to no valid records":

: The first batch produced zero valid records. The model may be underfitting
  or the schema may not match the training data. Increase
  `training.num_input_records_to_sample` to give the model more context,
  and check training logs for quality issues.

"Generation stopped prematurely because the average fraction of invalid records was higher than...":

: Too many invalid records across `patience` consecutive batches. Consider
  lowering `generation.invalid_fraction_threshold`, retraining with more data,
  or increasing `training.rope_scaling_factor` if records are being truncated.

For context-length errors during data assembly ("The number of tokens in an
example exceeds the available context length"), see
[Context Length and Record Fitting](#context-length-and-record-fitting).

### InternalError

Library bugs. If you encounter this error through documented interfaces,
please [file an issue on GitHub](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues).

---

## Reference

Background information on subsystems referenced by the troubleshooting sections
above. These sections explain how things work rather than diagnosing problems.

### Attention Backends

Safe Synthesizer uses configurable attention implementations for both training
and generation.

#### Training: `attn_implementation`

The HuggingFace backend loads models with an attention implementation controlled
by `training.attn_implementation`. The default is `kernels-community/vllm-flash-attn3`,
which uses Flash Attention 3 via the
[HuggingFace Kernels Hub](https://huggingface.co/docs/kernels/index).

If the `kernels` pip package is not installed, the backend automatically falls
back to `sdpa` (PyTorch scaled dot-product attention) and logs a warning:

```text
kernels package not installed, cannot use 'kernels-community/vllm-flash-attn3'.
Falling back to 'sdpa'. Install with: pip install kernels
```

To explicitly choose a backend:

=== "YAML"

    ```yaml
    training:
      attn_implementation: "sdpa"
    ```

=== "CLI"

    ```bash
    safe-synthesizer run --training__attn_implementation sdpa --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_train(attn_implementation="sdpa")
    )
    ```

Available values:

| Value | Description | Requires |
|-------|-------------|----------|
| `kernels-community/vllm-flash-attn3` | Flash Attention 3 via Kernels Hub (default) | `kernels` pip package |
| `kernels-community/flash-attn2` | Flash Attention 2 via Kernels Hub | `kernels` pip package |
| `flash_attention_2` | Flash Attention 2 (traditional) | `flash-attn` pip package |
| `sdpa` | PyTorch scaled dot-product attention | None (built-in) |
| `eager` | Standard PyTorch attention | None (built-in) |

!!! tip
    If you see compilation errors related to `flash-attn` or CUDA capability
    mismatches, switch to `sdpa` -- it requires no extra packages and works on
    all CUDA-capable GPUs.

#### Generation: `attention_backend`

The vLLM generation engine uses a separate attention backend setting. Defaults to
`"auto"`, which lets vLLM auto-select the best available backend.

In vLLM 0.11.x, this is set via the `VLLM_ATTENTION_BACKEND` environment variable.
Safe Synthesizer handles this automatically when `generation.attention_backend` is
configured.

=== "YAML"

    ```yaml
    generation:
      attention_backend: "FLASH_ATTN"
    ```

=== "CLI"

    ```bash
    safe-synthesizer run --generation__attention_backend FLASH_ATTN --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_generate(attention_backend="FLASH_ATTN")
    )
    ```

=== "Environment"

    ```bash
    export VLLM_ATTENTION_BACKEND=FLASH_ATTN
    ```

Common values: `FLASHINFER`, `FLASH_ATTN`, `TORCH_SDPA`, `TRITON_ATTN`, `FLEX_ATTENTION`.

!!! note
    Leave this as `"auto"` unless you have a specific reason to override it.
    vLLM's auto-detection works well in most environments.

### Network and Download Behavior

Safe Synthesizer downloads models, tokenizers, and other artifacts at various
pipeline stages. All model and tokenizer downloads go through
[Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

#### What Gets Downloaded

Training:

- Model weights, config, and tokenizer via `AutoModelForCausalLM.from_pretrained()`,
  `AutoConfig.from_pretrained()`, and `AutoTokenizer.from_pretrained()` -- all
  fetched from Hugging Face Hub on first use
- When `training.attn_implementation` starts with `kernels-community/`, the
  `kernels` package downloads compiled attention kernels from Hugging Face Hub
  (see [Attention Backends](#attention-backends))
- Unsloth backend also downloads models via `FastLanguageModel.from_pretrained()`
  through Hugging Face Hub

Generation:

- vLLM downloads the base model through Hugging Face Hub on initialization if
  not already cached

PII replacement:

- GLiNER downloads its NER model on first use via Hugging Face
- Column classification makes requests to a NIM/OpenAI-compatible endpoint
  (controlled by `NIM_ENDPOINT_URL`) for entity type detection

Evaluation:

- `SentenceTransformer("distiluse-base-multilingual-cased-v2")` is downloaded
  for text semantic similarity, attribute inference protection, and membership
  inference protection components
- Tiktoken downloads tokenizer encoding files on first use

!!! warning
    All of these downloads happen silently on first use. If your first run is in
    an environment without internet access, you will see connection errors at
    whichever stage tries to download first.

#### Pre-Caching Models

To avoid runtime downloads, run the pipeline once in an environment with internet
access. All downloaded artifacts are stored under `HF_HOME`
(defaults to `~/.cache/huggingface`). You can then copy or mount this cache
directory in your target environment.

```bash
export HF_HOME=/shared/cache/huggingface

safe-synthesizer run --config config.yaml --url data.csv
```

### Offline and Air-Gapped Environments

Several environment variables control network behavior:

| Variable | Scope | Effect |
|----------|-------|--------|
| `HF_HOME` | All Hugging Face downloads | Set the cache directory for downloaded models |
| `HF_HUB_OFFLINE` | All Hugging Face downloads | When set to `1`, error instead of attempting downloads |
| `LOCAL_FILES_ONLY` | Unsloth backend, GLiNER | When set to `true`, skip network downloads and use local files only |
| `VLLM_CACHE_ROOT` | vLLM | Set the vLLM model cache directory |

!!! warning
    `LOCAL_FILES_ONLY` is not consistently supported across all backends.
    The HuggingFace training backend and vLLM do not respect it. Use
    `HF_HUB_OFFLINE=1` combined with a pre-populated `HF_HOME` cache for
    the most reliable offline experience.

To avoid the `kernels` package making network calls to the Kernels Hub entirely,
set the training attention backend to a built-in option:

=== "YAML"

    ```yaml
    training:
      attn_implementation: "sdpa"
    ```

=== "CLI"

    ```bash
    safe-synthesizer run --training__attn_implementation sdpa --url data.csv
    ```

=== "SDK"

    ```python
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_train(attn_implementation="sdpa")
    )
    ```

To prevent outbound classification requests, either set `NIM_ENDPOINT_URL`
to a local endpoint or disable classification:

=== "YAML"

    ```yaml
    replace_pii:
      globals:
        classify:
          enable_classify: false
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig

    pii_config = PiiReplacerConfig.get_default_config()
    pii_config.globals.classify.enable_classify = False

    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_replace_pii(config=pii_config)
    )
    ```

=== "Environment"

    ```bash
    # Point to a local endpoint instead of the default
    export NIM_ENDPOINT_URL="https://your-local-nim-endpoint"
    export NIM_API_KEY="your-api-key"  # pragma: allowlist secret
    ```

### VRAM Management

Both training and generation allocate GPU memory via `get_max_vram()`, which
defaults to 80% of available VRAM with a 2 GiB safety buffer subtracted.

!!! note
    `training.max_vram_fraction` exists in config but is not currently wired
    into either the training or generation paths. Both use the hardcoded 0.8
    default in `get_max_vram()`.

### Memory Cleanup

After training completes, the pipeline calls `gc.collect()` and
`torch.cuda.empty_cache()` to free GPU memory before generation starts.
If you run training and generation in separate invocations, memory is
managed independently.

### NER CPU Parallelism

NER processing uses multiple CPU processes. Control the count with:

```bash
export SAFE_SYNTHESIZER_CPU_COUNT=4
```

Defaults to the system CPU count.

### SentenceTransformer Downloads

The `distiluse-base-multilingual-cased-v2` model is downloaded from
Hugging Face Hub. If the download fails, text semantic similarity metrics
are silently skipped.
