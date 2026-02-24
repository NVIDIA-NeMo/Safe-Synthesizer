<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Troubleshooting

This guide covers common issues, error messages, and configuration pitfalls
you may encounter when using NeMo Safe Synthesizer.

---

## Attention Backends

Safe Synthesizer uses configurable attention implementations for both training
and generation. Misconfiguration here is a common source of startup failures.

### Training: `attn_implementation`

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

To explicitly choose a backend, set it in your config or via CLI:

=== "YAML"

    ```yaml
    training:
      attn_implementation: "sdpa"
    ```

=== "CLI"

    ```bash
    safe-synthesizer run --training__attn_implementation sdpa --url data.csv
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

### Generation: `attention_backend`

The vLLM generation engine uses a separate attention backend setting. Defaults to
`"auto"`, which lets vLLM auto-select the best available backend.

In vLLM 0.11.x, this is set via the `VLLM_ATTENTION_BACKEND` environment variable.
Safe Synthesizer handles this automatically when `generation.attention_backend` is
configured.

```yaml
generation:
  attention_backend: "FLASH_ATTN"
```

Common values: `FLASHINFER`, `FLASH_ATTN`, `TORCH_SDPA`, `TRITON_ATTN`, `FLEX_ATTENTION`.

!!! note
    Leave this as `"auto"` unless you have a specific reason to override it.
    vLLM's auto-detection works well in most environments.

---

## Network and Download Behavior

Safe Synthesizer downloads models, tokenizers, and other artifacts from the
internet at various pipeline stages. If you are running in a network-restricted
environment, understanding these download points is critical.

### What Gets Downloaded

Training:

- Model weights, config, and tokenizer via `AutoModelForCausalLM.from_pretrained()`,
  `AutoConfig.from_pretrained()`, and `AutoTokenizer.from_pretrained()` -- all
  fetched from HuggingFace Hub on first use
- When `attn_implementation` starts with `kernels-community/`, the `kernels`
  package downloads compiled attention kernels from HuggingFace Hub
- Unsloth backend also downloads models via `FastLanguageModel.from_pretrained()`

Generation:

- vLLM downloads the base model on initialization if not already cached

PII replacement:

- GLiNER downloads its NER model on first use

Evaluation:

- `SentenceTransformer("distiluse-base-multilingual-cased-v2")` is downloaded in
  text semantic similarity, attribute inference protection, and membership inference
  protection components. The download retries up to 10 times with exponential
  backoff (up to ~6 minutes total)
- Tiktoken downloads tokenizer encoding files on first use

!!! warning
    All of these downloads happen silently on first use. If your first run is in
    an environment without internet access, you will see connection errors at
    whichever stage tries to download first.

### Pre-Caching Models

To avoid runtime downloads, run the pipeline once in an environment with internet
access. All downloaded artifacts are stored under `HF_HOME`
(defaults to `~/.cache/huggingface`). You can then copy or mount this cache
directory in your target environment.

```bash
# Set a shared cache location
export HF_HOME=/shared/cache/huggingface

# Run once to populate the cache
safe-synthesizer run --config config.yaml --url data.csv

# The cache at /shared/cache/huggingface now has everything needed
```

---

## Offline and Air-Gapped Environments

Several environment variables control network behavior:

| Variable | Scope | Effect |
|----------|-------|--------|
| `HF_HOME` | All HuggingFace downloads | Set the cache directory for downloaded models |
| `HF_HUB_OFFLINE=1` | All HuggingFace downloads | Error instead of attempting downloads |
| `LOCAL_FILES_ONLY=true` | Unsloth backend, GLiNER | Skip network downloads, use local files only |
| `VLLM_CACHE_ROOT` | vLLM | Set the vLLM model cache directory |

!!! warning
    `LOCAL_FILES_ONLY` is not consistently supported across all backends.
    The HuggingFace training backend and vLLM do not respect it. Use
    `HF_HUB_OFFLINE=1` combined with a pre-populated `HF_HOME` cache for
    the most reliable offline experience.

To avoid the `kernels` package making network calls to the Kernels Hub entirely,
set the training attention backend to a built-in option:

```yaml
training:
  attn_implementation: "sdpa"
```

---

## GPU, VRAM, and CUDA

### VRAM Management

Safe Synthesizer reserves GPU memory conservatively:

- `training.max_vram_fraction` (default `0.80`) -- fraction of total VRAM to allocate
- An additional 2 GiB safety buffer is subtracted from free VRAM before allocation
- vLLM's `gpu_memory_utilization` is derived from the same calculation

### Out of Memory During Training

If you see `torch.cuda.OutOfMemoryError` during training:

1. Reduce `training.batch_size` (default is `1`, effective batch size includes
   `gradient_accumulation_steps` which defaults to `8`)
2. Reduce `training.max_seq_length` or let `rope_scaling_factor` auto-resolve
   to a smaller value
3. Enable quantization with `load_in_4bit: true` or `load_in_8bit: true`
4. Lower `training.max_vram_fraction` if other processes share the GPU

!!! note
    There is a known memory leak in the HuggingFace Trainer evaluation loop
    that can cause OOM on large datasets. The codebase includes a workaround,
    but extremely large evaluation sets may still cause issues.

### Out of Memory During Generation

vLLM manages its own memory pool. If generation OOMs:

1. Lower `training.max_vram_fraction` to give vLLM less memory
2. Ensure no other processes hold GPU memory (training cleanup should release it)

### Out of Memory During Evaluation

Evaluation components that can be memory-intensive:

- PCA computation in deep structure analysis -- skipped automatically if the
  matrix is too small, but can OOM on very wide datasets
- Histogram binning uses `doane` method instead of `fd` to reduce memory usage

### No GPU Detected

If `torch.cuda.is_available()` returns `False`:

1. Verify NVIDIA drivers: `nvidia-smi`
2. Verify CUDA toolkit: `nvcc --version`
3. Verify PyTorch CUDA build: `python -c "import torch; print(torch.cuda.is_available())"`
4. Ensure you installed the `cu128` extras, not `cpu`

The Unsloth backend requires a GPU and raises a `RuntimeError` immediately
if none is found. Switch to the HuggingFace backend for CPU-only environments
(useful for development, not recommended for production training).

### Memory Cleanup

After training completes, the pipeline calls `gc.collect()` and
`torch.cuda.empty_cache()` to free GPU memory before generation starts.
If you run training and generation in separate invocations, memory is
managed independently.

---

## Configuration Gotchas

### Surprising Defaults

Several defaults may not match your expectations:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `training.batch_size` | `1` | Effective batch = `batch_size` x `gradient_accumulation_steps` (8) |
| `training.validation_ratio` | `0.0` | No validation split by default |
| `data.max_holdout` | `2000` | Caps holdout set even if percentage would be larger |
| `data.random_state` | `None` | Auto-generates a random seed -- set explicitly for reproducibility |
| `generation.num_records` | `1000` | May be too small for production use |
| `generation.temperature` | `0.9` | Relatively high randomness |

### Auto-Resolved Parameters

Many parameters accept `"auto"` and are resolved at runtime by the
AutoConfigResolver:

- `training.rope_scaling_factor` -- estimated from dataset token counts using
  heuristics (4 chars per token for text, 1 token per digit). Can underestimate
  for complex or multilingual data.
- `training.num_input_records_to_sample` -- derived from `rope_scaling_factor * 25000`
- `training.use_unsloth` -- resolves to `true` unless DP is enabled
- `privacy.delta` -- computed from record count

Use `safe-synthesizer config validate --config config.yaml` to see how `"auto"`
values resolve for your configuration.

### Common Validation Errors

`order_training_examples_by` requires `group_training_examples_by`:

: If you set `data.order_training_examples_by` without also setting
  `data.group_training_examples_by`, validation will fail. Ordering only
  makes sense within groups.

Unsupported dataset file extensions:

: The dataset registry supports `.csv`, `.txt`, `.json`, `.jsonl`, and `.parquet`.
  Other formats raise a `ValueError`.

---

## Differential Privacy

DP training has strict requirements. Violating them produces errors that may
not immediately point to the root cause.

### Requirements

- DP and Unsloth are mutually exclusive. If `privacy.dp_enabled` is `true`,
  `use_unsloth` must be `false` or `"auto"` (which resolves to `false`).
- `training.max_sequences_per_example` must be `1` when DP is enabled.
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

Time series synthesis has additional validation and generation requirements.

### Configuration Requirements

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

### Model Downloads

The PII replacer downloads the GLiNER NER model on first use. This respects
the `LOCAL_FILES_ONLY` environment variable, but has no retry logic -- if the
download fails, it raises a `RuntimeError`.

Pre-download the model by running PII replacement once in an environment with
internet access, or set `LOCAL_FILES_ONLY=true` after the model is cached.

### NER Processing

- NER has a `max_runtime_seconds` timeout. Chunks that exceed it are dropped
  with a warning in the logs.
- If the column classifier fails to initialize or classify, it falls back to
  default entity types silently. Check logs for classification errors if PII
  replacement seems to use unexpected entity types.

### NIM Endpoint

For external PII detection via NIM, set these environment variables:

```bash
export NIM_ENDPOINT_URL="https://your-nim-endpoint"
export NIM_API_KEY="your-api-key"  # pragma: allowlist secret
```

### CPU Parallelism

NER processing uses multiple CPU processes. Control the count with:

```bash
export SAFE_SYNTHESIZER_CPU_COUNT=4
```

Defaults to the system CPU count.

---

## Evaluation

### Minimum Data Requirements

Several evaluation metrics have minimum data requirements:

| Metric | Minimum | Behavior if Unmet |
|--------|---------|-------------------|
| Holdout / text + privacy metrics | 200 records | Raises `ValueError` |
| Attribute Inference Attack | 3+ columns | Silently skipped, warning logged |
| PCA (deep structure) | 2x2 matrix | Silently skipped, warning logged |

### Silent Failures

Many evaluation components catch errors and return `UNAVAILABLE` grades instead
of failing the pipeline. If your evaluation report shows missing or `UNAVAILABLE`
metrics:

1. Check the logs for warnings and exceptions
2. Verify you have enough records (>= 200) and columns (>= 3)
3. Verify the SentenceTransformer model downloaded successfully

### SentenceTransformer Downloads

The `distiluse-base-multilingual-cased-v2` model is downloaded with up to
10 retries and exponential backoff (total wait up to ~6 minutes). If all
retries fail, text semantic similarity metrics are silently skipped.

### Report Truncation

SQS reports are limited to `sqs_report_columns=250` columns and
`sqs_report_rows=5000` rows by default. Larger datasets are silently
truncated in the HTML report. Adjust these in `evaluation` config if needed.

### Optional Dependencies

FAISS is optional for membership inference protection. If not installed, some
functionality silently degrades. Install it for full evaluation coverage:

```bash
pip install faiss-cpu  # or faiss-gpu for GPU-accelerated search
```

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

Generation failures -- context length exceeded, rejection sampling exhausted,
or invalid fraction threshold reached.

Checklist:

1. Check `generation.patience` and `generation.invalid_fraction_threshold`
2. Verify the adapter trained successfully (check training logs)
3. If context length is exceeded, increase `training.rope_scaling_factor`
   or reduce record size
4. Increase `training.num_input_records_to_sample` if generation stops
   immediately with zero valid records

### InternalError

Library bugs. If you encounter this error through documented interfaces,
please file an issue on GitHub.

---

## Quick Reference

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "kernels package not installed" warning | `kernels` pip package missing | `pip install kernels` or set `attn_implementation: sdpa` |
| `ConnectionError` during startup | No internet, model not cached | Pre-cache models, set `HF_HUB_OFFLINE=1` after caching |
| `torch.cuda.OutOfMemoryError` in training | VRAM exhausted | Reduce `batch_size`, enable `load_in_4bit`, lower `max_vram_fraction` |
| `torch.cuda.OutOfMemoryError` in evaluation | Large dataset + PCA/binning | Reduce dataset size or column count |
| "Cannot use unsloth without GPU" | No CUDA device | Use HuggingFace backend or install CUDA drivers |
| "max_sequences_per_example must be 1" | Incompatible DP config | Set `max_sequences_per_example: 1` or `"auto"` |
| "Unsloth is currently not compatible with DP" | Mutual exclusion | Set `use_unsloth: false` for DP training |
| "Unable to determine a noise multiplier" | Epsilon too low for dataset | Increase `epsilon` or add more training records |
| Generation stops with 0 valid records | Model underfitting or schema mismatch | Increase `num_input_records_to_sample`, check training logs |
| "Number of tokens exceeds context length" | Records too long for model | Increase `rope_scaling_factor` or reduce record size |
| Evaluation metrics show UNAVAILABLE | Too few records or columns | Ensure >= 200 records, >= 3 columns for privacy metrics |
| PII uses default entities unexpectedly | Column classifier failed | Check logs for classification errors; set entities explicitly |
| "timestamp_column has missing values" | Dirty time series data | Clean NaN/null values from the timestamp column |
| "groups must have same start timestamp" | Inconsistent time series groups | Align group start/stop timestamps in preprocessing |

---

## Environment Variables

A consolidated reference of environment variables that affect runtime behavior.

| Variable | Affects | Default | Purpose |
|----------|---------|---------|---------|
| `HF_HOME` | All stages | `~/.cache/huggingface` | HuggingFace cache directory |
| `HF_HUB_OFFLINE` | All stages | unset | Set to `1` to error instead of downloading |
| `LOCAL_FILES_ONLY` | Unsloth, GLiNER | `False` | Skip network downloads for supported backends |
| `VLLM_ATTENTION_BACKEND` | Generation | unset (auto) | Override vLLM attention backend |
| `VLLM_CACHE_ROOT` | Generation | system default | vLLM model cache directory |
| `NSS_ARTIFACTS_PATH` | All stages | `./safe-synthesizer-artifacts` | Default artifact output path |
| `NSS_LOG_FORMAT` | All stages | `plain` | `json` or `plain` |
| `NSS_LOG_FILE` | All stages | under run dir | Log file path |
| `NSS_WANDB_MODE` | Training | `disabled` | WandB mode (`online`, `offline`, `disabled`) |
| `WANDB_API_KEY` | Training | unset | Required for WandB `online` mode |
| `SAFE_SYNTHESIZER_CPU_COUNT` | PII replacement | system CPU count | Number of CPU processes for NER |
| `NIM_ENDPOINT_URL` | PII replacement | unset | NIM endpoint for external PII detection |
| `NIM_API_KEY` | PII replacement | `not-needed` | NIM API key |
