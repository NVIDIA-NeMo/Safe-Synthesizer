<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Environment Variables

All environment variables that affect Safe Synthesizer behavior. For runtime
errors and OOM issues, see [Program Runtime](troubleshooting.md). For output
quality and evaluation metrics, see [Synthetic Data Quality](evaluating-data.md).

Synthesis parameters (`training.learning_rate`, `generation.num_records`, etc.)
are set via YAML, CLI flags, or the Python SDK -- not environment variables.
Environment variables control infrastructure: where artifacts go, how models
are cached, and which network endpoints are used.

---

## NSS Variables

| Variable | CLI flag | Purpose |
|----------|----------|---------|
| `NSS_CONFIG` | `--config` | Path to YAML config file |
| `NSS_ARTIFACTS_PATH` | `--artifact-path` | Default artifact path |
| `NSS_LOG_FORMAT` | `--log-format` | Log format (`json` or `plain`) |
| `NSS_LOG_FILE` | `--log-file` | Log file path |
| `NSS_LOG_COLOR` | `--log-color` / `--no-log-color` | Colorize console output (auto-detected from TTY) |
| `NSS_LOG_LEVEL` | -- | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, or `DEBUG_DEPENDENCIES` |
| `NSS_DATASET_REGISTRY` | `--dataset-registry` | Dataset registry YAML path/URL |
| `NSS_WANDB_MODE` | `--wandb-mode` | WandB mode (alias for `WANDB_MODE`) |
| `NSS_WANDB_PROJECT` | `--wandb-project` | WandB project name (alias for `WANDB_PROJECT`) |
| `NIM_ENDPOINT_URL` | -- | LLM endpoint for PII column classification |
| `NIM_API_KEY` | -- | API key (optional -- only for direct endpoints) |
| `NIM_MODEL_ID` | -- | Column classification model ID |
| `LOCAL_FILES_ONLY` | -- | Set to `true` for offline mode (Unsloth, GLiNER) |
| `SAFE_SYNTHESIZER_CPU_COUNT` | -- | NER CPU processes |

---

## Third-Party Variables

| Variable | Read by | Purpose |
|----------|---------|---------|
| `HF_HOME` | Hugging Face Hub | Cache directory for model downloads |
| `HF_HUB_OFFLINE` | Hugging Face Hub | Set to `1` to error instead of downloading |
| `VLLM_ATTENTION_BACKEND` | vLLM | Override attention backend |
| `VLLM_CACHE_ROOT` | vLLM | vLLM internal cache directory (defaults to `~/.cache/vllm`) |
| `WANDB_MODE` | WandB | Mode (`online`, `offline`, `disabled`) |
| `WANDB_PROJECT` | WandB | Project name |
| `WANDB_API_KEY` | WandB | API key for authentication |

---

## Precedence

Exactly what avenues of configuration are available, and thus how precedence is resolved, depends on how you run the pipeline. Settings are resolved in this order, from highest (first) to lowest priority (last).

Synthesis parameters:

- CLI: CLI flags > dataset registry overrides > YAML config file > defaults
- SDK: Python SDK builder calls > YAML config file > defaults

Infrastructure settings (artifact path, logging, WandB):

1. CLI flags (`--artifact-path`, `--log-format`, etc.)
2. Environment variables (`NSS_ARTIFACTS_PATH`, `NSS_LOG_FORMAT`, etc.)
3. Built-in defaults

---

## Hugging Face Cache

All model and tokenizer downloads go through
[Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).
The following variables control where downloads are stored and whether the
network is used. For a step-by-step offline setup guide, see
[Running in Offline Environments](running.md#running-in-offline-environments).

### `HF_HOME`

Sets the root cache directory for all Hugging Face downloads -- model weights,
tokenizers, compiled attention kernels, and the SentenceTransformer used for
evaluation.

```bash
export HF_HOME=/shared/cache/huggingface
```

### Pre-Caching Models

To avoid runtime downloads, run the pipeline once in an environment with
internet access, then copy or mount the populated cache in your target
environment:

```bash
export HF_HOME=/shared/cache/huggingface
safe-synthesizer run --config config.yaml --url data.csv
```

What gets downloaded on first use:

- Model weights, config, and tokenizer (all backends, via HF Hub)
- Compiled attention kernels when `training.attn_implementation` starts with
  `kernels-community/`
- GLiNER NER model (PII replacement)
- `distiluse-base-multilingual-cased-v2` (evaluation semantic similarity)
- vLLM base model (generation)

!!! warning "Silent downloads on first use"
    All downloads happen silently on first use. If the first run is in an
    environment without internet access, connection errors will appear at
    whichever pipeline stage tries to download first.

### `HF_HUB_OFFLINE`

When set, prevents all Hugging Face Hub network requests. Any attempt to
access a model that is not already cached raises an error immediately.

```bash
export HF_HUB_OFFLINE=1
```

Prefer this over `LOCAL_FILES_ONLY` for the most reliable offline experience --
see the warning under `LOCAL_FILES_ONLY` below.

### `LOCAL_FILES_ONLY`

Skips network downloads for the Unsloth backend and GLiNER. Not respected by
the HuggingFace training backend or vLLM.

```bash
export LOCAL_FILES_ONLY=true
```

!!! warning "Partial offline support"
    `LOCAL_FILES_ONLY` is not consistently supported across all backends.
    Set `HF_HUB_OFFLINE=1` combined with a pre-populated `HF_HOME` cache
    for the most reliable offline experience.

### `VLLM_CACHE_ROOT`

Sets the vLLM model cache directory.

```bash
export VLLM_CACHE_ROOT=/shared/cache/vllm
```

---

## Attention and Compute

GPU attention backend selection for the vLLM generation engine.

### `VLLM_ATTENTION_BACKEND`

Controls the attention implementation used by the vLLM generation engine.
Safe Synthesizer sets this automatically when `generation.attention_backend`
is configured. Leave it unset unless you have a specific reason to override
vLLM's auto-detection.

```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

Common values: `FLASHINFER`, `FLASH_ATTN`, `TORCH_SDPA`, `TRITON_ATTN`,
`FLEX_ATTENTION`.

---

## PII and NER

NIM endpoint, API keys, and CPU parallelism for PII detection.

### `NIM_ENDPOINT_URL`

The NIM/OpenAI-compatible endpoint used for PII column classification. When
unset, an error is logged and the pipeline falls back to NER-only detection.
Set this to enable LLM-based column classification:

```bash
export NIM_ENDPOINT_URL="https://your-local-nim-endpoint"
export NIM_API_KEY="your-api-key"  # pragma: allowlist secret
```

To disable column classification entirely instead of pointing it at a local
endpoint, use the `replace_pii.globals.classify.enable_classify` config option.
PII classify config is deeply nested -- use YAML or SDK:

=== "Config reference"

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

### `NIM_API_KEY`

API key for the NIM endpoint. Required when `NIM_ENDPOINT_URL` points to an
authenticated endpoint.

### `NIM_MODEL_ID`

Model ID sent to the NIM endpoint for PII column classification. Defaults to
`qwen/qwen2.5-coder-32b-instruct`.

### `SAFE_SYNTHESIZER_CPU_COUNT`

Controls the number of CPU worker processes used for NER (PII replacement).

```bash
export SAFE_SYNTHESIZER_CPU_COUNT=4
```

Defaults to `max(1, cpu_count - 1)` (one CPU left free), further capped so
there are at least 1,000 records per worker.

---

- [Running Safe Synthesizer](running.md) -- pipeline execution, CLI commands, and artifacts
- [Configuration Reference](configuration.md) -- parameter tables
- [Program Runtime](troubleshooting.md) -- runtime errors and OOM fixes
