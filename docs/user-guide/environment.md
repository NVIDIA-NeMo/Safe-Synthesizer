<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Environment Variables

All environment variables that affect Safe Synthesizer behavior. For runtime
errors and OOM issues, see [Troubleshooting](troubleshooting.md). For output
quality and evaluation metrics, see [Data Quality](data-quality.md).

Synthesis parameters (`training.learning_rate`, `generation.num_records`, etc.)
are set via YAML, CLI flags, or the Python SDK -- not environment variables.
Environment variables control infrastructure: where artifacts go, how models
are cached, and which network endpoints are used.

---

## Quick Reference

| Variable | Default | Effect |
|----------|---------|--------|
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face model cache directory |
| `HF_HUB_OFFLINE` | unset | Set to `1` to error instead of attempting downloads |
| `LOCAL_FILES_ONLY` | unset | Set to `true` to skip network downloads (Unsloth + GLiNER only) |
| `VLLM_CACHE_ROOT` | vLLM default | vLLM model cache directory |
| `VLLM_ATTENTION_BACKEND` | `auto` | Generation attention implementation |
| `NIM_ENDPOINT_URL` | unset | NIM/OpenAI-compatible endpoint for PII column classification; classification skipped when unset |
| `NIM_API_KEY` | unset | API key for the NIM endpoint |
| `NIM_MODEL_ID` | `qwen/qwen2.5-coder-32b-instruct` | Model ID for PII column classification |
| `SAFE_SYNTHESIZER_CPU_COUNT` | `max(1, cpu_count - 1)` | CPU worker count for NER processing |

---

## Hugging Face Cache

All model and tokenizer downloads go through
[Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).
The following variables control where downloads are stored and whether the
network is used.

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

!!! warning
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

!!! warning
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
unset, classification is skipped silently (the exception is caught and the
pipeline falls back to `describe_field()`). Set this to enable classification:

```bash
export NIM_ENDPOINT_URL="https://your-local-nim-endpoint"
export NIM_API_KEY="your-api-key"  # pragma: allowlist secret
```

To disable column classification entirely instead of pointing it at a local
endpoint, use the `replace_pii.globals.classify.enable_classify` config option.
PII classify config is deeply nested -- use YAML or SDK:

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
