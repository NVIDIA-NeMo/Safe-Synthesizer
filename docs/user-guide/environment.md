<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Environment and Runtime

How Safe Synthesizer interacts with GPUs, downloads models, and manages
resources. For troubleshooting runtime errors, see
[Troubleshooting](troubleshooting.md). For output quality and evaluation,
see [Data Quality](data-quality.md).

---

## Attention Backends

Safe Synthesizer enables changing the attention implementations for both training
and generation.

### Training: `attn_implementation`

The HuggingFace backend loads models with an attention implementation controlled
by `training.attn_implementation`. The default is `kernels-community/vllm-flash-attn3`,
which uses Flash Attention 3 via the
[HuggingFace Kernels Hub](https://huggingface.co/docs/kernels/index).

The `kernels` package is included in both the `cpu` and `cu128` extras, so it
is installed automatically. However, the Kernels Hub requires network access on
first use to download compiled attention kernels. In offline environments, the
backend falls back to `sdpa` (PyTorch scaled dot-product attention) and logs a
warning:

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
| `kernels-community/vllm-flash-attn3` | Flash Attention 3 via Kernels Hub (default) | `kernels` pip package + network |
| `kernels-community/flash-attn2` | Flash Attention 2 via Kernels Hub | `kernels` pip package + network |
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

---

## Network and Download Behavior

Safe Synthesizer downloads models, tokenizers, and other artifacts at various
pipeline stages. All model and tokenizer downloads go through
[Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

### What Gets Downloaded

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
export HF_HOME=/shared/cache/huggingface

safe-synthesizer run --config config.yaml --url data.csv
```

---

## Offline and Air-Gapped Environments

Several environment variables control network behavior:

| Variable | Scope | Effect |
|----------|-------|--------|
| `HF_HOME` | All Hugging Face downloads | Set the cache directory for downloaded models |
| `HF_HUB_OFFLINE` | All Hugging Face downloads | When set (e.g., `HF_HUB_OFFLINE=1`), errors instead of attempting downloads |
| `LOCAL_FILES_ONLY` | Unsloth backend, GLiNER | When set (e.g., `LOCAL_FILES_ONLY=true`), skips network downloads; local files only |
| `VLLM_CACHE_ROOT` | vLLM | Set the vLLM model cache directory |

!!! warning
    `LOCAL_FILES_ONLY` is not consistently supported across all backends.
    The HuggingFace training backend and vLLM do not respect it. Set
    `HF_HUB_OFFLINE` combined with a pre-populated `HF_HOME` cache for
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
to a local endpoint or disable classification. PII classify config is deeply
nested -- use YAML or SDK:

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

---

## VRAM Management

Both training and generation allocate GPU memory via `get_max_vram()`:

$$
\text{fraction} = \min\!\left(0.8,\;\frac{\text{free} - 2\,\text{GiB}}{\text{total}}\right)
$$

where \(\text{free}\) and \(\text{total}\) come from `torch.cuda.mem_get_info()`.
On a fresh 32 GiB GPU this yields \(0.8 \times 32 = 25.6\) GiB.

!!! note
    `training.max_vram_fraction` exists in config but is not currently wired
    into either the training or generation paths. Both use the hardcoded 0.8
    default in `get_max_vram()`.

### Memory Cleanup

After training completes, the pipeline calls `gc.collect()` and
`torch.cuda.empty_cache()` to free GPU memory before generation starts.
If you run training and generation in separate invocations, memory is
managed independently.

---

## NER CPU Parallelism

NER processing uses multiple CPU processes. Control the count with:

```bash
export SAFE_SYNTHESIZER_CPU_COUNT=4
```

Defaults to `max(1, cpu_count - 1)` (one CPU left free), further capped so
there are at least 1,000 records per worker.

---

## SentenceTransformer Downloads

The `distiluse-base-multilingual-cased-v2` model is downloaded from
Hugging Face Hub. If the download fails, text semantic similarity metrics
are silently skipped.
