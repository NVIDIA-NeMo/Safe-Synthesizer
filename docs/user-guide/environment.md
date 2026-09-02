<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Environment Variables

Reference for infrastructure settings: artifact paths, logging, model caches,
network endpoints, and third-party library behavior.

Synthesis parameters (`training.learning_rate`, `generation.num_records`, etc.)
are set via YAML, CLI flags, or the Python SDK -- not environment variables.
See [Configuration Reference](configuration.md) for parameter tables and
[Configuration Precedence](configuration.md#configuration-precedence) for how
YAML, CLI, and SDK layers combine.

For runtime errors and OOM issues, see [Program Runtime](troubleshooting.md).
For output quality and evaluation metrics, see
[Synthetic Data Quality](evaluating-data.md).

---

## At a glance

| Task | Start here |
|------|------------|
| Run offline or air-gapped | [HF cache and offline](#hugging-face-cache-and-offline) · [Running in Offline Environments](running.md#running-in-offline-environments) |
| Docker / container mounts | [Containers](#containers) · [Docker](docker.md) |
| Logging and WandB | [Running -- Logging and Experiment Tracking](running.md#logging-and-experiment-tracking) |
| Inference endpoint and API key (unused) | [PII Replacement](#pii-replacement) · [PII Replacement](../product-overview/pii_replacement.md) |
| Disable telemetry | [Telemetry](#telemetry) |
| Resolve CLI vs env vs defaults | [Precedence](#precedence) |

---

## Master reference table

Grouped by the `Category` column -- `nss`-native settings first, then
`telemetry`, `third-party`, `container`, and `internal`.

| Variable | Category | CLI flag | Read by | Default | Purpose | Details |
|----------|----------|----------|---------|---------|---------|---------|
| `NSS_CONFIG` | nss | `--config` | CLI | -- | Path to YAML config file | [Configuration Reference](configuration.md) |
| `NSS_ARTIFACTS_PATH` | nss | `--artifact-path` | CLI | `./safe-synthesizer-artifacts` | Base directory for run artifacts | [Running -- Artifacts](running.md#artifacts-and-output) |
| `NSS_LOG_FORMAT` | nss | `--log-format` | CLI / observability | auto (`plain` on TTY, else `json`) | Console log format | [Running -- Log Format](running.md#log-format) |
| `NSS_LOG_FILE` | nss | `--log-file` | CLI / observability | run log under workdir | Path to log file | [Running -- Logging](running.md#logging-and-experiment-tracking) |
| `NSS_LOG_COLOR` | nss | `--log-color` / `--no-log-color` | CLI / observability | auto (TTY) | Colorize console output | [Running -- Log Format](running.md#log-format) |
| `NSS_LOG_LEVEL` | nss | `--verbose` (0–2) | observability | `INFO` | Log level (`DEBUG`, `DEBUG_DEPENDENCIES`, etc.) | Set via verbosity, not a direct CLI flag |
| `NSS_DATASET_REGISTRY` | nss | `--dataset-registry` | CLI | -- | Dataset registry YAML path or URL | [Running -- Dataset Registry](running.md#dataset-registry) |
| `NSS_INFERENCE_ENDPOINT` | nss | `--inference-endpoint-url` | PII planning | NVIDIA integrate URL | OpenAI-compatible PII inference endpoint | [PII appendix](#pii-replacement) |
| `NSS_INFERENCE_KEY` | nss | `--inference-api-key` | PII planning | -- | Runtime-only inference credential; required by the default hosted endpoint | [PII appendix](#pii-replacement) |
| `NSS_INFERENCE_MODEL` | nss | `--inference-model-id` | PII planning | `nvidia/nemotron-3-ultra-550b-a55b` | Model ID served by the PII inference endpoint | [PII appendix](#pii-replacement) |
| `NSS_WANDB_MODE` | nss | `--wandb-mode` | WandB | `disabled` | WandB run mode | Alias for `WANDB_MODE` |
| `NSS_WANDB_PROJECT` | nss | `--wandb-project` | WandB | -- | WandB project name | Alias for `WANDB_PROJECT` |
| `NSS_WANDB_UPLOAD_EVALUATION_REPORT` | nss | `--wandb-upload-evaluation-report` / `--no-wandb-upload-evaluation-report` | WandB | `true` | Upload final evaluation HTML and artifact | Set to `false` to skip HTML and artifact publishing; summary metrics and the scorecard remain enabled |
| `NEMO_TELEMETRY_ENABLED` | telemetry | `--emit_telemetry` | telemetry | `true` | Enable anonymous usage telemetry | Also `emit_telemetry` in YAML; see [Telemetry](#telemetry) |
| `HF_HOME` | third-party | -- | Hugging Face Hub | platform cache dir | Root directory for HF downloads | [HF appendix](#hugging-face-cache-and-offline) |
| `HF_HUB_OFFLINE` | third-party | `--enable-huggingface-remote` / `--disable-huggingface-remote` | Hugging Face Hub | unset | Fail if a model is not cached | Preferred offline gate; CLI flag also sets `TRANSFORMERS_OFFLINE` |
| `VLLM_CACHE_ROOT` | third-party | -- | vLLM | `~/.cache/vllm` | vLLM model cache directory | [vLLM appendix](#vllm-and-attention) |
| `VLLM_ATTENTION_BACKEND` | third-party | -- | vLLM | auto | Override attention implementation | [vLLM appendix](#vllm-and-attention) |
| `WANDB_MODE` | third-party | `--wandb-mode` | WandB | `disabled` | WandB run mode | Same as `NSS_WANDB_MODE` |
| `WANDB_PROJECT` | third-party | `--wandb-project` | WandB | -- | WandB project name | Same as `NSS_WANDB_PROJECT` |
| `WANDB_API_KEY` | third-party | -- | WandB | -- | WandB authentication | Required for online logging |
| `NVIDIA_VISIBLE_DEVICES` | container | -- | NVIDIA runtime | all visible GPUs | Limit GPUs inside a container | [Containers](#containers) · [Docker -- GPU Access](docker.md#gpu-access) |
| `NEMO_TELEMETRY_ENDPOINT` | internal | -- | telemetry | NVIDIA default | Override telemetry upload URL | [Telemetry](#telemetry) |
| `NEMO_SESSION_PREFIX` | internal | -- | telemetry | -- | Prefix for telemetry session IDs | [Telemetry](#telemetry) |
| `NEMO_JOB_ID` | internal | -- | evaluation reports | -- | Cluster job ID in multimodal reports | [Internal](#internal-and-cluster) |

---

## Precedence

### Infrastructure (CLISettings)

For artifact paths, logging, WandB overrides, and the runtime flags
(`--inference-*`, `--enable-huggingface-remote` / `--disable-huggingface-remote`):

1. CLI flags
2. Environment variables
3. Built-in defaults

WandB accepts both `NSS_WANDB_*` and `WANDB_*` names; CLI `--wandb-mode` and
`--wandb-project` override either.

### Synthesis parameters

YAML fields, CLI `--section__field` overrides, and SDK builder calls follow
[Configuration Precedence](configuration.md#configuration-precedence) -- not
the order above.

### Telemetry precedence

`--emit_telemetry` / `emit_telemetry` in YAML override `NEMO_TELEMETRY_ENABLED`
when explicitly set. When unset, the env var defaults to enabled.

---

## Hugging Face cache and offline

Downloads go through
[Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).
For a step-by-step offline workflow, see
[Running in Offline Environments](running.md#running-in-offline-environments)
and [Docker -- Offline and Air-Gapped Environments](docker.md#offline-and-air-gapped-environments).

### `HF_HOME`

Root cache for model weights, tokenizers, compiled attention kernels,
evaluation SentenceTransformer weights, and other Hub assets.

```bash
export HF_HOME=/shared/cache/huggingface
```

### `HF_HUB_OFFLINE`

`HF_HUB_OFFLINE=1` tells Hugging Face Hub to refuse network access. It is the
canonical offline switch: huggingface_hub honors it globally. Pair it with a
pre-populated `HF_HOME`.

```bash
export HF_HUB_OFFLINE=1
```

Set it before the process starts. huggingface_hub reads the value once, when it
is first imported, and caches it -- changing it later has no effect for that
process. For the CLI, export it before launching `safe-synthesizer`. When
driving the pipeline programmatically, set it before importing
`nemo_safe_synthesizer`.

### `--enable-huggingface-remote` / `--disable-huggingface-remote`

CLI shorthand for the switch above, with no separate NSS env var:

- `--disable-huggingface-remote` -- offline run; sets `HF_HUB_OFFLINE=1` and
  `TRANSFORMERS_OFFLINE=1`.
- `--enable-huggingface-remote` -- online run; sets both to `0`, overriding any
  inherited offline environment.
- Default (neither flag) -- the environment is left untouched: the run inherits
  `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` if set, and otherwise allows remote
  downloads. The effective default is `--enable-huggingface-remote`.

The CLI applies the flag before huggingface_hub loads, so the flag always wins
over an inherited environment value. For env-based control, set `HF_HUB_OFFLINE`
directly.

```bash
safe-synthesizer run --disable-huggingface-remote ...
```

!!! warning "Models must be cached"
    Offline mode requires all models to already be present in `HF_HOME`.
    Loading fails if a required model is not cached.

### Pre-caching models

Run once with network access, then copy or mount the populated cache. Typical
first-run downloads include training weights, evaluation embeddings, and the
vLLM base model.

!!! warning "Silent downloads on first use"
    Downloads happen on first use. In an air-gapped environment, the first
    missing asset fails at the stage that needs it.

See [Running in Offline Environments](running.md#running-in-offline-environments)
for the full pre-cache checklist.

---

## PII Replacement

PII replacement v3 environment guidance will be added in a later update.

---

## vLLM and attention

### `VLLM_CACHE_ROOT`

Directory for vLLM's internal model cache (default `~/.cache/vllm`).

```bash
export VLLM_CACHE_ROOT=/shared/cache/vllm
```

### `VLLM_ATTENTION_BACKEND`

Override the vLLM attention implementation. Safe Synthesizer sets this from
`generation.attention_backend` when configured; leave unset to use vLLM
auto-detection.

```bash
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
```

Common values: `FLASHINFER`, `FLASH_ATTN`, `TORCH_SDPA`, `TRITON_ATTN`,
`FLEX_ATTENTION`. See [Running -- Attention Backends](running.md#attention-backends).

---

## Telemetry

### `NEMO_TELEMETRY_ENABLED`

Whether anonymous train/generate telemetry is sent. Defaults to enabled.

```bash
export NEMO_TELEMETRY_ENABLED=false
```

Also disable per run with `--emit_telemetry false` or `emit_telemetry: false`
in YAML. Explicit config/CLI values override the env var.

### `NEMO_TELEMETRY_ENDPOINT` and `NEMO_SESSION_PREFIX`

Override the telemetry upload endpoint or prefix session IDs. Env-only; no CLI
equivalent. Intended for controlled test environments.

---

## Containers

Common bind-mount targets when running in Docker:

| Variable | Typical value | Why |
|----------|---------------|-----|
| `HF_HOME` | `/workspace/.hf_cache` | Persist Hub downloads across runs |
| `HF_HUB_OFFLINE` | `1` | Air-gapped runs after pre-caching |
| `VLLM_CACHE_ROOT` | `/workspace/.vllm_cache` | Persist vLLM cache |
| `NSS_ARTIFACTS_PATH` | `/workspace/artifacts` | Write artifacts to a volume |
| `NSS_LOG_FORMAT` | `json` | Structured logs in non-TTY containers |
| `NVIDIA_VISIBLE_DEVICES` | `0` or `all` | GPU selection inside the container |

See [Docker](docker.md) for mount paths, secrets, GPU flags, and mise container tasks.
shortcuts.

---

## Internal and cluster

Advanced env-only settings without CLI equivalents:

| Variable | Purpose |
|----------|---------|
| `NSS_OPT_BUCKET` | S3 bucket for optional NER optimization artifacts |
| `NSS_OPT_CACHE_DIR` | Local cache directory for NER optimization downloads |
| `NEMO_JOB_ID` | Cluster job ID attached to multimodal evaluation reports |

---

## Related guides

- [Running Safe Synthesizer](running.md) -- pipeline execution, CLI commands, offline workflow
- [Configuration Reference](configuration.md) -- synthesis parameter tables and precedence
- [Docker](docker.md) -- container setup, caches, and secrets
- [Program Runtime](troubleshooting.md) -- runtime errors and OOM fixes
