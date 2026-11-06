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
| Inference endpoint and API key | [PII Replacement](#pii-replacement) · [Customizing PII Replacement](customize-pii-replacement.md) |
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
| `NSS_INFERENCE_ENDPOINT` | nss | `--inference-endpoint-url` | PII planning | -- (local vLLM server) | OpenAI-compatible PII inference endpoint. A non-loopback URL sends planning data to that service; see [where planning requests go](#where-planning-requests-go) | [PII appendix](#pii-replacement) |
| `NSS_INFERENCE_KEY` | nss | `--inference-api-key` | PII planning | -- | Inference credential for an explicit endpoint, such as the hosted NVIDIA endpoint; ignored by the local vLLM server | [PII appendix](#pii-replacement) |
| `NSS_INFERENCE_MODEL` | nss | `--inference-model-id` | PII planning | `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` locally; required with an endpoint | Model ID for PII inference; without an endpoint, selects the bundled local profile | [PII appendix](#pii-replacement) |
| `NSS_INFERENCE_TIMEOUT` | nss | `--inference-timeout-seconds` | PII planning | local profile's timeout, else `60` | Per-request inference timeout in seconds, including model reasoning | [PII appendix](#pii-replacement) |
| `NSS_INFERENCE_REQUEST_OPTIONS` | nss | `--inference-request-options` | PII planning | local profile's options, else none (server defaults) | JSON object of chat-completions fields sent with every request; overrides a profile's options field by field, and `null` drops a field | [PII appendix](#pii-replacement) |
| `NSS_INFERENCE_LOCAL_PROFILE` | nss | `--inference-local-profile` | PII planning | -- | Profile YAML path, or bundled profile name, for the local vLLM server Safe Synthesizer runs during plan discovery | [PII appendix](#pii-replacement) |
| `NSS_NEMOTRON_PERSONAS_PATH` | nss | -- | PII replacement | `~/.data-designer/managed-assets` | Directory containing downloaded Nemotron-Personas `datasets/<locale>.parquet` files; `replace_pii.sampler.nemotron_personas_path` takes precedence | [Nemotron-Personas](customize-pii-replacement.md#nemotron-personas) |
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

Automatic PII plan discovery uses an LLM. By default, Safe Synthesizer runs
`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` in a temporary local vLLM
server on the GPU, so no endpoint or key is needed. To choose a different
planning model or your own serving profile, see
[Customizing PII Replacement -- Planning LLM](customize-pii-replacement.md#planning-llm).

### Where planning requests go

Plan discovery sends the LLM column statistics, up to eight cell samples per
column, up to three sample rows of the columns a dependency could link, and
distinct values of dependency columns, such as sex or race, that need mapping
to the sampler's values. Where they go depends only on
`NSS_INFERENCE_ENDPOINT` (or `--inference-endpoint-url`):

- Unset: Safe Synthesizer starts its own vLLM server on this machine, and the
  data stays here. This is the default.
- A loopback URL (`localhost`, `127.0.0.0/8`, or `::1`): the data stays on this
  machine. Safe Synthesizer calls the server you run at that address, or, with
  `NSS_INFERENCE_LOCAL_PROFILE`, starts its own server there. To start that
  server yourself and reuse it across runs, see
  [Reuse one server across runs](customize-pii-replacement.md#reuse-one-server-across-runs).
- Any other URL: Safe Synthesizer sends the data to that service over HTTPS,
  for example `https://integrate.api.nvidia.com/v1` for the hosted NVIDIA
  service. Only use a service that is approved to receive the input data,
  including rows that may later be held out for evaluation.

The run log states which case applies when planning starts. Planning also runs
during evaluation when PII replacement is off and `evaluation.pii_replay_enabled`
is true, to find the PII the replay check looks for.

```bash title="PII inference environment" hl_lines="1 2 3"
export NSS_INFERENCE_ENDPOINT=https://integrate.api.nvidia.com/v1
export NSS_INFERENCE_MODEL=nvidia/nemotron-3-ultra-550b-a55b
export NSS_INFERENCE_KEY=<your key>
```

- The endpoint must use HTTPS unless it is a loopback address.
- With an endpoint, a model ID is required. It resolves from
  `--inference-model-id`, then `replace_pii.llm.model_id`, then
  `NSS_INFERENCE_MODEL`.
- `NSS_INFERENCE_KEY` (or `--inference-api-key`) is required by the hosted
  NVIDIA service; local OpenAI-compatible servers may be keyless. It is ignored,
  with a warning, while Safe Synthesizer runs its own local server.
- The endpoint and key are never stored in configuration or plan files, and
  are rejected in YAML.

If vLLM or a suitable GPU is unavailable and no endpoint is set, planning fails
with an error instead of falling back to a remote service.

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
