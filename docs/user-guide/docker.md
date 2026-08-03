<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Running in Docker

Run the Safe Synthesizer CLI from the published GPU runtime image. You do not
need a source checkout or local Python installation. The image contains the
installed runtime; it does **not** contain input data or a workload
configuration. Supply those at runtime.

## Select an image tag

The public image is:

```text
ghcr.io/nvidia-nemo/safe-synthesizer
```

Use `latest-cu129` to evaluate the current CUDA 12.9 release:

```bash
docker pull ghcr.io/nvidia-nemo/safe-synthesizer:latest-cu129
```

For a reproducible workload, replace that tag with an approved versioned
`<version>-cu129` tag or, preferably, pin the resolved manifest digest:

```text
ghcr.io/nvidia-nemo/safe-synthesizer:<version>-cu129
ghcr.io/nvidia-nemo/safe-synthesizer@sha256:<digest>
```

Release tags can be easier to audit; a digest identifies immutable image
content. Keep the `cu129` suffix when selecting a version tag because it names
the CUDA dependency variant.

## Prerequisites

- Docker with GPU support
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed and configured
- An NVIDIA driver compatible with the image's CUDA 12.9 libraries
- An NVIDIA GPU (A100 or larger recommended)

Verify that Docker can expose the GPU before running the workload:

```bash
docker run --rm --gpus all nvidia/cuda:12.9.1-base-ubuntu22.04 nvidia-smi
```

## Quick start

Create host directories for artifacts and the Hugging Face cache, ensure the
user running Docker can write to them, and use absolute mount paths:

```bash
mkdir -p /path/to/artifacts /path/to/hf-cache

docker run --rm --gpus all --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -v /path/to/input:/workspace/input:ro \
  -v /path/to/config:/workspace/config:ro \
  -v /path/to/artifacts:/workspace/artifacts \
  -v /path/to/hf-cache:/workspace/.hf_cache \
  --env HF_TOKEN \
  ghcr.io/nvidia-nemo/safe-synthesizer:latest-cu129 \
  run --config /workspace/config/config.yaml \
  --data-source /workspace/input/input.csv \
  --artifact-path /workspace/artifacts
```

Replace the paths and filenames with your own. Omit `--env HF_TOKEN` when the
selected models do not require it or when an approved token already exists in
the mounted Hugging Face cache. The inherited entrypoint passes everything
after the image reference to `safe-synthesizer` and warns about common mount,
cache, GPU, token, and shared-memory problems.

See [Running Safe Synthesizer](running.md) for other stages and CLI options and
[Configuration](configuration.md) for the YAML schema and override precedence.

## Runtime mounts and persistence

Docker bind mounts preserve host ownership. The image normally runs as
`appuser` with uid and gid 1000; the quick start uses `--user` to match the
host owner of writable mounts. In managed environments, you can instead
provision artifact and cache directories writable by uid/gid 1000.

| Content | Container path | Access | Lifecycle |
|---------|----------------|--------|-----------|
| Input data | `/workspace/input` | Read-only | Supplied by the user; never shipped in the image or repository |
| YAML configuration | `/workspace/config` | Read-only | Supplied by the user |
| Run artifacts | `/workspace/artifacts` | Read-write | Persist to retain adapters, generated data, reports, and logs |
| Hugging Face cache | `/workspace/.hf_cache` | Read-write | Persist to reuse downloaded models |

The image sets `HF_HOME=/workspace/.hf_cache`. It also starts in `/workspace`
and defaults artifacts to a relative `safe-synthesizer-artifacts` directory,
so pass the explicit `/workspace/artifacts` path whenever you mount a dedicated
artifact volume. See [Running -- Artifacts and Output](running.md#artifacts-and-output)
for the output tree and [Environment Variables](environment.md) for cache,
offline, logging, endpoint, and artifact settings.

Docker treats a relative source such as `-v data:/workspace/input` as a named
volume. Use an absolute host path or expand one with `$(pwd)`.

## Secrets

Inject credentials only at runtime. For example, export an approved value in
the calling shell and pass its name without putting the value in shell history:

```bash
export HF_TOKEN="<token>"
docker run --rm --gpus all --env HF_TOKEN ...
```

Other workflows can require `NSS_INFERENCE_KEY` or `WANDB_API_KEY`. Do not bake
credentials into an image. The complete variables and their purposes are in
[Environment Variables](environment.md).

## GPU Access

The image declares NVIDIA runtime visibility and compute capabilities, but
Docker still needs `--gpus all` (or an explicit device selection). Training
uses `/dev/shm` for worker communication; use `--shm-size=1g` as a starting
point and size it for your workload. The entrypoint warns below 256 MiB.

GPU, CPU, memory, and shared-memory requirements vary with the model, dataset,
and configuration. See [Program Runtime](troubleshooting.md) for GPU, OOM,
permissions, cache, and offline failures.

## Offline and Air-Gapped Environments

Populate a persistent model cache in an approved connected environment, move
or attach it according to organizational policy, and mount it at
`/workspace/.hf_cache`. Then add `--env HF_HUB_OFFLINE=1`. Required models must
already exist in the cache. See
[Environment -- Hugging Face cache and offline](environment.md#hugging-face-cache-and-offline)
for the complete offline contract.

## Building the project image from source

Consuming the public image above is the normal user path. Building the project
image is a separate developer workflow that requires a source checkout and
produces local tags rather than pulling the published runtime:

```bash
mise run container:build:gpu
mise run container:build:gpu-dev
```

See [Developer Guide -- Docker](../developer-guide/docker.md) for build stages,
arguments, and developer-image behavior. Those internals do not change the
public-image consumption contract on this page.

## Other deployment paths

- [Kubernetes Job](kubernetes.md) translates this workflow to a portable
  `batch/v1` Job.
- [Private Workload Images](private-workload-images.md) explains how to derive
  a governed image from an immutable public base while keeping sensitive data,
  artifacts, and caches external by default.

For the shared runtime contract, continue with
[Running Safe Synthesizer](running.md), [Configuration](configuration.md),
[Environment Variables](environment.md), or [Program Runtime](troubleshooting.md).
