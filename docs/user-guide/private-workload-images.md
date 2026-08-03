<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Private Workload Images

Runtime mounts are the default and recommended way to supply sensitive input
data and configuration. In a governed environment, an organization can instead
derive a private workload image containing an intentionally approved,
non-secret snapshot. The derived image uses the same Docker and Kubernetes
runtime contract as the published image.

## Pin the published base

Start `FROM` the published runtime at an approved immutable digest. Do not use
`latest-cu129` as the base of a reproducible derived image. Resolve and review
the digest for the selected release through your registry tooling, then record
it directly in the Dockerfile:

```dockerfile
# syntax=docker/dockerfile:1

FROM ghcr.io/nvidia-nemo/safe-synthesizer@sha256:<approved-runtime-digest>

COPY --chown=1000:1000 --chmod=0440 governed/config.yaml /workspace/workload/config.yaml
COPY --chown=1000:1000 --chmod=0440 governed/input.csv /workspace/workload/input.csv
```

This Dockerfile intentionally has no `USER`, `ENTRYPOINT`, or `CMD`
instruction. It therefore preserves the published runtime's non-root
`appuser` (uid/gid 1000), `tini`-wrapped CLI entrypoint, and default command.
`--chown=1000:1000` makes the governed files readable by that inherited user.
The filenames are schematic: the repository does not provide a dataset or
workload configuration.

Build and push the derived image only to an approved private registry, then
capture its digest for deployment:

```bash
docker build -t registry.example.com/team/safe-synthesizer-workload:<version> .
docker push registry.example.com/team/safe-synthesizer-workload:<version>
```

Registry login, access controls, scanning, signing/attestation, retention, and
promotion are organization-owned controls.

## Understand layer exposure

`COPY` makes the snapshot part of the image content. Anyone who can pull the
image may be able to extract it. The content can also persist in local build
caches, registry layers and replicas, vulnerability-scanning systems, backups,
and downstream images for as long as their separate retention policies allow.
Deleting the file in a later layer does not remove it from earlier layers.

Use this pattern only when governance explicitly approves that distribution
and retention boundary. For sensitive or frequently changing data and
configuration, keep using read-only runtime mounts as shown in
[Docker](docker.md) and [Kubernetes Job](kubernetes.md).

Never embed:

- credentials, private keys, Hugging Face tokens, inference API keys, or other
  secrets
- Hugging Face, vLLM, Triton, or other model/runtime caches
- artifacts, generated data, evaluation reports, logs, or trained adapters

Inject credentials through runtime secret mechanisms. Keep artifact and model
caches on external writable volumes with an organization-defined lifecycle.
See [Environment Variables](environment.md) for their paths and variables.

## Run the derived image

For Docker, replace the public image reference in the canonical command and
keep writable artifact and cache mounts external:

```bash
docker run --rm --gpus all --shm-size=1g \
  --user "$(id -u):$(id -g)" \
  -v /path/to/artifacts:/workspace/artifacts \
  -v /path/to/hf-cache:/workspace/.hf_cache \
  registry.example.com/team/safe-synthesizer-workload@sha256:<derived-digest> \
  run --config /workspace/workload/config.yaml \
  --data-source /workspace/workload/input.csv \
  --artifact-path /workspace/artifacts
```

For Kubernetes, replace `image` in the [Job template](kubernetes.md#portable-job-template)
with the derived digest. If the approved snapshot supplies both files, remove
only the input/config volume mounts and volumes; retain writable artifact and
HF-cache PVCs, runtime Secrets, GPU limit, `/dev/shm`, and security contexts.

The CLI workflow and outputs do not change. Use
[Running Safe Synthesizer](running.md) for execution and artifacts,
[Configuration](configuration.md) for parameters,
[Environment Variables](environment.md) for infrastructure settings, and
[Program Runtime](troubleshooting.md) for failure handling.

For the Safe Synthesizer project's own multi-stage Dockerfile and publication
mechanics—not for workload-image policy—see
[Developer Guide -- Docker](../developer-guide/docker.md).
