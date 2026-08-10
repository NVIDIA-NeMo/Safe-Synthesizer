<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Running as a Kubernetes Job

The published Safe Synthesizer runtime can run the same CLI workflow as a
Kubernetes `batch/v1` Job. The project does not ship input data, workload
configuration, Kubernetes resources, a Helm chart, or an operator. The
template below is a portable starting point to adapt to your platform.

Start with [Docker -- Select an image tag](docker.md#select-an-image-tag). Use
an approved versioned `cu129` tag or, preferably, an immutable digest for a
repeatable Job.

## Platform prerequisites

Your platform team owns:

- NVIDIA GPU Operator or device-plugin installation and compatible GPU drivers
- StorageClass and PVC provisioning, access modes, retention policy, and input
  data upload
- Node selection, taints/tolerations, admission policy, and any RuntimeClass
- Registry authentication and image-pull Secrets when required
- CPU, GPU, memory, ephemeral storage, and `/dev/shm` sizing
- Any Helm packaging or higher-level workload orchestration

Safe Synthesizer does not claim a supported Kubernetes chart or operator. GPU,
storage, security-context, and scheduling behavior varies by cluster; validate
the template with the platform owner.

Before creating the Job, provision these names in its namespace:

- `safe-synthesizer-input`: PVC containing the input file
- `safe-synthesizer-artifacts`: writable PVC for all run outputs
- `safe-synthesizer-hf-cache`: writable PVC for Hugging Face downloads
- `safe-synthesizer-config`: ConfigMap with a `config.yaml` key
- `safe-synthesizer-runtime`: Secret with the runtime credential keys your
  configuration needs

The ConfigMap is for non-secret YAML only. Do not put tokens in it. The
[Configuration](configuration.md) and [Environment Variables](environment.md)
pages define configuration and credential requirements without binding them to
a particular secret-management system.

## Portable Job template

Replace the image placeholder with an approved versioned or digest-pinned
`cu129` reference, and replace the input filename. This manifest deliberately
references resources that already exist; it does not provision storage or
include data/configuration content.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: safe-synthesizer
spec:
  backoffLimit: 1
  template:
    metadata:
      labels:
        app.kubernetes.io/name: safe-synthesizer
    spec:
      restartPolicy: Never
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
        - name: safe-synthesizer
          image: ghcr.io/nvidia-nemo/safe-synthesizer@sha256:<digest>
          imagePullPolicy: IfNotPresent
          args:
            - run
            - --config
            - /workspace/config/config.yaml
            - --data-source
            - /workspace/input/input.csv
            - --artifact-path
            - /workspace/artifacts
          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: safe-synthesizer-runtime
                  key: HF_TOKEN
                  optional: true
            - name: NSS_INFERENCE_KEY
              valueFrom:
                secretKeyRef:
                  name: safe-synthesizer-runtime
                  key: NSS_INFERENCE_KEY
                  optional: true
          resources:
            limits:
              nvidia.com/gpu: 1
          securityContext:
            allowPrivilegeEscalation: false
            capabilities:
              drop:
                - ALL
          volumeMounts:
            - name: input
              mountPath: /workspace/input
              readOnly: true
            - name: config
              mountPath: /workspace/config
              readOnly: true
            - name: artifacts
              mountPath: /workspace/artifacts
            - name: hf-cache
              mountPath: /workspace/.hf_cache
            - name: dshm
              mountPath: /dev/shm
      volumes:
        - name: input
          persistentVolumeClaim:
            claimName: safe-synthesizer-input
            readOnly: true
        - name: config
          configMap:
            name: safe-synthesizer-config
            defaultMode: 0444
        - name: artifacts
          persistentVolumeClaim:
            claimName: safe-synthesizer-artifacts
        - name: hf-cache
          persistentVolumeClaim:
            claimName: safe-synthesizer-hf-cache
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: 1Gi
```

The image already supplies its non-root CLI entrypoint, so the Job sets
`args`, not `command`. Pod- and container-level security contexts establish a
uid/gid/fsGroup 1000 baseline that matches the image. A storage driver or
cluster policy can interpret ownership and read-only requests differently;
confirm that the two writable PVCs permit writes by uid/gid 1000.

The `nvidia.com/gpu: 1` limit requests one GPU when the cluster advertises that
resource. It does not install or configure GPU support. Likewise, the
memory-backed `emptyDir` provides `/dev/shm`, but its capacity and accounting
are cluster-dependent. Add CPU and memory requests/limits based on measured
workload needs; do not treat this example's omission as sizing guidance.

## Submit and observe

Save your adapted manifest outside this repository, submit it with your normal
deployment process, and inspect status and logs:

```bash
kubectl apply -f safe-synthesizer-job.yaml
kubectl get job safe-synthesizer
kubectl get pods -l job-name=safe-synthesizer
kubectl logs --follow job/safe-synthesizer
kubectl describe job safe-synthesizer
```

For logs intended for aggregation, set `NSS_LOG_FORMAT=json` through your
platform's environment policy. See [Running -- Logging and Experiment Tracking](running.md#logging-and-experiment-tracking)
for log behavior and [Program Runtime](troubleshooting.md) for failures. Job and
Pod conditions reflect Kubernetes scheduling/process state; Safe Synthesizer
results remain in the artifact PVC.

## Persistence and follow-on generation

The artifact PVC retains the resolved configuration, training adapter,
generated data, evaluation report, metrics, and run logs described in
[Running -- Artifacts and Output](running.md#artifacts-and-output). The cache
PVC avoids downloading Hugging Face models for every Job. Whether PVC data
survives Job or namespace deletion depends on platform retention policy.

For a follow-on generation Job, create a new Job name, mount the same artifact
and cache PVCs, and replace `args` with an exact persisted run path:

```yaml
args:
  - run
  - generate
  - --run-path
  - /workspace/artifacts/<config>---<dataset>/<run-name>
```

An exact run path avoids ambiguity when a PVC holds multiple trained runs. See
[Running Safe Synthesizer](running.md) for generation options and
[Configuration](configuration.md) for workload tuning.

## Related deployment paths

- [Docker](docker.md) is the canonical public-image runtime contract.
- [Private Workload Images](private-workload-images.md) describes governed
  derived images that can replace the public image reference in this Job.
- [Environment Variables](environment.md) covers cache, offline, endpoint,
  logging, and artifact settings.
- [Program Runtime](troubleshooting.md) covers runtime and resource failures.
