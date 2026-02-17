<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# BuildKit Bake Orchestration

Multi-image build orchestration with `docker-bake.hcl`, registry cache, and CI integration.

## docker-bake.hcl Basics

Bake lets you define multiple build targets in a single file, with shared variables, dependency chains, and caching.

```hcl
# Variables
variable "REGISTRY" {
    default = "ghcr.io/myorg/myproject"
}
variable "TAG" {
    default = "latest"
}
variable "CACHE_REGISTRY" {
    default = "ghcr.io/myorg/myproject/cache"
}

# Base image target (used as context by others)
target "python-base" {
    dockerfile = "docker/base/Dockerfile.python-base"
    tags       = ["${REGISTRY}/python-base:${TAG}"]
}

# Service target that depends on base
target "api" {
    dockerfile = "docker/Dockerfile.api"
    tags       = ["${REGISTRY}/api:${TAG}"]
    contexts   = {
        python-base = "target:python-base"
    }
    cache-from = ["type=registry,ref=${CACHE_REGISTRY}/api:main"]
    cache-to   = ["type=registry,ref=${CACHE_REGISTRY}/api:${TAG},mode=max"]
}

# GPU tasks target
target "gpu-tasks" {
    dockerfile = "docker/Dockerfile.gpu-tasks"
    tags       = ["${REGISTRY}/gpu-tasks:${TAG}"]
    platforms  = ["linux/amd64"]  # GPU: amd64 only
    contexts   = {
        gpu-base = "target:gpu-base"
    }
}

# Groups for building multiple targets at once
group "docker-cpu" {
    targets = ["python-base", "api", "cpu-tasks"]
}

group "docker-gpu" {
    targets = ["gpu-base", "gpu-tasks"]
}
```

## Context Injection

Pass one target's output as a build context to another:

```hcl
target "workspace" {
    dockerfile = "docker/base/Dockerfile.workspace"
    contexts   = {
        python-base = "target:python-base"
    }
}

target "api" {
    contexts = {
        workspace    = "target:workspace"
        runtime-base = "target:python-base"
    }
}
```

In the Dockerfile, reference injected contexts:

```dockerfile
FROM workspace AS builder
# workspace context is available as the build stage

COPY --from=runtime-base /usr/local/bin/python /usr/local/bin/python
```

## Registry Cache

Branch-specific caching for CI:

```hcl
variable "BRANCH" {
    default = "main"
}

function "cache_from" {
    params = [target]
    result = [
        "type=registry,ref=${CACHE_REGISTRY}/${target}:${BRANCH}",
        "type=registry,ref=${CACHE_REGISTRY}/${target}:main",
    ]
}

function "cache_to" {
    params = [target]
    result = ["type=registry,ref=${CACHE_REGISTRY}/${target}:${BRANCH},mode=max"]
}

target "api" {
    cache-from = cache_from("api")
    cache-to   = cache_to("api")
}
```

## Secrets

Pass build-time secrets without baking them into layers:

```hcl
target "gpu-tasks" {
    secret = [
        "type=env,id=HF_TOKEN",
        "type=env,id=NGC_CLI_API_KEY",
    ]
}
```

In the Dockerfile:

```dockerfile
RUN --mount=type=secret,id=HF_TOKEN \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) && \
    huggingface-cli download mymodel --token $HF_TOKEN
```

## Multi-Platform Builds

```hcl
# CPU targets: both architectures
target "api" {
    platforms = ["linux/amd64", "linux/arm64"]
}

# GPU targets: amd64 only (CUDA doesn't support arm64)
target "gpu-tasks" {
    platforms = ["linux/amd64"]
}
```

## Make Integration

```makefile
# Build a specific target
docker/%:
	docker buildx bake $*-docker --load

# Build all CPU images
docker/all-cpu:
	docker buildx bake docker-cpu --load

# Build all GPU images
docker/all-gpu:
	docker buildx bake docker-gpu --load
```

## CI Patterns

### Docker-in-Docker (DinD)

```yaml
# GitLab CI example
build:
  image: docker:28-dind
  services:
    - docker:28-dind
  variables:
    DOCKER_BUILDKIT: "1"
  script:
    - docker buildx bake docker-cpu --push
```

### Remote BuildKit

For faster builds, use a remote BuildKit instance:

```bash
docker buildx create --name remote-builder \
    --driver remote \
    --platform linux/amd64 \
    tcp://buildkit-amd64:1234

docker buildx create --name remote-builder \
    --append \
    --driver remote \
    --platform linux/arm64 \
    tcp://buildkit-arm64:1234
```

### Debug bake config

```bash
# Print resolved bake config without building
docker buildx bake --print docker-cpu
```
