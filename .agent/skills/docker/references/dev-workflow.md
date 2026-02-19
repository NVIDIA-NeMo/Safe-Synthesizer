<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Dev Workflow

Dockerfile.dev for fast iteration, test containers, Docker Compose for development, and bind mount patterns.

## Dockerfile.dev Pattern

Overlay code on a pre-built base image for fast iteration. Instead of rebuilding the full image on every code change, install only the changed package:

```dockerfile
ARG BASE_IMAGE=myregistry/myapp:latest
FROM ${BASE_IMAGE}

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /bin/uv

# Copy only what changed and reinstall
COPY pyproject.toml uv.lock src/ ./
RUN uv pip install --no-deps --reinstall --no-cache .
```

Build:

```bash
docker build -f Dockerfile.dev \
    --build-arg BASE_IMAGE=myapp:latest \
    -t myapp:dev .
```

## Test Containers

Run CI tests in a container to match the CI environment exactly (this repo's pattern):

```makefile
CONTAINER_CMD ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
CONTAINER_TEST_IMAGE ?= myapp-test:latest

container-build-test:
	$(CONTAINER_CMD) build --platform linux/amd64 \
	    --tag $(CONTAINER_TEST_IMAGE) \
	    --progress=plain \
	    -f containers/Dockerfile.test_ci .

test-ci-container: container-build-test
	$(CONTAINER_CMD) run --rm \
	    --platform linux/amd64 \
	    --mount type=bind,source=$(PWD),target=/workspace \
	    -e DEBIAN_FRONTEND=noninteractive \
	    $(CONTAINER_TEST_IMAGE) \
	    make test-ci
```

Key points:
- Bind-mount the repo so container sees current source
- Set `UV_PROJECT_ENVIRONMENT=/opt/venv` (outside `/workspace`) so the bind mount doesn't hide the venv
- Prefer podman, fall back to docker

## Docker Compose for Development

```yaml
services:
  app:
    build:
      context: .
      target: builder  # Use builder stage for dev (has tools)
    volumes:
      - .:/workspace
      - /workspace/.venv  # Anonymous volume to preserve venv
    ports:
      - "8080:8080"
    environment:
      - UV_PROJECT_ENVIRONMENT=/opt/venv
    command: uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_PASSWORD: devpassword
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  pgdata:
```

Note: omit the `version` field -- it's obsolete as of Compose v2.40+.

## Bind Mount Patterns

### Preserve venv across mounts

When bind-mounting the repo into a container, the mount hides the venv that was built inside the image. Fix by placing the venv outside the mount path:

```dockerfile
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
```

Or use an anonymous volume in Compose:

```yaml
volumes:
  - .:/workspace
  - /workspace/.venv  # Preserves image's .venv
```

### Hot reload

Mount source and use a reloading server:

```yaml
services:
  app:
    volumes:
      - ./src:/app/src:ro
    command: uvicorn app.main:app --reload --reload-dir /app/src
```

## GPU Development

```yaml
services:
  gpu-app:
    build:
      context: .
      dockerfile: Dockerfile.gpu
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
```

Or with `docker run`:

```bash
docker run --gpus all -it myapp:gpu
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/) on the host.
