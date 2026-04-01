# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

### CONFIGURATION ###

SHELL := /bin/bash
UNAME_S := $(shell uname -s)
ARCH := $(shell uname -m)
PLATFORM := $(shell echo $(UNAME_S) | tr '[:upper:]' '[:lower:]')
NSS_ROOT_PATH := $(shell pwd)

# Normalize architecture names
ifeq ($(ARCH),x86_64)
	ARCH := amd64
	PYTORCH_DEPS := cu128
	export BUILD_ARCH ?= linux/amd64
endif
ifeq ($(ARCH),aarch64)
	ARCH := arm64
endif
ifeq ($(ARCH),arm64)
	export BUILD_ARCH ?= linux/arm64
endif

PYTORCH_DEPS ?= cpu

# Pytest configuration
PYTEST_ADDOPTS := -n auto --dist loadscope -vv
PYTEST_CI_OPTS := --cov --cov-report json:coverage.json
PYTEST_CMD := uv run --frozen pytest $(PYTEST_ADDOPTS)
PYTEST_NO_XDIST_CMD := $(PYTEST_CMD) -n 0

# Display platform info
$(info local system architecture: $(PLATFORM)/$(ARCH))


### HELP ###

# taken from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help:
	@echo "Makefile commands:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


### BOOTSTRAP AND SETUP ###

.PHONY: bootstrap-tools
bootstrap-tools: ## Bootstrap tools
	bash tools/binaries/bootstrap_tools.sh
	@echo "tools bootstrapped successfully"

.PHONY: bootstrap-tools-ci
bootstrap-tools-ci: ## Bootstrap tools for CI
	bash tools/binaries/bootstrap_tools.sh --bootstrap-only

.PHONY: install-uv
install-uv: ## Install uv tool
	bash tools/binaries/install_uv.sh
	@echo "uv tool installed successfully"

.PHONY: clean-python
clean-python: ## Remove python virtual environment
	rm -rf .venv/

.PHONY: clean-uv
clean-uv: ## Remove uv cache files
	uv cache clear

.PHONY: clean-unsloth
clean-unsloth: ## Remove unsloth cache files
	rm -rf unsloth_compiled_cache/

.PHONY: clean-cache
clean-cache: clean-unsloth clean-uv clean-python ## Remove cache files from unsloth, uv, and other tools

.PHONY: verify-python-version
verify-python-version: ## Verify Python version and install if necessary
	@uv python find 3.11 || uv python install 3.11

.venv: verify-python-version ## Create a Python virtual environment
	uv venv --seed --allow-existing

.PHONY: bootstrap-python
bootstrap-python: .venv ## Bootstrap Python dependencies. Set PYTORCH_DEPS to 'cpu' or 'cu128'. Here mostly for legacy usage.
	uv sync --frozen --extra ${PYTORCH_DEPS} --extra engine --group dev

# Dynamic targets for bootstrap-nss
# Usage: make bootstrap-nss {dev,engine,cpu,cuda}
BOOTSTRAP_EXTRAS := dev engine cpu cuda cu128
$(BOOTSTRAP_EXTRAS):
	@:

.PHONY: bootstrap-nss
bootstrap-nss: .venv ## Bootstrap Python dependencies. Usage: make bootstrap-nss {dev,engine,cpu,cuda}
	$(eval EXTRA := $(filter-out $@, $(MAKECMDGOALS)))
	@echo "~~~~~~"
	@echo "attempting to install nss package with primary extra: $(EXTRA)"
	@if [ "$(EXTRA)" = "cuda" ]; then \
		uv sync --frozen --extra cu128 --extra engine --group dev; \
	elif [ "$(EXTRA)" = "cu128" ]; then \
		uv sync --frozen --extra cu128 --extra engine --group dev; \
	elif [ "$(EXTRA)" = "cpu" ]; then \
		uv sync --frozen --extra cpu --extra engine --group dev; \
	elif [ "$(EXTRA)" = "engine" ]; then \
		uv sync --frozen --extra engine --group dev; \
	elif [ "$(EXTRA)" = "dev" ]; then \
		uv sync --frozen --group dev; \
	else \
		echo "Error: Invalid extra '$(EXTRA)'. Use one of: $(BOOTSTRAP_EXTRAS)"; \
		exit 1; \
	fi


### DOCUMENTATION ###

.PHONY: docs-serve
docs-serve: ## Serve the documentation site locally with live reload
	uv run --frozen --group docs mkdocs serve --livereload

.PHONY: docs-build
docs-build: ## Build the documentation site
	uv run --frozen --no-project --group docs mkdocs build

.PHONY: docs-deploy
docs-deploy: ## Deploy the documentation site to GitHub Pages
	uv run --frozen --no-project --group docs mkdocs gh-deploy --force


### CODE QUALITY ###
# `make format` mutates files (ruff format + ruff check --fix + copyright).
# `make format-check`, `typecheck`, `lock-check` are atomic read-only targets (used by CI).
# `make check` runs all read-only checks (format-check + typecheck).

.PHONY: format
format: ## Format the code (ruff format + lint fix + copyright headers)
	bash tools/codestyle/format.sh
	uv run --script tools/codestyle/copyright_fixer.py .

.PHONY: format-check
format-check: ## Check formatting, lint rules, and copyright headers (read-only)
	bash tools/codestyle/format.sh --check
	bash tools/codestyle/ruff_check.sh
	uv run --script tools/codestyle/copyright_fixer.py --check .

.PHONY: typecheck
typecheck: ## Run ty type checks
	bash tools/codestyle/typecheck.sh

.PHONY: lock-check
lock-check: ## Check that uv.lock is up to date
	uv lock
	git diff --exit-code uv.lock

.PHONY: check
check: format-check typecheck ## Run all read-only CI checks locally


### TESTING ###

.PHONY: test
test: ## Run unit tests excluding slow tests
	$(PYTEST_CMD) -m "unit and not slow"

.PHONY: test-slow
test-unit-slow: ## Run unit tests including slow tests (excludes e2e and smoke)
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests -m "unit"

.PHONY: test-ci
test-ci: ## Run CI unit tests excluding slow, GPU, and smoke tests
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(PYTEST_CI_OPTS) $(NSS_ROOT_PATH)/tests -m "not e2e and not requires_gpu and not slow and not smoke"

.PHONY: test-ci-slow
test-ci-slow: ## Run slow tests in CI with coverage
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(PYTEST_CI_OPTS) $(NSS_ROOT_PATH)/tests -m "slow"


.PHONY: test-smoke
test-smoke: ## Run CPU smoke tests (~few min, no GPU required)
	$(PYTEST_CMD) -m "smoke and not requires_gpu"

SMOKE_DIR := tests/smoke
.PHONY: test-smoke-gpu
test-smoke-gpu: ## Run GPU smoke tests (requires CUDA)
# Uses PYTEST_NO_XDIST_CMD (-n 0) because CUDA device-side asserts poison
# xdist workers. Groups are split for GPU memory isolation.
#
# When adding a new GPU smoke test file:
#   - Train-only (no vLLM): add pytest.mark.requires_gpu -> auto-discovered below
#   - Uses vLLM: also add pytest.mark.vllm -> add the file to the vLLM list below
#   - Uses Unsloth: also add pytest.mark.unsloth -> auto-discovered below
#   - Downloads from Hub: also add pytest.mark.smollm2 (or similar) -> auto-discovered below
#
# 1) Train-only tests share a process (no vLLM, safe to batch).
	$(PYTEST_NO_XDIST_CMD) $(SMOKE_DIR)/ -m "requires_gpu and not vllm and not smollm2 and not unsloth"
# 2) Each vLLM test file gets its own process -- vLLM pre-allocates all GPU
#    memory and never releases it within a process.
	$(PYTEST_NO_XDIST_CMD) $(SMOKE_DIR)/test_nss_generation_gpu.py
	$(PYTEST_NO_XDIST_CMD) $(SMOKE_DIR)/test_nss_resume_gpu.py
	$(PYTEST_NO_XDIST_CMD) $(SMOKE_DIR)/test_nss_structured_gen_gpu.py
	$(PYTEST_NO_XDIST_CMD) $(SMOKE_DIR)/test_nss_timeseries_gpu.py
# 3) SmolLM2 (Hub download + vLLM) and Unsloth (patches transformers) are marker-isolated.
	$(PYTEST_NO_XDIST_CMD) $(SMOKE_DIR)/ -m "requires_gpu and smollm2"
	$(PYTEST_NO_XDIST_CMD) $(SMOKE_DIR)/ -m "requires_gpu and unsloth"


E2E_TEST_FILE := $(NSS_ROOT_PATH)/tests/e2e/test_safe_synthesizer.py
.PHONY: test-gpu-integration
test-gpu-integration: ## Run GPU e2e tests (default + DP configs)
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(E2E_TEST_FILE) -k default && \
	$(PYTEST_CMD) $(E2E_TEST_FILE) -k dp


# Please modify these based on updating the e2e tests for NMP CI
.PHONY: test-e2e
test-e2e: test-e2e-default test-e2e-dp ## Run all e2e tests (requires CUDA)

.PHONY: test-e2e-default
test-e2e-default: ## Run default e2e tests (requires CUDA)
# -n 0 is a workaround to run the tests in a single process.
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) -n 0 $(E2E_TEST_FILE) -k default

.PHONY: test-e2e-dp
test-e2e-dp: ## Run dp e2e tests (requires CUDA)
# -n 0 is a workaround to run the tests in a single process.
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) -n 0 $(E2E_TEST_FILE) -k dp

.PHONY: test-e2e-collect
test-e2e-collect: ## Dry-run: show which tests e2e/gpu targets select (requires CUDA deps)
	@echo "--- test-e2e-default ---"
	-@cd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) -n 0 $(E2E_TEST_FILE) -k default -o "addopts=" --collect-only -qq -p no:warnings 2>/dev/null
	@echo "--- test-e2e-dp ---"
	-@cd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) -n 0 $(E2E_TEST_FILE) -k dp -o "addopts=" --collect-only -qq -p no:warnings 2>/dev/null
	@echo "--- test-gpu-integration (default) ---"
	-@cd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(E2E_TEST_FILE) -k default -o "addopts=" --collect-only -qq -p no:warnings 2>/dev/null
	@echo "--- test-gpu-integration (dp) ---"
	-@cd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(E2E_TEST_FILE) -k dp -o "addopts=" --collect-only -qq -p no:warnings 2>/dev/null

### CONTAINER-BASED TESTING ###

# Auto-detect container runtime: prefer podman, fall back to docker
CONTAINER_CMD ?= $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)
CONTAINER_TEST_IMAGE ?= nss-test:latest
CONTAINER_TEST_FILE := containers/Dockerfile.test_ci
CONTAINER_TEST_PLATFORM := linux/amd64

CONTAINER_BUILD_ARGS ?= --platform $(CONTAINER_TEST_PLATFORM) \
	--tag $(CONTAINER_TEST_IMAGE) \
	--progress=plain \
	-f $(CONTAINER_TEST_FILE)

.PHONY: container-build-test
container-build-test: ## Build the container image for running CI tests locally
	$(CONTAINER_CMD) build $(CONTAINER_BUILD_ARGS) .

.PHONY: test-ci-container
test-ci-container: container-build-test ## Run CI unit tests in a Linux container
	$(CONTAINER_CMD) run \
		--rm \
		--platform $(CONTAINER_TEST_PLATFORM) \
		--mount type=bind,source=$(NSS_ROOT_PATH),target=/workspace \
		-e DEBIAN_FRONTEND=noninteractive \
		$(CONTAINER_TEST_IMAGE) \
		make test-ci


### CONTAINER GPU (CUDA) ###

CONTAINER_GPU_FILE := containers/Dockerfile.cuda
CONTAINER_GPU_IMAGE ?= nss-gpu:latest
CONTAINER_GPU_IMAGE_DEV ?= nss-gpu-dev:latest
# Multi-arch: override for arm64 builds (e.g., Blackwell).
#   make container-build-gpu CONTAINER_GPU_PLATFORM=linux/arm64
CONTAINER_GPU_PLATFORM ?= linux/amd64
# Registry for multi-arch manifest pushes (required by container-build-gpu-multiarch).
# The tag is built as CONTAINER_GPU_REGISTRY/CONTAINER_GPU_IMAGE, so
# CONTAINER_GPU_IMAGE should be a bare name:tag (e.g., nss-gpu:latest),
# not a fully-qualified registry path.
CONTAINER_GPU_REGISTRY ?=

# GPU access flag. Docker and Podman 4.x+ support --gpus all.
# Override for older Podman: CONTAINER_GPU_FLAG="--device nvidia.com/gpu=all"
CONTAINER_GPU_FLAG ?= --gpus all

# Bind-mount the host HF cache so model downloads persist across container runs.
CONTAINER_HF_CACHE ?= $(HOME)/.cache/huggingface

# Extra mounts for data outside the repo tree.
#   make container-run-gpu CONTAINER_EXTRA_MOUNTS="-v /data/input:/workspace/data" CMD="run --data-source /workspace/data/input.csv"
CONTAINER_EXTRA_MOUNTS ?=

CONTAINER_GPU_RUN_OPTS := \
	--rm \
	--platform $(CONTAINER_GPU_PLATFORM) \
	--shm-size=1g \
	--mount type=bind,source=$(NSS_ROOT_PATH),target=/workspace \
	--mount type=bind,source=$(CONTAINER_HF_CACHE),target=/workspace/.hf_cache \
	-e HF_HOME=/workspace/.hf_cache \
	-e DEBIAN_FRONTEND=noninteractive \
	$(CONTAINER_GPU_FLAG) \
	$(CONTAINER_EXTRA_MOUNTS)

CONTAINER_GPU_BUILD_RUNTIME := --platform $(CONTAINER_GPU_PLATFORM) \
	--tag $(CONTAINER_GPU_IMAGE) \
	--target runtime \
	--progress=plain \
	-f $(CONTAINER_GPU_FILE)

CONTAINER_GPU_BUILD_DEV := --platform $(CONTAINER_GPU_PLATFORM) \
	--tag $(CONTAINER_GPU_IMAGE_DEV) \
	--target dev \
	--progress=plain \
	-f $(CONTAINER_GPU_FILE)

.PHONY: container-build-gpu
container-build-gpu: ## Build CUDA runtime container (CLI wrapper)
	$(CONTAINER_CMD) build $(CONTAINER_GPU_BUILD_RUNTIME) .

.PHONY: container-build-gpu-dev
container-build-gpu-dev: ## Build CUDA dev container (tools + test deps)
	$(CONTAINER_CMD) build $(CONTAINER_GPU_BUILD_DEV) .

.PHONY: container-run-gpu
container-run-gpu: container-build-gpu ## Run command in GPU container. Usage: make container-run-gpu CMD="run --config ..."
	@mkdir -p $(CONTAINER_HF_CACHE)
	$(CONTAINER_CMD) run $(CONTAINER_GPU_RUN_OPTS) \
		-w /workspace \
		$(CONTAINER_GPU_IMAGE) \
		$(or $(CMD),--help)

.PHONY: container-run-gpu-dev
container-run-gpu-dev: container-build-gpu-dev ## Run command in GPU dev container. Usage: make container-run-gpu-dev CMD="make test"
	@mkdir -p $(CONTAINER_HF_CACHE)
	$(CONTAINER_CMD) run $(CONTAINER_GPU_RUN_OPTS) \
		-w /workspace \
		$(CONTAINER_GPU_IMAGE_DEV) \
		$(or $(CMD),make test)

.PHONY: container-build-gpu-multiarch
container-build-gpu-multiarch: ## Build multi-arch GPU image (requires docker + registry). Usage: make container-build-gpu-multiarch CONTAINER_GPU_REGISTRY=ghcr.io/nvidia-nemo
ifeq ($(strip $(CONTAINER_GPU_REGISTRY)),)
	$(error CONTAINER_GPU_REGISTRY is required for multi-arch builds. Multi-platform manifests must be pushed to a registry.)
endif
ifeq (,$(shell command -v docker 2>/dev/null))
	$(error Multi-arch builds require Docker with buildx. Podman does not support buildx.)
endif
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		--tag $(CONTAINER_GPU_REGISTRY)/$(CONTAINER_GPU_IMAGE) \
		--target runtime \
		--progress=plain \
		--push \
		-f $(CONTAINER_GPU_FILE) .


### BUILD AND PUBLISH ###

.PHONY: build-wheel
build-wheel: ## Build wheel (version from git tag via uv-dynamic-versioning)
	@echo "~~~~~~"
	rm -rf dist/
	uv build --wheel
	@echo "wheel built: $$(ls dist/*.whl)"

.PHONY: publish-internal
publish-internal: build-wheel ## Build and publish wheel to NVIDIA Artifactory. Uses TWINE_REPOSITORY_URL, TWINE_USERNAME, and TWINE_PASSWORD env vars.
ifndef TWINE_REPOSITORY_URL
	$(error TWINE_REPOSITORY_URL is not set. Set it to the URL of the Artifactory repository.)
endif
ifndef TWINE_USERNAME
	$(error TWINE_USERNAME is not set. Set it to the username for the Artifactory repository.)
endif
ifndef TWINE_PASSWORD
	$(error TWINE_PASSWORD is not set. Set it to the password for the Artifactory repository.)
endif
	@echo "~~~~~~"
	@echo "uploading to Artifactory: $(TWINE_REPOSITORY_URL)"
	uvx twine upload \
		--repository-url $(TWINE_REPOSITORY_URL) \
		--non-interactive \
		--verbose \
		dist/*.whl
	@echo "published: $$(ls dist/*.whl)"

.PHONY: publish-pypi
publish-pypi: build-wheel ## Build and publish wheel to PyPI. Uses TWINE_USERNAME and TWINE_PASSWORD env vars.
ifndef TWINE_USERNAME
	$(error TWINE_USERNAME is not set. For PyPI token auth, set TWINE_USERNAME=__token__.)
endif
ifndef TWINE_PASSWORD
	$(error TWINE_PASSWORD is not set. For PyPI token auth, set TWINE_PASSWORD=<your-pypi-token>.)
endif
	@echo "~~~~~~"
	@echo "uploading to PyPI"
	uvx twine upload \
		--non-interactive \
		--verbose \
		dist/*.whl
	@echo "published: $$(ls dist/*.whl)"



# ============================================================
# Config-Dataset Combination Tests (12 total)
# ============================================================
# Generated targets: test-nss-{CONFIG}-{DATASET}-ci
#   CONFIGS : tinyllama_unsloth tinyllama_dp smollm3_unsloth smollm3_dp mistral_nodp mistral_dp
#   DATASETS: clinc_oos dow_jones_index
# Example usage:
#   make test-nss-tinyllama_unsloth-clinc_oos-ci
#   make test-nss-tinyllama_dp-dow_jones_index-ci

NSS_CONFIGS  := tinyllama_unsloth tinyllama_dp smollm3_unsloth smollm3_dp mistral_nodp mistral_dp
NSS_DATASETS := clinc_oos dow_jones_index

define nss_combo_test
test-nss-$(1)-$(2)-ci: ## Run pytest test for $(shell echo $(1) | tr '_' '-') config with $(shell echo $(2) | tr '_' '-') dataset
	$(MAKE) bootstrap-nss cu128
	$(PYTEST_NO_XDIST_CMD) -vv $(PYTEST_CI_OPTS) $(NSS_ROOT_PATH)/tests/e2e/test_dataset_config.py -k "test_$(2)_dataset[$(subst _,-,$(1))]"
endef

$(foreach config,$(NSS_CONFIGS),\
  $(foreach dataset,$(NSS_DATASETS),\
    $(eval $(call nss_combo_test,$(config),$(dataset)))))

