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

# Display platform info
$(info local system architecture: $(PLATFORM)/$(ARCH))


### HELP ###

# taken from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help
help:
	@echo "Makefile commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


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
	uv run --group docs mkdocs serve

.PHONY: docs-build
docs-build: ## Build the documentation site
	uv run --frozen --no-project --group docs mkdocs build

.PHONY: docs-deploy
docs-deploy: ## Deploy the documentation site to GitHub Pages
	uv run --frozen --no-project --group docs mkdocs gh-deploy --force


### CODE QUALITY ###

.PHONY: format
format: ## Format the code
	bash tools/format/format.sh

.PHONY: lint
lint: ## Lint the code
	bash tools/lint/ruff-lint.sh
	bash tools/lint/run-ty-check.sh
	python tools/lint/copyright_fixer.py --check .


### TESTING ###

.PHONY: test
test: ## Run unit tests excluding slow tests
	$(PYTEST_CMD) -m "unit and not slow"

.PHONY: test-slow
test-slow: ## Run all tests including slow tests (excludes e2e)
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests -m "not e2e" --run-slow

.PHONY: test-sdk-related
test-sdk-related: ## Run SDK-related tests (config, sdk, cli, api)
	$(PYTEST_CMD) \
		$(NSS_ROOT_PATH)/tests/config \
		$(NSS_ROOT_PATH)/tests/sdk \
		$(NSS_ROOT_PATH)/tests/cli \
		$(NSS_ROOT_PATH)/tests/api

.PHONY: test-ci
test-ci: ## Run CI unit tests excluding slow and GPU tests
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(PYTEST_CI_OPTS) $(NSS_ROOT_PATH)/tests -m "not e2e and not gpu_integration and not slow"

.PHONY: test-ci-slow
test-ci-slow: ## Run slow tests in CI with coverage
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(PYTEST_CI_OPTS) $(NSS_ROOT_PATH)/tests -m "slow"

.PHONY: test-gpu-integration
test-gpu-integration: ## Run GPU integration tests
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests/e2e/ -m "gpu_integration and not e2e" -k default && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests/e2e/ -m "gpu_integration and not e2e" -k dp

# Please modify these based on updating the e2e tests for NMP CI
.PHONY: test-e2e
test-e2e: test-e2e-default test-e2e-dp ## Run all e2e tests (requires CUDA)

.PHONY: test-e2e-default
test-e2e-default: ## Run default e2e tests (requires CUDA)
# -n 0 is a workaround to run the tests in a single process.
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) -n 0 $(NSS_ROOT_PATH)/tests/e2e/ -m "e2e" -k default

.PHONY: test-e2e-dp
test-e2e-dp: ## Run dp e2e tests (requires CUDA)
# -n 0 is a workaround to run the tests in a single process.
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) -n 0 $(NSS_ROOT_PATH)/tests/e2e/ -m "e2e" -k dp

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
