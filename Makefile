### CONFIGURATION ###

SHELL := /bin/bash
UNAME_S := $(shell uname -s)
ARCH := $(shell uname -m)
PLATFORM := $(shell echo $(UNAME_S) | tr '[:upper:]' '[:lower:]')
NSS_ROOT_PATH := $(shell pwd)
PYTORCH_DEPS ?= cpu

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

# Pytest configuration
PYTEST_ADDOPTS := -n auto --dist loadscope --maxprocesses=8 -vv
PYTEST_CI_OPTS := --cov --cov-report json:coverage.json --cov-report xml:coverage.xml
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
BOOTSTRAP_EXTRAS := dev engine cpu cuda
$(BOOTSTRAP_EXTRAS):
	@:

.PHONY: bootstrap-nss
bootstrap-nss: .venv ## Bootstrap Python dependencies. Usage: make bootstrap-nss {dev,engine,cpu,cuda}
	$(eval EXTRA := $(filter-out $@, $(MAKECMDGOALS)))
	@echo "~~~~~~"
	@echo "attempting to install nss package with primary extra: $(EXTRA)"
	@if [ "$(EXTRA)" = "cuda" ]; then \
		uv sync --frozen --extra cu128 --extra engine --group dev; \
	elif [ "$(EXTRA)" = "cpu" ]; then \
		uv sync --frozen --extra cpu --extra engine --group dev; \
	elif [ "$(EXTRA)" = "engine" ]; then \
		uv sync --frozen --extra engine --group dev; \
	elif [ "$(EXTRA)" = "dev" ]; then \
		uv sync --frozen --group dev; \
	elif [ "$(EXTRA)" = "microservice" ]; then \
		uv sync --frozen --extra microservice --group dev; \
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
	uv run --group docs mkdocs build

.PHONY: docs-deploy
docs-deploy: ## Deploy the documentation site to GitHub Pages
	uv run --group docs mkdocs gh-deploy --force


### CODE QUALITY ###

.PHONY: format
format: ## Format the code
	bash tools/format/format.sh

.PHONY: lint
lint: ## Lint the code
	bash tools/lint/ruff-lint.sh
	bash tools/lint/run-ty-check.sh


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
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests -m "gpu_integration"

# Please modify these based on updating the e2e tests for NMP CI
.PHONY: test-e2e
test-e2e: ## Run all e2e tests (requires CUDA)
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests/e2e/ -m "e2e" -k default && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests/e2e/ -m "e2e" -k dp


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


### NMP SYNCHRONIZATION ###

# Guard: require NMP_REPO_PATH to be set before running sync targets
define check-nmp-repo-path
$(if $(NMP_REPO_PATH),,$(error NMP_REPO_PATH is not set. Set it to the root path of the NMP repo.))
endef

RSYNC_EXCLUDES := \
	--exclude='.git' \
	--exclude='.github' \
	--exclude='.vscode' \
	--exclude='.gitignore' \
	--exclude='.agent' \
	--exclude='__pycache__' \
	--exclude='*.pyc' \
	--exclude='.pytest_cache' \
	--exclude='.envrc' \
	--exclude='.venv' \
	--exclude='*.pycache.*' \
	--exclude='.cursor'

RSYNC_METAFILES_EXCLUDES := \
	--exclude='__init__.py' \
	--exclude='.pre-commit-config.yaml' \
	--exclude='.markdownlint.json' \
	--exclude='CODE_OF_CONDUCT.md' \
	--exclude='CONTRIBUTING.md' \
	--exclude='DCO' \
	--exclude='Makefile' \
	--exclude='LICENSE' \
	--exclude='README.md' \
	--exclude='SECURITY.md' \
	--exclude='THIRD_PARTY.md' \
	--exclude='ruff.toml' \
	--exclude='pytest.ini' \
	--exclude='pyproject.toml' \
	--exclude='tools' \
	--exclude='uv.lock'

RSYNC_CMD := rsync -av $(RSYNC_EXCLUDES)
RSYNC_METAFILES_CMD := rsync -av $(RSYNC_METAFILES_EXCLUDES)

.PHONY: synchronize-from-nmp-mr
synchronize-from-nmp-mr: ## Sync from NMP MR. Usage: make synchronize-from-nmp-mr MR=<number>
ifndef MR
	$(error MR is required. Usage: make synchronize-from-nmp-mr MR=5603)
endif
ifeq ($(shell git rev-parse --abbrev-ref HEAD), main)
	@echo "~~~~~~"
	@echo "you are on the main branch"
	@echo "creating a new branch from main"
	git checkout -b $$USER/sync-$(MR)-from-nmp
endif
	@echo "~~~~~~"
	@echo "synchronizing the changes from the NMP MR $(MR) to the nemo_safe_synthesizer package"
	bash tools/sync-from-mr.sh $(MR)

.PHONY: synchronize-py-files-from-nmp
synchronize-py-files-from-nmp: ## Synchronize python files from the NMP package
	$(call check-nmp-repo-path)
	@echo "~~~~~~"
	@echo "synchronizing python files from the NMP package"
	$(RSYNC_CMD) \
		$(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/src/ $(NSS_ROOT_PATH)/src/
	$(RSYNC_CMD) \
		$(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/tests/ $(NSS_ROOT_PATH)/tests/

.PHONY: synchronize-py-files-to-nmp
synchronize-py-files-to-nmp: ## Synchronize python files to the NMP package
	$(call check-nmp-repo-path)
	@echo "~~~~~~"
	@echo "synchronizing python files to the NMP package"
	$(RSYNC_CMD) \
		$(NSS_ROOT_PATH)/src/ $(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/src/
	$(RSYNC_CMD) \
		$(NSS_ROOT_PATH)/tests/ $(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/tests/

.PHONY: synchronize-metafiles-from-nmp
synchronize-metafiles-from-nmp: ## Synchronize metafiles from the NMP package
	$(call check-nmp-repo-path)
	@echo "~~~~~~"
	@echo "synchronizing metafiles from the NMP package"
	$(RSYNC_METAFILES_CMD) \
		$(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/ $(NSS_ROOT_PATH)/

.PHONY: synchronize-metafiles-to-nmp
synchronize-metafiles-to-nmp: ## Synchronize metafiles to the NMP package
	$(call check-nmp-repo-path)
	@echo "~~~~~~"
	@echo "synchronizing metafiles to the NMP package"
	$(RSYNC_METAFILES_CMD) \
		$(NSS_ROOT_PATH)/ $(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/

.PHONY: synchronize-to-nmp
synchronize-to-nmp: synchronize-py-files-to-nmp synchronize-metafiles-to-nmp ## Synchronize all files to the NMP package

.PHONY: synchronize-from-nmp
synchronize-from-nmp: synchronize-py-files-from-nmp synchronize-metafiles-from-nmp ## Synchronize the full NMP nemo_safe_synthesizer package locally
