# Platform and architecture detection
SHELL := /bin/bash
UNAME_S := $(shell uname -s)
ARCH := $(shell uname -m)
EXTRA ?= cpu
PLATFORM := $(shell echo $(UNAME_S) | tr '[:upper:]' '[:lower:]')
NSS_ROOT_PATH := $(shell pwd)
PYTEST_ADDOPTS := -n auto --dist loadscope --maxprocesses=8 -vv
PYTEST_CI_OPTS := --cov --cov-report json:coverage.json --cov-report xml:coverage.xml
PYTEST_CMD := uv run --frozen pytest $(PYTEST_ADDOPTS)
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
# CLI configurable environment variable for Python extras (defaults to cpu)

# Display platform info
$(info local system architecture: $(PLATFORM)/$(ARCH))


help:
    # taken from https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
	@echo "Makefile commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.PHONY: bootstrap-tools
bootstrap-tools: ## Bootstrap tools
	bash tools/binaries/bootstrap_tools.sh
	@echo "tools bootstrapped successfully"

.PHONY: install-uv
install-uv: ## Install uv tool
	bash tools/install_uv.sh
	@echo "uv tool installed successfully"

.PHONY: clean-python
clean-python: ## remove python virtual environment
	rm -rf .venv/

verify-python-version: ## Verify Python version and install if necessary
	@echo "~~~~~~"
	@echo "verifying python version"
	uv python find 3.11 || uv python install 3.11

.venv: verify-python-version ## Create a Python virtual environment
	@echo "~~~"
	@echo "setting up a venv wit uv"
	uv venv --seed --allow-existing

.PHONY: bootstrap-python
bootstrap-python: .venv ## Bootstrap Python dependencies with optional Pytorch cuda/cpu version. set PYTORCH_DEPS to 'cpu|cu128'.
	@echo "~~~~~~"
	@echo "installing python dependencies ${PYTORCH_DEPS} version of torch"
	@echo "cpu/cuda version is set with the env variable 'PYTORCH_DEPS=cpu|cu128'"
	@echo "PYTORCH_DEPS=cu128 make bootstrap-python"
	uv sync --frozen --extra ${PYTORCH_DEPS} --extra engine --group dev

.PHONY: format
format: ## Format the code
	uv run --frozen ruff format  && uv run --frozen ruff check --select I --fix

.PHONY: lint
lint: ## Lint the code
	uv run --frozen ruff check .
	bash tools/lint/run-ty-check.sh

.PHONY: install-safe-synthesizer
install-safe-synthesizer: ## Install the safe-synthesizer package into the sdk
	cd ${NSS_ROOT_PATH} && uv sync --frozen --extra ${PYTORCH_DEPS} --dev --extra engine


test-sdk-related: install-safe-synthesizer ## Run all pytest tests
		$(PYTEST_CMD) \
		$(NSS_ROOT_PATH)/tests/config  \
		$(NSS_ROOT_PATH)/tests/sdk  \
		$(NSS_ROOT_PATH)/tests/cli  \
		$(NSS_ROOT_PATH)/tests/api

test: bootstrap-python ## Run all pytest tests for the nemo_safe_synthesizer package
	$(PYTEST_CMD) -m "not e2e"

test-slow: install-safe-synthesizer ## Run all pytest tests for the nemo_safe_synthesizer package
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests -m "not e2e" --run-slow

test-ci: install-safe-synthesizer ## Run all pytest tests for the nemo_safe_synthesizer package in CI
	pushd $(NSS_ROOT_PATH) && \
	uv sync --extra cu128 --dev && \
	$(PYTEST_CMD) $(PYTEST_CI_OPTS) $(NSS_ROOT_PATH)/tests -m "not e2e"

# please modify these based on updating the e2e tests for nmp ci
test-e2e: install-safe-synthesizer ## Run all e2e tests for the nemo_safe_synthesizer package (requires cuda)
	pushd $(NSS_ROOT_PATH) && \
	uv sync --extra cu128 --dev  && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests/e2e/ -m "e2e" -k default && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests/e2e/ -m "e2e" -k dp





RSYNC_EXCLUDES := 
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
--exclude='.ruff_cache' \

RSYNC_METAFILES_EXCLUDES :=
--exclude='Makefile' \
--exclude='pytest.ini' \
--exclude='pyproject.toml' \
--exclude='README.md' \
--exclude='LICENSE' \
--exclude='THIRD_PARTY.md' \
--exclude='CODE_OF_CONDUCT.md' \
--exclude='CONTRIBUTING.md' \
--exclude='SECURITY.md' \
--exclude='uv.lock' \
--exclude='tools' \
--exclude='script' \
--exclude='__init__.py' \
--exclude='ruff.toml' \
--exclude='.pre-commit-config.yaml' \
--exclude='.markdownlint.json'

RSYNC_CMD := rsync -av $(RSYNC_EXCLUDES)
RSYNC_METAFILES_CMD := rsync -av $(RSYNC_METAFILES_EXCLUDES)


synchronize-from-nmp-mr: ## Sync from NMP MR. Usage: make synchronize-from-nmp-mr MR=<number>
ifndef MR
	$(error MR is required. Usage: make synchronize-from-nmp-mr MR=5603)
endif
	bash tools/sync-from-mr.sh $(MR)

synchronize-py-files-to-nmp: ## Synchronize the python files with the nmp package
	@echo "~~~~~~"
	@echo "synchronizing the python files with the nmp package"
ifeq ($(NMP_REPO_PATH),)
	@echo "~~~~~~"
	@echo "NMP_REPO_PATH is not set"
	@echo "please set the NMP_REPO_PATH environment variable"
	@echo "NMP_REPO_PATH is the root path of the nmp package"
	@exit 1
endif
	@echo "~~~~~~"
	$(RSYNC_CMD) \
		$(NSS_ROOT_PATH)/src/ $(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/src/
	$(RSYNC_CMD) \
		$(NSS_ROOT_PATH)/tests/ $(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/tests/

synchronize-metafiles-from-nmp: ## Synchronize the metafiles with the nmp package
ifeq ($(NMP_REPO_PATH),)
	@echo "~~~~~~"
	@echo "NMP_REPO_PATH is not set"
	@echo "please set the NMP_REPO_PATH environment variable"
	@echo "NMP_REPO_PATH is the root path of the nmp package"
	@exit 1
endif
	@echo "~~~~~~"
	@echo "synchronizing the metafiles with the nmp package"
	$(RSYNC_METAFILES_CMD) \
		$(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/ $(NSS_ROOT_PATH)/


synchronize-to-nmp: synchronize-metafiles-to-nmp


synchronize-from-nmp: ## Synchronize the nemo_safe_synthesizer package with the nmp package
	@echo "~~~~~~"
	@echo "synchronizing the nss package with the nmp package"

ifeq ($(NMP_REPO_PATH),)
	@echo "~~~~~~"
	@echo "NMP_REPO_PATH is not set"
	@echo "please set the NMP_REPO_PATH environment variable"
	@echo "NMP_REPO_PATH is the root path of the nmp package"
	@exit 1
endif
	# this is annoying but it will work for now. we can remove as soon as we fully migrate. 
	$(RSYNC_CMD) \
		$(NMP_REPO_PATH)/packages/nemo_safe_synthesizer/ $(NSS_ROOT_PATH)/