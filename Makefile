# Platform and architecture detection
SHELL := /bin/bash
UNAME_S := $(shell uname -s)
ARCH := $(shell uname -m)
EXTRA ?= cpu
PLATFORM := $(shell echo $(UNAME_S) | tr '[:upper:]' '[:lower:]')
NSS_ROOT_PATH := $(shell dirname $(shell dirname $(shell pwd)))
PYTEST_ADDOPTS := -c $(NSS_ROOT_PATH)/pytest.ini -n auto --dist loadscope --maxprocesses=8 -vv
PYTEST_CI_OPTS := --cov --cov-report json:coverage.json --cov-report xml:coverage.xml
PYTEST_CMD := uv run --frozen pytest $(PYTEST_ADDOPTS)

# Normalize architecture names
ifeq ($(ARCH),x86_64)
	ARCH := amd64
	EXTRA := cu128
	export BUILD_ARCH ?= linux/amd64
endif
ifeq ($(ARCH),aarch64)
	ARCH := arm64
endif
ifeq ($(ARCH),arm64)
	export BUILD_ARCH ?= linux/arm64
endif
# CLI configurable environment variable for Python extras (defaults to cpu)
PYTORCH_DEPS ?= cpu

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
	uv sync --frozen --all-packages --extra ${PYTORCH_DEPS}


install-safe-synthesizer: ## Install the safe-synthesizer package into the sdk
	cd ${NMP_ROOT_PATH} && uv sync --frozen --package nemo-safe-synthesizer --extra cu128 --dev --extra engine


test-sdk-related: install-safe-synthesizer ## Run all pytest tests
		$(PYTEST_CMD) \
		$(NSS_ROOT_PATH)/tests/config  \
		$(NSS_ROOT_PATH)/tests/sdk  \
		$(NSS_ROOT_PATH)/tests/cli  \
		$(NSS_ROOT_PATH)/tests/api

test: install-safe-synthesizer ## Run all pytest tests for the nemo_safe_synthesizer package
	pushd $(NSS_ROOT_PATH) && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests -m "not e2e"

test-slow: install-safe-synthesizer ## Run all pytest tests for the nemo_safe_synthesizer package
	pushd $(NMP_ROOT_PATH) && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests -m "not e2e" --run-slow

test-ci: install-safe-synthesizer ## Run all pytest tests for the nemo_safe_synthesizer package in CI
	pushd $(NMP_ROOT_PATH) && \
	uv sync --extra cu128 --dev && \
	$(PYTEST_CMD) $(PYTEST_CI_OPTS) $(NSS_ROOT_PATH)/tests -m "not e2e"

# please modify these based on updating the e2e tests for nmp ci
test-e2e: install-safe-synthesizer ## Run all e2e tests for the nemo_safe_synthesizer package (requires cuda)
	pushd $(NMP_ROOT_PATH) && \
	uv sync --extra cu128 --dev  && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests/e2e/ -m "e2e" -k default && \
	$(PYTEST_CMD) $(NSS_ROOT_PATH)/tests/e2e/ -m "e2e" -k dp