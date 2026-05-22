# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

SHELL := /bin/bash
export PATH := $(HOME)/.local/share/mise/shims:$(HOME)/.local/bin:$(PATH)

MISE_GPG_KEY := 24853EC9F655CE80B48E6C3A8B81C9D17413A06D

.PHONY: help
help: ## Show mise tasks
	@mise tasks

.PHONY: install-mise
install-mise: ## Install mise (version from .mise.toml min_version; GPG-verified when gpg + gpg-agent + dirmngr are all available)
	@MISE_GPG_KEY=$(MISE_GPG_KEY) bash tools/install-mise.sh

.PHONY: setup
setup: install-mise ## Install dev tools and create the virtual environment via mise
	MISE_YES=1 mise trust
	MISE_YES=1 mise run setup

.PHONY: run
run: ## Run a mise task. Usage: make run TASK=format-check [MISE_ARGS="..."] [ARGS="..."]
	@if [ -z "$(TASK)" ]; then \
		echo "Error: missing TASK. Usage: make run TASK=format-check [MISE_ARGS=\"...\"] [ARGS=\"...\"]" >&2; \
		exit 1; \
	fi
	mise run $(MISE_ARGS) "$(TASK)" $(ARGS)

define deprecated_target
.PHONY: $(1)
$(1):
	@printf '%s\n' 'deprecated; forwarding to `mise run $(2)`' >&2
	@mise run $(2)
endef

BOOTSTRAP_EXTRAS := dev engine cpu cuda cu129
$(BOOTSTRAP_EXTRAS):
	@:

.PHONY: bootstrap-nss
bootstrap-nss:
	$(eval EXTRA := $(filter-out $@, $(MAKECMDGOALS)))
	@if [ -z "$(EXTRA)" ]; then \
		echo 'usage: mise run bootstrap-nss <extra>' >&2; \
		exit 1; \
	fi
	@printf '%s\n' 'deprecated; forwarding to `mise run bootstrap-nss $(EXTRA)`' >&2
	@mise run bootstrap-nss "$(EXTRA)"

$(eval $(call deprecated_target,verify-python-version,verify-python-version))
$(eval $(call deprecated_target,.venv,venv))

.PHONY: bootstrap-python
bootstrap-python:
	@extra="$${PYTORCH_DEPS:-cpu}"; \
		printf '%s\n' "deprecated; forwarding to \`mise run bootstrap-nss $$extra\`" >&2; \
		mise run bootstrap-nss "$$extra"

$(eval $(call deprecated_target,bootstrap-tools,bootstrap-tools))
$(eval $(call deprecated_target,bootstrap-tools-ci,bootstrap-tools-ci))
$(eval $(call deprecated_target,build-wheel,build-wheel))
$(eval $(call deprecated_target,check,check))
$(eval $(call deprecated_target,clean-cache,clean-cache))
$(eval $(call deprecated_target,clean-python,clean-python))
$(eval $(call deprecated_target,clean-uv,clean-uv))
$(eval $(call deprecated_target,container-build-gpu,container:build:gpu))
$(eval $(call deprecated_target,container-build-gpu-dev,container:build:gpu-dev))
$(eval $(call deprecated_target,container-build-gpu-multiarch,container:build:gpu-multiarch))
$(eval $(call deprecated_target,container-build-test,container:build:test))
$(eval $(call deprecated_target,container-build-test-setup,container:build:test-setup))
$(eval $(call deprecated_target,container-run-gpu,container:run:gpu))
$(eval $(call deprecated_target,container-run-gpu-dev,container:run:gpu-dev))
$(eval $(call deprecated_target,docs-build,docs:build))
$(eval $(call deprecated_target,docs-deploy,docs:deploy))
$(eval $(call deprecated_target,docs-serve,docs:serve))
$(eval $(call deprecated_target,format,format))
$(eval $(call deprecated_target,format-check,format-check))
$(eval $(call deprecated_target,lock-check,lock-check))
$(eval $(call deprecated_target,publish-internal,publish:internal))
$(eval $(call deprecated_target,publish-pypi,publish:pypi))
$(eval $(call deprecated_target,test,test))
$(eval $(call deprecated_target,test-ci,test:ci))
$(eval $(call deprecated_target,test-ci-container,test:ci-container))
$(eval $(call deprecated_target,test-ci-slow,test:ci-slow))
$(eval $(call deprecated_target,test-e2e,test:e2e))
$(eval $(call deprecated_target,test-e2e-collect,test:e2e:collect))
$(eval $(call deprecated_target,test-e2e-default,test:e2e:default))
$(eval $(call deprecated_target,test-e2e-dp,test:e2e:dp))
$(eval $(call deprecated_target,test-gpu-integration,test:gpu-integration))
$(eval $(call deprecated_target,test-smoke,test:smoke))
$(eval $(call deprecated_target,test-smoke-gpu,test:smoke:gpu))
$(eval $(call deprecated_target,test-smoke-gpu-generation,test:smoke:gpu:generation))
$(eval $(call deprecated_target,test-smoke-gpu-resume,test:smoke:gpu:resume))
$(eval $(call deprecated_target,test-smoke-gpu-smollm2,test:smoke:gpu:smollm2))
$(eval $(call deprecated_target,test-smoke-gpu-structured-generation,test:smoke:gpu:structured-generation))
$(eval $(call deprecated_target,test-smoke-gpu-timeseries,test:smoke:gpu:timeseries))
$(eval $(call deprecated_target,test-smoke-gpu-train-only,test:smoke:gpu:train-only))
$(eval $(call deprecated_target,test-tool-install,test:tool-install))
$(eval $(call deprecated_target,test-unit-slow,test:unit-slow))
$(eval $(call deprecated_target,typecheck,typecheck))

.PHONY: test-nss-%-ci
test-nss-%-ci:
	@printf '%s\n' 'deprecated; forwarding to `mise run $@`' >&2
	@mise run "$@"
