#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT
readonly INSTALLER="${REPO_ROOT}/install_nss.sh"

assert_contains() {
    local actual="$1"
    local expected="$2"
    [[ "$actual" == *"$expected"* ]] || {
        echo "Expected output to contain: $expected" >&2
        echo "Actual output: $actual" >&2
        exit 1
    }
}

assert_not_contains() {
    local actual="$1"
    local unexpected="$2"
    [[ "$actual" != *"$unexpected"* ]] || {
        echo "Expected output not to contain: $unexpected" >&2
        echo "Actual output: $actual" >&2
        exit 1
    }
}

assert_dry_run() {
    local cuda="$1"
    local expected_extra="$2"
    local expected_index="${3:-}"
    local output
    output="$(DRY_RUN=1 CUDA="$cuda" "$INSTALLER")"

    assert_contains "$output" "nemo-safe-synthesizer\[engine\,${expected_extra}\]"
    if [[ -n "$expected_index" ]]; then
        assert_contains "$output" "--index"
        assert_contains "$output" "$expected_index"
    else
        assert_not_contains "$output" "--index"
    fi
}

assert_dry_run 129 cu129 "https://wheels.vllm.ai/"
assert_dry_run 130 cu130 "https://pypi.nvidia.com"
if [[ "$(uname -s)" == "Linux" ]]; then
    assert_dry_run cpu cpu "https://download.pytorch.org/whl/cpu"
else
    assert_dry_run cpu cpu
fi

help_output="$(DRY_RUN=1 CUDA=help "$INSTALLER")"
assert_contains "$help_output" "Usage:"
assert_not_contains "$help_output" "Installing with:"

if DRY_RUN=1 CUDA=unsupported "$INSTALLER" >/dev/null 2>&1; then
    echo "Unsupported CUDA value unexpectedly succeeded" >&2
    exit 1
fi
