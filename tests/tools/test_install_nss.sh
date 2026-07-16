#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

assert_dry_run() {
    local cuda="$1"
    local expected_extra="$2"
    local expected_index="$3"
    local output
    output="$(DRY_RUN=1 CUDA="$cuda" "$INSTALLER")"

    assert_contains "$output" "nemo-safe-synthesizer\[engine\,${expected_extra}\]"
    assert_contains "$output" "--index"
    assert_contains "$output" "$expected_index"
}

assert_dry_run 129 cu129 "https://wheels.vllm.ai/"
assert_dry_run 130 cu130 "https://pypi.nvidia.com"
assert_dry_run cpu cpu "https://download.pytorch.org/whl/cpu"

if DRY_RUN=1 CUDA=unsupported "$INSTALLER" >/dev/null 2>&1; then
    echo "Unsupported CUDA value unexpectedly succeeded" >&2
    exit 1
fi
