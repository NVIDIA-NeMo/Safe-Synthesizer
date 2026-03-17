#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "$REPO_ROOT/tools/binaries/defs.sh"
test_image="python:3.13"


test_tools_linux() {
    echo "Testing tool installation..."
    cd "$REPO_ROOT/tools/binaries" || {
        echo "can't find directory $REPO_ROOT/tools/binaries"
        exit 1
    }

    local worktree_volumes=()
    local git_common_dir
    git_common_dir="$(cd "$REPO_ROOT" && git rev-parse --git-common-dir)"
    if [[ "$git_common_dir" != ".git" ]]; then
        git_common_dir="$(cd "$REPO_ROOT" && cd "$git_common_dir" && pwd)"
        echo "Worktree detected, mounting git dir: $git_common_dir"
        worktree_volumes=(--volume "$git_common_dir:$git_common_dir:ro")
    fi

    docker run \
        --rm \
        --interactive \
        --name test_tool_install \
        --volume "$REPO_ROOT":/safe-synthesizer \
        "${worktree_volumes[@]}" \
        -e DEBIAN_FRONTEND=noninteractive \
        -e REPO_ROOT=/safe-synthesizer \
        --platform linux/amd64 \
        "$test_image" \
        bash -c "
            apt-get update &&
            apt-get install -y curl build-essential &&
            cd /safe-synthesizer &&
            make bootstrap-tools &&
            export PATH=\$HOME/.local/bin:\${PATH:+\${PATH}:} &&
            echo '=== Verifying installed tools ===' &&
            yq --version &&
            jq --version &&
            osv-scanner --version &&
            buildctl --version &&
            direnv --version &&
            uv --version &&
            gh --version
            " || {
        echo "Failed to test tool installation"
        exit 1
        }
    echo "Tool installation succeeded"
    exit 0
}

test_tools_linux
