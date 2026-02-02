#!/usr/bin/env bash

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "$REPO_ROOT/tools/binaries/defs.sh"
test_image="python:3.13"


test_tools_linux() {
    echo "Testing tool installation..."
    cd "$REPO_ROOT/tools/binaries" || {
        echo "can't find directory $REPO_ROOT/tools/binaries"
        exit 1
    }

    docker run \
        --rm \
        --interactive \
        --name test_tool_install \
        --volume $REPO_ROOT:/nmp \
        -e SETUP_KUBE_AUTH=false \
        -e DEBIAN_FRONTEND=noninteractive \
        --platform linux/amd64 \
        $test_image \
        bash -c "
            apt-get update &&
            apt-get install -y git curl build-essential &&
            git config --global --add safe.directory /nmp &&
            cd /nmp &&
            make bootstrap-tools &&
            export KREW_ROOT=\$HOME/.krew &&
            export PATH=\${KREW_ROOT}/bin:\$HOME/.local/bin:\${PATH:+\${PATH}:} &&
            kubectl krew version &&
            uv --version &&
            jq --version &&
            k9s version
            " || {
        echo "Failed to test tool installation"
        exit 1
        }
    echo "Tool installation succeeded"
    exit 0
}

test_tools_linux