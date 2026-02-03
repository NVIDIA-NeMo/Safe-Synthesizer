#!/usr/bin/env bash

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

if [ $OS != "darwin" ]; then
    echo "This script is only for Darwin (macOS) systems."
    exit 1
fi

install_gnutar() {
    brew install gnu-tar
    if [ $? -ne 0 ]; then
        echo "Failed to install gnu-tar. Please check your Homebrew installation."
        exit 1
    fi
    echo "gnu-tar installed successfully."
    echo "To use gnutar as the default tar, need to add it to your PATH:"
    echo 'export PATH="$(brew --prefix gnu-tar)/libexec/gnubin:$PATH"'
    echo "otherwise use it like: \`gtar\`"
}

install_gnutar