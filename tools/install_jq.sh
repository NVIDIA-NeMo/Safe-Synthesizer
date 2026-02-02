#!/usr/bin/env bash
set -eu

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

install_jq() {
	if command -v jq >/dev/null; then
		echo "jq is already installed"
	else
    echo "Installing jq..."
    jq_version="1.8.1"
    mkdir -p "$HOME/.local/bin"
    echo "Installing jq to $HOME/.local/bin"
    jq_arch=$(uname -m)
    if [ "$jq_arch" == "x86_64" ]; then
      jq_arch="amd64"
    elif [ "$jq_arch" == "aarch64" ]; then
      jq_arch="arm64"
    fi

		if [ "$OS" == "linux" ]; then
      wget -q "https://github.com/jqlang/jq/releases/download/jq-${jq_version}/jq-linux-${jq_arch}" -O jq
      chmod +x jq
      mv jq "$HOME/.local/bin/jq"
		elif [ "$OS" == "darwin" ]; then
      wget -q "https://github.com/jqlang/jq/releases/download/jq-${jq_version}/jq-macos-arm64" -O jq
      chmod +x jq
      mv jq "$HOME/.local/bin/jq"
		else
			echo "Error: Unsupported OS: $OS"
			exit 1
		fi
	fi
}

install_jq