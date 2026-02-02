#!/usr/bin/env bash
set -eu

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

print_buildkit_instructions() {
    echo "buildkit has been installed"
    echo "to use buildkit, you need to create a builder and use it:"
    echo "e.g.:"
    echo "docker buildx create --bootstrap --use --name local-docker"
    echo "docker buildx use local-docker"
    echo "see the readme for more details"
}

install_buildkit() {
  version="0.23.2"
	if command -v docker buildx >/dev/null; then
		echo "docker buildkit is already installed"
		exit 0
	fi
  echo "Installing docker buildkit version ${version}..."


  if [ "$OS" == "darwin" ]; then
      brew install buildkit
  elif [ "$OS" == "linux" ]; then
    buildkit_arch=$(uname -m)
    if [ "$buildkit_arch" == "x86_64" ]; then
      buildkit_arch="amd64"
    elif [ "$buildkit_arch" == "aarch64" ]; then
      buildkit_arch="arm64"
    fi
      (
        workdir=$(mktemp -d) &&
        cd "$workdir" &&
        curl -L "https://github.com/moby/buildkit/releases/download/v${version}/buildkit-v${version}.linux-${buildkit_arch}.tar.gz" -o buildkit.tar.gz &&
        tar -xzf buildkit.tar.gz &&
        mv bin/buildctl "$HOME/.local/bin/buildctl" &&
        cd "$HOME" &&
        rm -rf "$workdir"
      )
  fi
  print_buildkit_instructions
}


install_buildkit
