#!/usr/bin/env bash

# shellcheck disable=SC2034
DEV="${DEV:-"$HOME/dev"}"
# shellcheck disable=SC2034
AIRE_REPOS=${AIRE_REPOS:-"$HOME/dev/aire"}
OS="$(uname | tr '[:upper:]' '[:lower:]')"
# shellcheck disable=SC2034
ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/\(arm\)\(64\)\?.*/\1\2/' -e 's/aarch64$/arm64/')"
# shellcheck disable=SC2034
