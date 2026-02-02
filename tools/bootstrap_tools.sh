#!/usr/bin/env bash
set -eu

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
NMP_TOOL_BASE="${REPO_ROOT}/tools/binaries"
source "${REPO_ROOT}/tools/binaries/defs.sh"
source "${REPO_ROOT}/tools/binaries/common_functions.sh"

bootstrap_tools() {
  bootstrap=(kubectl)
  tools=(uv buildkit jq osv_scanner)
  mkdir -p "$HOME/.local/bin"
  add_to_path "$HOME/.local/bin"
  for tool in "${tools[@]}"; do
    bash "$NMP_TOOL_BASE/install_${tool}.sh" || echo "Failed to install $tool - continuing with other tools"
  done

  for tool in "${bootstrap[@]}"; do
    bash "$NMP_TOOL_BASE/bootstrap_${tool[0]}.sh" || echo "Failed to bootstrap $tool - continuing with other tools"
  done
}

bootstrap_tools
