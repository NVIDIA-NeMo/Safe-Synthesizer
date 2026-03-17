#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"
source "${REPO_ROOT}/tools/binaries/common_functions.sh"

_install_uv() {
    local version="${1:-}"
    if [[ -n "$version" ]]; then
        echo "Installing uv version ${version}..."
        curl -LsSf "https://astral.sh/uv/${version}/install.sh" | sh
    else
        echo "Installing the latest uv version..."
        curl -LsSf "https://astral.sh/uv/install.sh" | sh
    fi
    # shellcheck disable=SC1091
    [[ -f "$HOME/.local/bin/env" ]] && source "$HOME/.local/bin/env"

    echo "Installed: $(uv --version)"
}

_find_non_venv_uv() {
  local venv="${VIRTUAL_ENV:-}"
  type -a -P uv 2>/dev/null | awk -v venv="${venv}" '
    seen[$0]++          { next }   # skip duplicates
    /\/\//              { next }   # skip malformed double-slash paths
    venv != "" &&
      index($0, venv)  { next }   # skip paths inside the active virtualenv
                        { print; exit }
  ' || true
}

install_uv() {
  print_tool_manager_transition_warning
  echo "installing uv..."
  uv_constraint=$(sed -n 's/^[[:space:]]*required-version[[:space:]]*=[[:space:]]*"\([^"]*\)".*/\1/p' "${REPO_ROOT}/pyproject.toml") || {
    echo "cannot parse uv required-version correctly from pyproject.toml"
    exit 1
  }
  if [[ -z "${uv_constraint}" ]]; then
    echo "uv required-version is not set in pyproject.toml"
    exit 1
  fi

  uv_min_version=$(sed -n 's/.*>=[[:space:]]*\([0-9][0-9.]*\).*/\1/p' <<< "${uv_constraint}")
  uv_max_version=$(sed -n 's/.*<[[:space:]]*\([0-9][0-9.]*\).*/\1/p' <<< "${uv_constraint}")
  if [[ -z "${uv_min_version}" ]]; then
    echo "cannot parse minimum uv version from required-version: ${uv_constraint}"
    exit 1
  fi

  echo "looking for uv version \">=${uv_min_version}${uv_max_version:+, <${uv_max_version}}\""

  uvs="$(type -a -P uv 2>/dev/null | awk '!seen[$0]++' || true)"

  if [[ -z "$uvs" ]]; then
    echo "no uv found on PATH, installing uv ${uv_min_version}"
    _install_uv "$uv_min_version"
  else
    echo "found the following uv installations:"
    echo "${uvs}"
    non_venv_uv=$(_find_non_venv_uv)
    if [[ -n "$non_venv_uv" ]]; then
      current_version=$("$non_venv_uv" --version | awk '{print $2}')
      if version_in_range "${current_version}" "${uv_min_version}" "${uv_max_version}"; then
        echo "uv version ${current_version} found at ${non_venv_uv}, skipping install"
      else
        echo "uv version ${current_version} found at ${non_venv_uv}, updating to ${uv_min_version}"
        _install_uv "$uv_min_version"
      fi
    else
      echo "only venv uv found, installing uv ${uv_min_version}"
      _install_uv "$uv_min_version"
    fi
  fi

  selected_uv="$(_find_non_venv_uv)"
  if [[ -z "${selected_uv}" ]]; then
    echo "error: no non-venv uv executable found in PATH after installation"
    exit 1
  fi
  selected_version=$("${selected_uv}" --version | awk '{print $2}')
  if ! version_in_range "${selected_version}" "${uv_min_version}" "${uv_max_version}"; then
    echo "error: no non-venv uv satisfies required-version '${uv_constraint}'"
    echo "found ${selected_uv} at version ${selected_version}"
    echo "please put a compatible uv first on PATH before running bootstrap"
    exit 1
  fi

  echo "to use non-venv uv by default, add the following to your shell config file (e.g., ~/.bashrc or ~/.zshrc):"
  echo "alias uv='${selected_uv}'"
  echo ""
  printf "done installing uv"
}

install_uv
