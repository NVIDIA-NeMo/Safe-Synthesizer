#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

maybe_install_brew_dep() {
    dep=$1
    if ! brew list -1 | grep -q "$dep"; then
        brew install "$dep"
    fi
}

set -ue
## adds a directory to the PATH if it exists and is not already in the PATH
# If the second argument is "after", it adds the directory to the end of the PATH
# Otherwise, it adds the directory to the beginning of the PATH.
add_to_path() {
  new_path=${1%/}
  if [ -d "$1" ] && ! echo "$PATH" | grep -E -q "(^|:)$new_path($|:)" ; then
      if [ "$2" = "after" ] ; then
          PATH="${PATH:+${PATH}:}$new_path"
      else
          PATH="$new_path:${PATH:+${PATH}:}"
      fi
  fi
}

version_at_least() {
  local current="${1}"
  local required="${2}"
  [[ "$(printf '%s\n' "${required}" "${current}" | sort -V | head -n1)" == "${required}" ]]
}

version_less_than() {
  local current="${1}"
  local upper_bound="${2}"
  [[ "$(printf '%s\n' "${current}" "${upper_bound}" | sort -V | head -n1)" == "${current}" ]] && [[ "${current}" != "${upper_bound}" ]]
}

version_in_range() {
  local current="${1}"
  local min_version="${2}"
  local max_version="${3:-}"

  version_at_least "${current}" "${min_version}" || return 1
  if [[ -n "${max_version}" ]]; then
    version_less_than "${current}" "${max_version}" || return 1
  fi
}

version_matches_exact() {
  local current="${1}"
  local required="${2}"
  [[ "${current}" == "${required}" ]]
}

print_tool_manager_transition_warning() {
  echo "warning: bootstrap tooling may migrate to Mise en place in a future update."
  echo "warning: these scripts remain supported for now and are the current source of truth."
}

set +ue
