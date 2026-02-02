#!/usr/bin/env bash

REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}
source "${REPO_ROOT}/tools/binaries/defs.sh"

_install_latest_uv() {
    echo "Installing the latest uv version..."
    curl -LsSf "https://astral.sh/uv/install.sh" | sh
    source "$HOME/.local/bin/env"

    echo "Installed: $(uv --version)"
}

install_uv() {
  echo "installing uv..."
  uv_version=$(grep "required-version" "${REPO_ROOT}/pyproject.toml" | sed 's/required-version = "[~>=<]\{2,3\}\(.*\)"/\1/') || {
    echo "cannot parse uv_version correctly from pyproject.toml"
    exit 1
  }
  uvs="$(which -a uv | uniq )"

  if [[ ${#uvs[*]} -eq 0 ]]; then
    _install_latest_uv
  else
    echo "validating uv version: ${uv_version}"
    echo "found the following uv installations:"
    echo "${uvs}"
    virtualenv=${VIRTUAL_ENV:-"venv"}
    # we want the uv that is not in a virtual environment
    # so we filter out the ones that contain the virtualenv path
    # and any that show up like `<path>//uv` which are usually because of a duplicate in PATH
    non_venv_uv=$(grep -i --invert-match "$virtualenv" <<< "$uvs" | grep --invert-match '//')
    if [[ -n "$non_venv_uv" ]]; then
      echo "ensuring $non_venv_uv is at least at version ${uv_version}"
      current_version=$("$non_venv_uv" --version | awk '{print $2}')
      # If the current version is less than the required version, update uv to the latest one
      if [[ "$(printf '%s\n' "${uv_version}" "${current_version}" | sort -V | head -n1)" != "${uv_version}" ]]; then
          $non_venv_uv self update
      else
        echo "uv is installed outside of venv at $non_venv_uv"
      fi
    else
      echo "only venv uv is found; installing non-venv uv"
      _install_latest_uv
    fi
  fi
  echo "to use non-venv uv by default, add the following to your shell config file (e.g., ~/.bashrc or ~/.zshrc):"
  echo "alias uv='$(which uv)'"
  echo ""
  printf "done installing uv"
}

install_uv
