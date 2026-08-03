#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# setup.sh -- provision a Brev VM-Mode Launchable for NeMo Safe Synthesizer.
# Paste into the Launchable's Setup Script field. Runs once, unprivileged,
# after Jupyter is installed; everything lands under $HOME. Idempotent.
#
# Launch parameters, passed by Brev as environment variables:
#   NSS_INFERENCE_KEY  NIM key for column classification (degraded without it)
#   HF_TOKEN           only for gated Hugging Face models
#
# script/brev/README.md explains every non-obvious step below. Each one exists
# because a deploy failed without it -- read it before simplifying anything.
set -euo pipefail

readonly UV_VERSION="0.9.30"
readonly PYTHON_VERSION="3.13"
# No .git suffix: this is only used to build archive URLs, not to clone.
readonly REPO_URL="https://github.com/NVIDIA-NeMo/Safe-Synthesizer"

: "${HOME:?HOME is not set}"

# $HOME is the file browser root: only tutorials/ and README.md are visible.
readonly TUTORIALS_DIR="${HOME}/tutorials"
readonly README_FILE="${HOME}/README.md"
readonly WAIT_FILE="${HOME}/SETUP-IN-PROGRESS.md"
readonly WELCOME_STAGED="${HOME}/.nss-welcome.md"

readonly BIN_DIR="${HOME}/.local/bin"
readonly VENV_DIR="${HOME}/.nss-venv"
readonly USER_KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/python3"
readonly ENV_FILE="${HOME}/.nss-env.sh"
readonly LOG_FILE="${HOME}/.nss-setup.log"

readonly CUDA_EXTRA="cu129"

mkdir -p "${BIN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { printf '\n=== [nss-setup] %s\n' "$*"; }
fail() {
  printf '\n!!! [nss-setup] FAILED at line %s\n' "$1" >&2
  # Truthful, rather than leaving the wait notice up forever.
  cat >"${WAIT_FILE}" <<EOF
# Setup failed

Provisioning stopped at line $1, so the environment is incomplete and the
tutorials will not run.

Full log: \`.nss-setup.log\` (hidden files are off by default in the browser).
EOF
}
trap 'fail "${LINENO}"' ERR

# Before any slow work: JupyterLab is reachable long before this finishes.
cat >"${WAIT_FILE}" <<'EOF'
# Setting up -- please wait

NeMo Safe Synthesizer is still installing -- roughly 5-10 minutes from when
the instance started. Files appear as it progresses, so a partly-filled file
browser is expected. Nothing here is ready to run yet.

When setup finishes, this file is replaced by README.md. Refresh to check.
EOF

export PATH="${BIN_DIR}:${PATH}"

log "user=$(id -un) home=${HOME}"

# Preflight: this Launchable only makes sense on an NVIDIA GPU instance.

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  log "WARNING: nvidia-smi not found. Training and generation will not work."
fi

# uv -- installed into ~/.local/bin, which needs no privileges.

if [[ "$(uv --version 2>/dev/null | awk '{print $2}')" != "${UV_VERSION}" ]]; then
  log "installing uv ${UV_VERSION} into ${BIN_DIR}"
  # Verify the published SHA-256; astral's installer verifies nothing.
  case "$(uname -m)" in
    x86_64) uv_target="x86_64-unknown-linux-gnu" ;;
    aarch64 | arm64) uv_target="aarch64-unknown-linux-gnu" ;;
    *)
      log "ERROR: unsupported architecture $(uname -m)"
      exit 1
      ;;
  esac
  uv_dir="$(mktemp -d)"
  uv_url="https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-${uv_target}.tar.gz"
  curl -fsSL "${uv_url}" -o "${uv_dir}/uv.tar.gz"
  uv_want="$(curl -fsSL "${uv_url}.sha256" | awk '{print $1}')"
  uv_got="$(sha256sum "${uv_dir}/uv.tar.gz" | awk '{print $1}')"
  if [[ "${uv_want}" != "${uv_got}" ]]; then
    log "ERROR: uv checksum mismatch (want ${uv_want}, got ${uv_got})"
    exit 1
  fi
  tar -xzf "${uv_dir}/uv.tar.gz" -C "${uv_dir}" --strip-components=1
  install -m 0755 "${uv_dir}/uv" "${uv_dir}/uvx" "${BIN_DIR}/"
  rm -rf "${uv_dir}"
else
  log "uv ${UV_VERSION} already present"
fi

# The cu129 wheels are large and their indexes are not always fast.
export UV_HTTP_TIMEOUT=300

# Venv kept separate from Jupyter's, so a failed install cannot break it.

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "creating venv at ${VENV_DIR} (python ${PYTHON_VERSION})"
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
  log "venv already present at ${VENV_DIR}"
fi

if "${VENV_DIR}/bin/safe-synthesizer" --version >/dev/null 2>&1; then
  log "safe-synthesizer already installed; skipping package install"
else
  NSS_VERSION="$(curl -fsSL https://pypi.org/pypi/nemo-safe-synthesizer/json \
    | "${VENV_DIR}/bin/python" -c 'import json, sys; print(json.load(sys.stdin)["info"]["version"])')"

  # Indexes come from the installed release's pyproject. Match both generated
  # names and URLs because static index names do not enforce the CUDA suffix.
  pyproject="$(mktemp)"
  curl -fsSL "${REPO_URL}/raw/v${NSS_VERSION}/pyproject.toml" -o "${pyproject}"
  index_args=()
  index_count=0
  while IFS= read -r url; do
    [[ -n "${url}" ]] || continue
    index_args+=(--index "${url}")
    index_count=$((index_count + 1))
    log "index: ${url}"
  done < <(env CUDA_EXTRA="${CUDA_EXTRA}" "${VENV_DIR}/bin/python" - "${pyproject}" <<'PY'
import os
import sys
import tomllib

with open(sys.argv[1], "rb") as handle:
    indexes = tomllib.load(handle)["tool"]["uv"]["index"]

cuda_extra = os.environ["CUDA_EXTRA"]
print(
    "\n".join(
        index["url"]
        for index in indexes
        if index["name"].endswith(f"-{cuda_extra}")
        or f"/{cuda_extra}" in index["url"]
    )
)
PY
  )
  rm -f "${pyproject}"

  # The parse cannot fail the script from a process substitution; count validates.
  if [[ "${index_count}" -lt 3 ]]; then
    log "ERROR: expected 3+ ${CUDA_EXTRA} indexes in v${NSS_VERSION}, got ${index_count}"
    exit 1
  fi

  log "installing nemo-safe-synthesizer ${NSS_VERSION} -- this takes several minutes"
  VIRTUAL_ENV="${VENV_DIR}" uv pip install \
    "nemo-safe-synthesizer[${CUDA_EXTRA},engine]==${NSS_VERSION}" \
    "${index_args[@]}" \
    --index-strategy unsafe-best-match
fi

# Checked separately, or a rerun skips it and the kernel cannot start.
if ! "${VENV_DIR}/bin/python" -c "import ipykernel, ipywidgets" 2>/dev/null; then
  log "installing notebook support"
  VIRTUAL_ENV="${VENV_DIR}" uv pip install ipykernel ipywidgets
fi

# Tutorials: tarball extract pinned to the installed release. See README.

if [[ -f "${TUTORIALS_DIR}/.fetched" ]]; then
  log "tutorials already present"
else
  # Set above if this run installed; read off the package if the install ran previously.
  NSS_VERSION="${NSS_VERSION:-$("${VENV_DIR}/bin/python" -c \
    'from importlib.metadata import version; print(version("nemo-safe-synthesizer"))')}"
  log "fetching tutorials for version ${NSS_VERSION}"

  # Staged under $HOME: same-filesystem rename, and never half-visible.
  tarball_dir="$(mktemp -d "${HOME}/.nss-fetch.XXXXXX")"
  tarball="${tarball_dir}/src.tar.gz"
  stage="${tarball_dir}/stage"
  fetched=0

  for ref in "refs/tags/v${NSS_VERSION}" "refs/heads/main"; do
    if ! curl -fsSL "${REPO_URL}/archive/${ref}.tar.gz" -o "${tarball}"; then
      log "WARNING: ${ref} not downloadable; trying next ref"
      continue
    fi

    # Exact member path, and no pipe to head: GNU/BSD tar traps. See README.
    listing="$(tar -tzf "${tarball}")"
    top="${listing%%/*}"
    rm -rf "${stage}"
    mkdir -p "${stage}"
    if tar -xzf "${tarball}" -C "${stage}" --strip-components=3 \
      "${top}/docs/tutorials"; then
      rm -rf "${TUTORIALS_DIR}"
      mv "${stage}" "${TUTORIALS_DIR}"
      # Written last; the guard keys on this, so partial runs are redone.
      : >"${TUTORIALS_DIR}/.fetched"
      log "tutorials extracted from ${ref}"
      # Same tarball as the tutorials. Non-fatal -- see README.
      if tar -xzf "${tarball}" -C "${tarball_dir}" --strip-components=3 \
        "${top}/script/brev/welcome.md" 2>/dev/null; then
        mv "${tarball_dir}/welcome.md" "${WELCOME_STAGED}"
      else
        log "WARNING: welcome.md not present in ${ref}"
      fi
      fetched=1
      break
    fi
    log "WARNING: no tutorials in ${ref}; trying next ref"
  done

  rm -rf "${tarball_dir}"
  if [[ "${fetched}" -ne 1 ]]; then
    log "ERROR: could not fetch tutorial notebooks"
    exit 1
  fi
fi

# Environment for terminals; the kernelspec below covers notebooks.

log "writing ${ENV_FILE}"
{
  echo "# Managed by setup.sh -- do not edit."
  echo "export PATH=\"${VENV_DIR}/bin:${BIN_DIR}:\${PATH}\""
  # Pin VIRTUAL_ENV or uv walks up and finds Brev's own ~/.venv instead.
  echo "export VIRTUAL_ENV=\"${VENV_DIR}\""
  if [[ -n "${NSS_INFERENCE_KEY:-}" ]]; then
    printf 'export NSS_INFERENCE_KEY=%q\n' "${NSS_INFERENCE_KEY}"
  fi
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf 'export HF_TOKEN=%q\n' "${HF_TOKEN}"
    printf 'export HUGGING_FACE_HUB_TOKEN=%q\n' "${HF_TOKEN}"
  fi
} >"${ENV_FILE}"
chmod 0600 "${ENV_FILE}"

if ! grep -qF "${ENV_FILE}" "${HOME}/.bashrc" 2>/dev/null; then
  printf '\n[ -f %s ] && . %s\n' "${ENV_FILE}" "${ENV_FILE}" >>"${HOME}/.bashrc"
fi

# Kernel registered as `python3` (the name the notebooks declare), in the
# directory the server actually consults -- not the user one. See README.

log "building kernelspec"
KERNEL_JSON="$(mktemp)"
# `env`, because bash rejects prefix assignments to readonly variables.
env VENV_DIR="${VENV_DIR}" BIN_DIR="${BIN_DIR}" \
  "${VENV_DIR}/bin/python" - "${KERNEL_JSON}" <<'PY'
import json
import os
import sys

venv = os.environ["VENV_DIR"]
env = {
    "PATH": f"{venv}/bin:{os.environ['BIN_DIR']}:/usr/local/bin:/usr/bin:/bin",
    # Keeps `!uv pip install ...` in a notebook from resolving to the Brev
    # image's own ~/.venv, which uv would otherwise discover by walking up.
    "VIRTUAL_ENV": venv,
}
# Secrets live in the kernelspec because the Jupyter server is not launched
# from a login shell. The VM is single-tenant and the file is mode 0600.
for key in ("NSS_INFERENCE_KEY", "HF_TOKEN"):
    value = os.environ.get(key)
    if value:
        env[key] = value
if "HF_TOKEN" in env:
    env["HUGGING_FACE_HUB_TOKEN"] = env["HF_TOKEN"]

spec = {
    "argv": [f"{venv}/bin/python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Safe Synthesizer",
    "language": "python",
    "env": env,
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(spec, handle, indent=2)
PY

# Find the interpreter the Jupyter server runs on, so we can ask it -- rather
# than guess -- where its highest-precedence kernels directory is.
JUPYTER_PY=""
if command -v jupyter >/dev/null 2>&1; then
  candidate="$(dirname "$(command -v jupyter)")/python"
  if [[ -x "${candidate}" ]]; then
    JUPYTER_PY="${candidate}"
  fi
fi
if [[ -z "${JUPYTER_PY}" && -x "${HOME}/.venv/bin/python" ]]; then
  JUPYTER_PY="${HOME}/.venv/bin/python"
fi

kernel_targets=("${USER_KERNEL_DIR}")
if [[ -n "${JUPYTER_PY}" ]]; then
  primary="$("${JUPYTER_PY}" -c \
    'from jupyter_core.paths import jupyter_path; print(jupyter_path("kernels")[0])' \
    2>/dev/null || true)"
  if [[ -n "${primary}" ]]; then
    kernel_targets=("${primary}/python3" "${kernel_targets[@]}")
  fi
else
  log "WARNING: could not locate the Jupyter interpreter; kernel may not be picked up"
fi

# First writable target wins. Writing both would leave a duplicate python3
# kernelspec that never gets consulted and only confuses `kernelspec list`.
registered=0
for target in "${kernel_targets[@]}"; do
  # `mkdir -p` succeeds on an existing directory even without write permission,
  # so test writability separately -- otherwise an unwritable first target
  # reaches `cp`, and `set -e` aborts the run instead of trying the next one.
  mkdir -p "${target}" 2>/dev/null || { log "WARNING: cannot create ${target}"; continue; }
  if [[ ! -w "${target}" ]]; then
    log "WARNING: ${target} is not writable"
    continue
  fi
  # Keep whatever was there, once, so the original kernel stays recoverable.
  if [[ -f "${target}/kernel.json" && ! -f "${target}/kernel.json.orig" ]]; then
    cp "${target}/kernel.json" "${target}/kernel.json.orig" 2>/dev/null || true
  fi
  cp "${KERNEL_JSON}" "${target}/kernel.json" 2>/dev/null || {
    log "WARNING: cannot write ${target}/kernel.json"
    continue
  }
  chmod 0600 "${target}/kernel.json"
  log "registered kernel at ${target}"
  registered=1
  break
done
rm -f "${KERNEL_JSON}"
if [[ "${registered}" -ne 1 ]]; then
  log "WARNING: kernel not registered; notebooks may open on the wrong Python"
fi

# Smoke check -- fail provisioning loudly rather than handing over a broken VM.

log "verifying install"
"${VENV_DIR}/bin/safe-synthesizer" --version
"${VENV_DIR}/bin/python" \
  -c "import torch; print('cuda available:', torch.cuda.is_available())"

# Hand over: swap the "please wait" file for the welcome text.

if [[ -f "${WELCOME_STAGED}" ]]; then
  mv "${WELCOME_STAGED}" "${README_FILE}"
else
  log "WARNING: no welcome.md staged; skipping ${README_FILE}"
fi
rm -f "${WAIT_FILE}"

trap - ERR
log "setup complete"
cat <<EOF

  Safe Synthesizer is ready.

  Start here : ${README_FILE}
  Tutorials  : ${TUTORIALS_DIR}/
  Kernel     : "Safe Synthesizer" -- already the default
  Setup log  : ${LOG_FILE}

EOF
