#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# setup.sh -- provision a Brev VM-Mode Launchable for NeMo Safe Synthesizer.
#
# Brev runs this once as the unprivileged instance user (the account name
# varies by provider), after the VM is created and Jupyter is installed. It
# installs the CUDA build of Safe Synthesizer into a dedicated venv, registers
# it as the default Jupyter kernel, and drops the tutorials in $HOME.
#
# Everything lands under $HOME; the script never requires root. Paste it into
# the Launchable's Setup Script field. See README.md in this directory for the
# console settings and the reasoning behind the non-obvious steps below.
#
# Launch parameters (Brev exposes them as environment variables):
#   NSS_INFERENCE_KEY  Optional. NVIDIA NIM key for column classification.
#                      Without it, classification runs in degraded mode.
#   HF_TOKEN           Optional. Needed only for gated Hugging Face models.
#
# The script is idempotent: rerunning it is safe and skips completed stages.

set -euo pipefail

readonly UV_VERSION="0.9.30"
readonly PYTHON_VERSION="3.13"
# No .git suffix: this is only used to build archive URLs, not to clone.
readonly REPO_URL="https://github.com/NVIDIA-NeMo/Safe-Synthesizer"

: "${HOME:?HOME is not set}"

# $HOME is the JupyterLab file browser root, so only tutorials/ and README.md
# are visible there; everything operational is a dotfile. No data folder is
# created -- users keep their files wherever they like.
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
  # Leave the waiting user something truthful rather than "please wait" forever.
  cat >"${WAIT_FILE}" <<EOF
# Setup failed

Provisioning stopped at line $1, so the environment is incomplete and the
tutorials will not run.

Full log: \`.nss-setup.log\` (hidden files are off by default in the browser).
EOF
}
trap 'fail "${LINENO}"' ERR

# Written before any slow work: JupyterLab accepts connections well before
# this script finishes, and an empty home looks like a broken Launchable.
cat >"${WAIT_FILE}" <<'EOF'
# Setting up -- please wait

NeMo Safe Synthesizer is still installing -- roughly 5-10 minutes from when
the instance started. Files appear as it progresses, so a partly-filled file
browser is expected. Nothing here is ready to run yet.

When setup finishes, this file is replaced by README.md. Refresh to check.
EOF

export PATH="${BIN_DIR}:${PATH}"

log "user=$(id -un) home=${HOME}"

# ---------------------------------------------------------------------------
# Preflight: this Launchable only makes sense on an NVIDIA GPU instance.
# ---------------------------------------------------------------------------

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  log "WARNING: nvidia-smi not found. Training and generation will not work."
fi

# ---------------------------------------------------------------------------
# uv -- installed into ~/.local/bin, which needs no privileges.
# ---------------------------------------------------------------------------

if [[ "$(uv --version 2>/dev/null | awk '{print $2}')" != "${UV_VERSION}" ]]; then
  log "installing uv ${UV_VERSION} into ${BIN_DIR}"
  # Verify the published SHA-256 rather than piping astral.sh/install.sh into
  # a shell -- that installer logs "no checksums to verify". See README.
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

# ---------------------------------------------------------------------------
# Virtual environment -- deliberately separate from the one Jupyter runs on, so
# a failed cu129 install cannot take the notebook server down with it.
# ---------------------------------------------------------------------------

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

  # Index URLs are install-time config, not wheel metadata, so they must match
  # the release being installed rather than this repo's main. Read them from
  # that release's own pyproject.toml, keyed on URL not index name. See README.
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
print("\n".join(i["url"] for i in indexes if os.environ["CUDA_EXTRA"] in i["url"]))
PY
  )
  rm -f "${pyproject}"

  # The parse runs in a process substitution, so it cannot fail the script.
  # The count is what actually validates it.
  if [[ "${index_count}" -ne 3 ]]; then
    log "ERROR: expected 3 ${CUDA_EXTRA} indexes in v${NSS_VERSION}, got ${index_count}"
    exit 1
  fi

  log "installing nemo-safe-synthesizer ${NSS_VERSION} -- this takes several minutes"
  VIRTUAL_ENV="${VENV_DIR}" uv pip install \
    "nemo-safe-synthesizer[${CUDA_EXTRA},engine]==${NSS_VERSION}" \
    "${index_args[@]}" \
    --index-strategy unsafe-best-match
fi

# Checked separately from the package install: if this step is what failed, a
# rerun would otherwise see a working safe-synthesizer, skip the whole block,
# and leave a registered kernel that cannot start.
if ! "${VENV_DIR}/bin/python" -c "import ipykernel, ipywidgets" 2>/dev/null; then
  log "installing notebook support"
  VIRTUAL_ENV="${VENV_DIR}" uv pip install ipykernel ipywidgets
fi

# ---------------------------------------------------------------------------
# Tutorial notebooks. Tarball extract rather than a clone, so they land flat in
# tutorials/ and git is not required. Pinned to the tag matching the installed
# wheel so the notebooks cannot outrun the package. See README.md.
# ---------------------------------------------------------------------------

if compgen -G "${TUTORIALS_DIR}/*.ipynb" >/dev/null 2>&1; then
  log "tutorials already present"
else
  # Set above if this run installed; read off the package if the install ran previously.
  NSS_VERSION="${NSS_VERSION:-$("${VENV_DIR}/bin/python" -c \
    'from importlib.metadata import version; print(version("nemo-safe-synthesizer"))')}"
  log "fetching tutorials for version ${NSS_VERSION}"

  # Staged under $HOME so the final move is a rename on the same filesystem,
  # and so a half-written extract never becomes the visible tutorials/ -- the
  # ".ipynb exists" check above would otherwise treat a partial result as done.
  tarball_dir="$(mktemp -d "${HOME}/.nss-fetch.XXXXXX")"
  tarball="${tarball_dir}/src.tar.gz"
  stage="${tarball_dir}/stage"
  fetched=0

  for ref in "refs/tags/v${NSS_VERSION}" "refs/heads/main"; do
    if ! curl -fsSL "${REPO_URL}/archive/${ref}.tar.gz" -o "${tarball}"; then
      log "WARNING: ${ref} not downloadable; trying next ref"
      continue
    fi

    # Exact member path, not a glob, and the listing read into a variable
    # rather than piped to `head` -- both are GNU/BSD tar portability traps
    # documented in README.md. `%%/*` yields the archive's top-level directory.
    listing="$(tar -tzf "${tarball}")"
    top="${listing%%/*}"
    rm -rf "${stage}"
    mkdir -p "${stage}"
    if tar -xzf "${tarball}" -C "${stage}" --strip-components=3 \
      "${top}/docs/tutorials"; then
      rm -rf "${TUTORIALS_DIR}"
      mv "${stage}" "${TUTORIALS_DIR}"
      log "tutorials extracted from ${ref}"
      # Same tarball, so the welcome text matches the tutorials. Staged
      # hidden, moved into place at the end. Non-fatal -- see README.
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

# ---------------------------------------------------------------------------
# Environment. Set in two places: this file for terminals, and the kernelspec
# below for notebooks, since the Jupyter server never reads shell rc files.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Jupyter kernel. Registered as `python3` -- the name the notebooks declare --
# so nobody has to switch kernels. Which directory wins is not obvious, and the
# user one usually loses; see README.md. Ask the server rather than guess.
# ---------------------------------------------------------------------------

log "building kernelspec"
KERNEL_JSON="$(mktemp)"
# `env` rather than a command-prefix assignment: bash rejects prefix
# assignments to readonly variables, and these three are readonly.
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

# ---------------------------------------------------------------------------
# Smoke check -- fail provisioning loudly rather than handing over a broken VM.
# ---------------------------------------------------------------------------

log "verifying install"
"${VENV_DIR}/bin/safe-synthesizer" --version
"${VENV_DIR}/bin/python" \
  -c "import torch; print('cuda available:', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
# Hand over: swap the "please wait" file for the welcome text.
# ---------------------------------------------------------------------------

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
