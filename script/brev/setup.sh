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

# $HOME is the JupyterLab file browser root, so anything visible there is part
# of the first impression. Only tutorials/ and README.md are; everything
# operational is a dotfile, which the browser hides by default. Users pick
# their own location for data -- the script does not create a folder for it.
readonly TUTORIALS_DIR="${HOME}/tutorials"
readonly README_FILE="${HOME}/README.md"

readonly BIN_DIR="${HOME}/.local/bin"
readonly VENV_DIR="${HOME}/.nss-venv"
readonly USER_KERNEL_DIR="${HOME}/.local/share/jupyter/kernels/python3"
readonly ENV_FILE="${HOME}/.nss-env.sh"
readonly LOG_FILE="${HOME}/.nss-setup.log"

# vLLM publishes cu129 wheels from a pinned commit rather than a release index.
# Keep these three in sync with the install command in docs/user-guide/getting-started.md.
readonly INDEX_FLASHINFER="https://flashinfer.ai/whl/cu129"
readonly INDEX_TORCH="https://download.pytorch.org/whl/cu129"
readonly INDEX_VLLM="https://wheels.vllm.ai/ee0da84ab9e04ac7610e28580af62c365e898389/cu129"

mkdir -p "${BIN_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() { printf '\n=== [nss-setup] %s\n' "$*"; }
fail() { printf '\n!!! [nss-setup] FAILED at line %s\n' "$1" >&2; }
trap 'fail "${LINENO}"' ERR

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
  # Fetch the release tarball and verify its published SHA-256 rather than
  # piping astral.sh/install.sh into a shell -- that installer logs "no
  # checksums to verify", so nothing validates what it downloads. Mirrors the
  # GPG-verified mise install in tools/install-mise.sh.
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
  log "installing nemo-safe-synthesizer[cu129,engine] -- this takes several minutes"
  VIRTUAL_ENV="${VENV_DIR}" uv pip install "nemo-safe-synthesizer[cu129,engine]" \
    --index "${INDEX_FLASHINFER}" \
    --index "${INDEX_TORCH}" \
    --index "${INDEX_VLLM}" \
    --index-strategy unsafe-best-match
fi

# Checked separately from the package install: if this step is what failed, a
# rerun would otherwise see a working safe-synthesizer, skip the whole block,
# and leave a registered kernel that cannot start.
if ! "${VENV_DIR}/bin/python" -c "import ipykernel" 2>/dev/null; then
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
  NSS_VERSION="$("${VENV_DIR}/bin/python" -c \
    'from importlib.metadata import version; print(version("nemo-safe-synthesizer"))')"
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
# Customer-facing README, rendered by JupyterLab on double-click.
# ---------------------------------------------------------------------------

log "writing ${README_FILE}"
cat >"${README_FILE}" <<EOF
# NeMo Safe Synthesizer

Create private, safe versions of sensitive tabular data -- entirely synthetic
records with no one-to-one mapping back to your originals.

Everything is installed and ready. Nothing to set up.

## Start here

Open \`tutorials/safe-synthesizer-101.ipynb\` and run the cells top to bottom.
It takes about 15 minutes and walks through the full pipeline on a sample
dataset.

The other two notebooks go deeper:

- \`tutorials/differential-privacy.ipynb\` -- formal privacy guarantees (~1 hour)
- \`tutorials/time-series-financial-transactions.ipynb\` -- sequential data (~20 minutes)

## Using your own data

Upload a CSV anywhere you like -- drag and drop into the file browser on the
left -- and point the notebook at it:

\`\`\`python
from nemo_safe_synthesizer import SafeSynthesizer

results = SafeSynthesizer().with_data_source("my-data.csv").run()
\`\`\`

There is no required location for your files. Put them wherever suits you.

## Good to know

- Notebooks already run on the right Python. You should never need to change
  the kernel; if you do switch it, pick **Safe Synthesizer**.
- Model weights download on first use, so the first run is slower than later
  ones.
- This instance bills continuously and cannot be paused. **Delete it when you
  are finished**, and download anything you want to keep first.
- Provisioning log: \`.nss-setup.log\` (hidden files are off by default in the
  file browser).

## Learn more

- Documentation: <https://nvidia-nemo.github.io/Safe-Synthesizer/>
- Source: <${REPO_URL}>
EOF

trap - ERR
log "setup complete"
cat <<EOF

  Safe Synthesizer is ready.

  Start here : ${README_FILE}
  Tutorials  : ${TUTORIALS_DIR}/
  Kernel     : "Safe Synthesizer" -- already the default
  Setup log  : ${LOG_FILE}

EOF
