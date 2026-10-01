#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs script/brev/setup.sh end to end against stubbed network, uv, GPU, and
# Jupyter commands. The release installer is real, so the Brev handoff to
# install_nss.sh is exercised; package resolution itself is covered elsewhere.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT SETUP="${REPO_ROOT}/script/brev/setup.sh"
readonly VERSION="0.1.14"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/nss-brev.XXXXXX")"
readonly test_dir
trap 'rm -rf "$test_dir"' EXIT
fake_bin="${test_dir}/bin"
jupyter_bin="${test_dir}/jupyter-bin"
mkdir -p "$fake_bin" "$jupyter_bin"
REAL_PYTHON="$(command -v python3)"
export REAL_PYTHON

fail() { echo "FAIL: $*" >&2; exit 1; }
assert_contains() { [[ "$1" == *"$2"* ]] || fail "expected to find '$2' in: $1"; }
assert_not_contains() { [[ "$1" != *"$2"* ]] || fail "did not expect '$2' in: $1"; }
assert_mode_600() { [[ -n "$(find "$1" -perm 0600)" ]] || fail "expected mode 0600: $1"; }

# Brev's setup script field has a hard size limit.
size="$(wc -c < "$SETUP")"
(( size < 16384 )) || fail "setup.sh is ${size} bytes; Brev rejects 16 KiB or more"

bash "${REPO_ROOT}/tools/build_release_installer.sh" "$VERSION" "${test_dir}/release" >/dev/null
export FAKE_INSTALLER="${test_dir}/release/install_nss.sh"

# A minimal archive shaped like GitHub's: one top-level dir, tutorials under docs/.
mkdir -p "${test_dir}/archive/Safe-Synthesizer-${VERSION}/docs/tutorials/datasets"
printf '{}\n' > "${test_dir}/archive/Safe-Synthesizer-${VERSION}/docs/tutorials/safe-synthesizer-101.ipynb"
printf 'a,b\n' > "${test_dir}/archive/Safe-Synthesizer-${VERSION}/docs/tutorials/datasets/sample.csv"
printf 'readme\n' > "${test_dir}/archive/Safe-Synthesizer-${VERSION}/README.md"
tar -czf "${test_dir}/source.tar.gz" -C "${test_dir}/archive" "Safe-Synthesizer-${VERSION}"
export FAKE_TARBALL="${test_dir}/source.tar.gz"

cat > "${fake_bin}/curl" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
out=""; url=""
while (( $# )); do
    case "$1" in
        -o) out="$2"; shift ;;
        -*) ;;
        *) url="$1" ;;
    esac
    shift
done
printf '%s\n' "$url" >> "${FAKE_CURL_LOG:?}"
case "$url" in
    */releases/latest/download/install_nss.sh) src="$FAKE_INSTALLER" ;;
    */archive/refs/tags/*.tar.gz) [[ -z "${FAKE_TAG_MISSING:-}" ]] || exit 22; src="$FAKE_TARBALL" ;;
    */archive/refs/heads/main.tar.gz) src="$FAKE_TARBALL" ;;
    *) echo "fake curl: unexpected URL $url" >&2; exit 22 ;;
esac
if [[ -n "$out" ]]; then cp "$src" "$out"; else cat "$src"; fi
EOF

# The venv's python wraps the host interpreter with a private site dir, so
# setup.sh's real Python snippets run against stub packages.
cat > "${fake_bin}/uv" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_UV_LOG:?}"
make_venv() {
    mkdir -p "$1/bin" "$1/site"
    printf '#!/usr/bin/env bash\nPYTHONPATH=%q exec %q "$@"\n' "$1/site" "$REAL_PYTHON" > "$1/bin/python"
    chmod +x "$1/bin/python"
}
case "${1:-}" in
    --version) echo "uv 0.9.30" ;;
    venv) make_venv "${!#}" ;;
    pip)
        venv="${VIRTUAL_ENV:-}"; prev=""
        for arg in "$@"; do
            [[ "$prev" == "--python" ]] && venv="$(dirname "$(dirname "$arg")")"
            [[ "$arg" == "--overrides" ]] && cat > /dev/null
            prev="$arg"
        done
        site="${venv:?}/site"
        case "$*" in
            *nemo-safe-synthesizer*)
                [[ -z "${FAKE_UV_FAIL_INSTALL:-}" ]] || exit 1
                mkdir -p "$site/nemo_safe_synthesizer-${FAKE_VERSION}.dist-info"
                printf 'Metadata-Version: 2.1\nName: nemo-safe-synthesizer\nVersion: %s\n' "$FAKE_VERSION" \
                    > "$site/nemo_safe_synthesizer-${FAKE_VERSION}.dist-info/METADATA"
                printf 'class cuda:\n    is_available = staticmethod(lambda: True)\n' > "$site/torch.py"
                printf '#!/usr/bin/env bash\necho %s\n' "$FAKE_VERSION" > "$venv/bin/safe-synthesizer"
                chmod +x "$venv/bin/safe-synthesizer"
                ;;
            *ipykernel*) touch "$site/ipykernel.py" "$site/ipywidgets.py" ;;
        esac
        ;;
    *) echo "fake uv: unexpected call: $*" >&2; exit 1 ;;
esac
EOF

cat > "${fake_bin}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
echo "NVIDIA H100 80GB HBM3, 81559 MiB, 580.65.06"
EOF

# Hosts without gcc or libc headers take the sudo apt-get path; never run it for real.
cat > "${fake_bin}/sudo" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "${FAKE_SUDO_LOG:?}"
EOF

# setup.sh finds the Jupyter server's interpreter next to `jupyter` and asks it
# for the highest-precedence kernels directory.
printf '#!/usr/bin/env bash\n' > "${jupyter_bin}/jupyter"
cat > "${jupyter_bin}/python" <<'EOF'
#!/usr/bin/env bash
echo "${FAKE_JUPYTER_KERNELS:?}"
EOF
chmod +x "$fake_bin"/* "$jupyter_bin"/*

# Usage: run_setup HOME [VAR=value...]. Command substitution at call sites waits
# for setup.sh's tee to close, so output is complete when it returns.
run_setup() {
    local home="$1"
    shift
    mkdir -p "$home"
    env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT -u CUDA -u NSS_INFERENCE_KEY -u HF_TOKEN \
        HOME="$home" PATH="${fake_bin}:${jupyter_bin}:${PATH}" \
        FAKE_VERSION="$VERSION" FAKE_UV_LOG="${home}.uv" FAKE_CURL_LOG="${home}.curl" \
        FAKE_SUDO_LOG="${home}.sudo" FAKE_JUPYTER_KERNELS="${home}/.venv/share/jupyter/kernels" \
        "$@" bash "$SETUP" 2>&1
}

# Fresh instance: installs through the release installer and hands over a ready home.
home="${test_dir}/fresh"
# Brev's image ships a world-readable python3 kernel where ours must land.
mkdir -p "${home}/.venv/share/jupyter/kernels/python3"
printf '{"display_name": "brev"}\n' > "${home}/.venv/share/jupyter/kernels/python3/kernel.json"
chmod 0644 "${home}/.venv/share/jupyter/kernels/python3/kernel.json"
output="$(run_setup "$home" NSS_INFERENCE_KEY=nim-key HF_TOKEN=hf-token)" || fail "fresh setup failed: $output"
assert_contains "$output" "setup complete"
uv_calls="$(<"${home}.uv")"
assert_contains "$uv_calls" "venv --python 3.13 ${home}/.nss-venv"
assert_contains "$uv_calls" "pip install nemo-safe-synthesizer[engine,cu129]==${VERSION}"
assert_contains "$uv_calls" "-c https://raw.githubusercontent.com/NVIDIA-NeMo/Safe-Synthesizer/v${VERSION}/constraints.txt"
assert_contains "$uv_calls" "--python ${home}/.nss-venv/bin/python"
assert_contains "$uv_calls" "--index https://download.pytorch.org/whl/cu129"
[[ "$(grep -c '^venv ' "${home}.uv")" == 1 ]] || fail "installer created a second venv: $uv_calls"
assert_contains "$(<"${home}.curl")" "/archive/refs/tags/v${VERSION}.tar.gz"
[[ -f "${home}/tutorials/safe-synthesizer-101.ipynb" && -f "${home}/tutorials/datasets/sample.csv" ]] ||
    fail "tutorials were not extracted flat into ~/tutorials"
[[ ! -e "${home}/tutorials/README.md" ]] || fail "extracted files outside docs/tutorials"
[[ ! -e "${home}/SETUP-IN-PROGRESS.md" ]] || fail "wait notice left behind after success"
kernel="${home}/.venv/share/jupyter/kernels/python3/kernel.json"
[[ -f "$kernel" ]] || fail "kernel not registered in the Jupyter server's primary kernels dir"
[[ ! -e "${home}/.local/share/jupyter/kernels/python3/kernel.json" ]] || fail "duplicate user-scope kernelspec"
assert_mode_600 "$kernel"
assert_contains "$(<"${kernel}.orig")" '"brev"'
"$REAL_PYTHON" - "$kernel" "${home}/.nss-venv" <<'PY' || fail "unexpected kernelspec: $(<"$kernel")"
import json, sys
spec = json.load(open(sys.argv[1]))
venv = sys.argv[2]
assert spec["argv"][0] == f"{venv}/bin/python", spec["argv"]
env = spec["env"]
assert env["VIRTUAL_ENV"] == venv
assert env["PATH"].startswith(f"{venv}/bin:")
assert env["NSS_INFERENCE_KEY"] == "nim-key"
assert env["HF_TOKEN"] == env["HUGGING_FACE_HUB_TOKEN"] == "hf-token"
PY
assert_mode_600 "${home}/.nss-env.sh"
assert_contains "$(<"${home}/.nss-env.sh")" "export VIRTUAL_ENV=\"${home}/.nss-venv\""
assert_contains "$(<"${home}/.nss-env.sh")" "export HUGGING_FACE_HUB_TOKEN=hf-token"

# Rerun: idempotent, with no reinstall, refetch, or duplicate .bashrc hook.
: > "${home}.uv"; : > "${home}.curl"
output="$(run_setup "$home")" || fail "rerun failed: $output"
assert_contains "$output" "setup complete"
assert_not_contains "$(<"${home}.uv")" "pip install"
[[ ! -s "${home}.curl" ]] || fail "rerun downloaded again: $(<"${home}.curl")"
[[ "$(grep -c '.nss-env.sh' "${home}/.bashrc")" == 1 ]] || fail "duplicate .bashrc hook"

# Missing release tag: tutorials fall back to main.
home="${test_dir}/tag-missing"
output="$(run_setup "$home" FAKE_TAG_MISSING=1)" || fail "tag fallback failed: $output"
assert_contains "$(<"${home}.curl")" "/archive/refs/heads/main.tar.gz"
[[ -f "${home}/tutorials/safe-synthesizer-101.ipynb" ]] || fail "tutorials missing after main fallback"

# Failed install: nonzero exit and a truthful notice instead of the wait message.
home="${test_dir}/install-failure"
if output="$(run_setup "$home" FAKE_UV_FAIL_INSTALL=1)"; then fail "setup succeeded despite install failure"; fi
assert_not_contains "$output" "setup complete"
assert_contains "$(<"${home}/SETUP-IN-PROGRESS.md")" "# Setup failed"
[[ ! -e "${home}/tutorials" ]] || fail "setup continued past the failed install"

echo "Brev setup tests passed"
