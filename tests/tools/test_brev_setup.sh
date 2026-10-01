#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Checks that script/brev/setup.sh still works with what a release ships:
# install_nss.sh and the docs/tutorials folder it extracts. Changes to setup.sh
# itself are verified by deploying on Brev; these two can break it without
# touching it.
#
# The Brev Launchable downloads the latest *released* install_nss.sh and runs
# it. This test builds that release asset from this checkout, so a PR that
# changes the installer in a way that breaks Brev fails here before release.
#
# Only the installer is real. Network, uv, sudo, and Jupyter are stubbed, so
# nothing is downloaded or installed; dependency resolution is covered by the
# GPU CI jobs, which also install through install_nss.sh.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT SETUP="${REPO_ROOT}/script/brev/setup.sh"
readonly VERSION="0.1.14"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/nss-brev.XXXXXX")"
readonly test_dir
trap 'rm -rf "$test_dir"' EXIT
fake_bin="${test_dir}/bin"
mkdir -p "$fake_bin"
REAL_PYTHON="$(command -v python3)"
export REAL_PYTHON

fail() { echo "FAIL: $*" >&2; exit 1; }
assert_contains() { [[ "$1" == *"$2"* ]] || fail "expected '$2' in: $1"; }

# The installer exactly as the release workflow publishes it.
bash "${REPO_ROOT}/tools/build_release_installer.sh" "$VERSION" "${test_dir}/release" >/dev/null
export FAKE_INSTALLER="${test_dir}/release/install_nss.sh"

# A source archive shaped like GitHub's, holding this checkout's real
# docs/tutorials, so setup.sh's extraction runs against the actual layout.
[[ -d "${REPO_ROOT}/docs/tutorials" ]] ||
    fail "docs/tutorials is missing; setup.sh extracts the tutorials from that path"
mkdir -p "${test_dir}/archive/Safe-Synthesizer-${VERSION}/docs"
cp -R "${REPO_ROOT}/docs/tutorials" "${test_dir}/archive/Safe-Synthesizer-${VERSION}/docs/"
tar -czf "${test_dir}/source.tar.gz" -C "${test_dir}/archive" "Safe-Synthesizer-${VERSION}"
export FAKE_TARBALL="${test_dir}/source.tar.gz"

# curl: serves the installer and archive, and fails on any other URL.
cat > "${fake_bin}/curl" <<'EOF'
#!/usr/bin/env bash
out=""; url=""
while (( $# )); do
    case "$1" in -o) out="$2"; shift ;; -*) ;; *) url="$1" ;; esac
    shift
done
printf '%s\n' "$url" >> "${FAKE_CURL_LOG:?}"
case "$url" in
    */releases/latest/download/install_nss.sh) cp "$FAKE_INSTALLER" "$out" ;;
    */archive/refs/tags/v*.tar.gz) cp "$FAKE_TARBALL" "$out" ;;
    *) echo "fake curl: unexpected URL $url" >&2; exit 22 ;;
esac
EOF

# uv: logs every call. `venv` makes a venv whose python is the host
# interpreter; installing the package drops stand-ins for what setup.sh later
# checks (package version, torch, the CLI). FAKE_UV_FAIL makes that install fail.
cat > "${fake_bin}/uv" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "${FAKE_UV_LOG:?}"
case "$1" in
    --version) echo "uv 0.9.30" ;;
    venv)
        mkdir -p "${!#}/bin" "${!#}/site"
        printf '#!/usr/bin/env bash\nPYTHONPATH=%q exec %q "$@"\n' "${!#}/site" "$REAL_PYTHON" > "${!#}/bin/python"
        chmod +x "${!#}/bin/python"
        ;;
    pip)
        venv="${VIRTUAL_ENV:-}"; prev=""
        for arg in "$@"; do
            [[ "$prev" == "--python" ]] && venv="${arg%/bin/python}"
            [[ "$arg" == "--overrides" ]] && cat > "${FAKE_UV_OVERRIDES:?}"
            prev="$arg"
        done
        site="${venv:?}/site"
        if [[ "$*" == *nemo-safe-synthesizer* ]]; then
            [[ -z "${FAKE_UV_FAIL:-}" ]] || exit 1
            mkdir -p "$site/nemo_safe_synthesizer-${FAKE_VERSION}.dist-info"
            printf 'Name: nemo-safe-synthesizer\nVersion: %s\n' "$FAKE_VERSION" \
                > "$site/nemo_safe_synthesizer-${FAKE_VERSION}.dist-info/METADATA"
            printf 'class cuda:\n    is_available = staticmethod(lambda: True)\n' > "$site/torch.py"
            printf '#!/usr/bin/env bash\necho %s\n' "$FAKE_VERSION" > "$venv/bin/safe-synthesizer"
            chmod +x "$venv/bin/safe-synthesizer"
        else
            touch "$site/ipykernel.py" "$site/ipywidgets.py"
        fi
        ;;
    *) echo "fake uv: unexpected call: $*" >&2; exit 1 ;;
esac
EOF

# sudo, jupyter: keep setup.sh from running apt-get or writing a kernel into
# the developer's real Jupyter install.
printf '#!/usr/bin/env bash\n' > "${fake_bin}/sudo"
printf '#!/usr/bin/env bash\n' > "${fake_bin}/jupyter"
printf '#!/usr/bin/env bash\necho "$HOME/jupyter-kernels"\n' > "${fake_bin}/python"
chmod +x "$fake_bin"/*

# Runs setup.sh with a fresh $HOME. Extra VAR=value arguments are passed through.
run_setup() {
    local home="$1"
    shift
    mkdir -p "$home"
    env -u VIRTUAL_ENV -u UV_PROJECT_ENVIRONMENT -u CUDA \
        HOME="$home" PATH="${fake_bin}:${PATH}" FAKE_VERSION="$VERSION" \
        FAKE_UV_LOG="${home}.uv" FAKE_UV_OVERRIDES="${home}.overrides" FAKE_CURL_LOG="${home}.curl" \
        "$@" bash "$SETUP" 2>&1
}

# 1. Successful install.
home="${test_dir}/ok"
output="$(run_setup "$home")" || fail "setup.sh failed: $output"
venv="${home}/.nss-venv"
uv_calls="$(<"${home}.uv")"

# setup.sh downloads the installer from the URL the release publishes it at.
assert_contains "$(<"${home}.curl")" "/releases/latest/download/install_nss.sh"

# The installer reused setup.sh's Python 3.13 venv instead of creating its own.
assert_contains "$uv_calls" "venv --python 3.13 ${venv}"
[[ "$(grep -c '^venv ' "${home}.uv")" == 1 ]] || fail "installer created another venv: $uv_calls"

# The installer ran the release policy against that venv: pinned version,
# cu129 extra and index, and release constraints. Overrides are checked below.
assert_contains "$uv_calls" "pip install nemo-safe-synthesizer[engine,cu129]==${VERSION}"
assert_contains "$uv_calls" "--python ${venv}/bin/python"
assert_contains "$uv_calls" "--index https://download.pytorch.org/whl/cu129"
assert_contains "$uv_calls" "-c https://raw.githubusercontent.com/NVIDIA-NeMo/Safe-Synthesizer/v${VERSION}/constraints.txt"

# Notebook setup cells run their own `uv pip install`; without the installer's
# overrides uv re-resolves and downgrades the stack. setup.sh saves the exact
# overrides the installer used and points the kernel and terminals at them.
overrides="${home}/.nss-overrides.txt"
[[ "$(<"$overrides")" == "$(cat "${home}.overrides" 2>/dev/null)" ]] ||
    fail "saved overrides differ from the installer's: $(<"$overrides")"
assert_contains "$(<"${home}/jupyter-kernels/python3/kernel.json")" "\"UV_OVERRIDE\": \"${overrides}\""
assert_contains "$(<"${home}/.nss-env.sh")" "export UV_OVERRIDE=\"${overrides}\""

# The tutorials were extracted from the release archive into ~/tutorials.
compgen -G "${home}/tutorials/*.ipynb" >/dev/null || fail "no notebooks extracted into ~/tutorials"

# The rest of setup.sh found the install where it expects it and finished.
assert_contains "$output" "setup complete"

# 2. Failed install: setup.sh stops and says so, instead of reporting ready.
home="${test_dir}/install-fails"
if output="$(run_setup "$home" FAKE_UV_FAIL=1)"; then fail "setup.sh succeeded despite a failed install"; fi
[[ "$output" != *"setup complete"* ]] || fail "reported complete after a failed install"
assert_contains "$(<"${home}/SETUP-IN-PROGRESS.md")" "# Setup failed"

echo "Brev setup tests passed"
