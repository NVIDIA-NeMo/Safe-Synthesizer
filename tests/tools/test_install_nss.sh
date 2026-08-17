#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT
readonly INSTALLER="${REPO_ROOT}/install_nss.sh"

assert_contains() {
    local actual="$1"
    local expected="$2"
    [[ "$actual" == *"$expected"* ]] || {
        echo "Expected output to contain: $expected" >&2
        echo "Actual output: $actual" >&2
        exit 1
    }
}

assert_not_contains() {
    local actual="$1"
    local unexpected="$2"
    [[ "$actual" != *"$unexpected"* ]] || {
        echo "Expected output not to contain: $unexpected" >&2
        echo "Actual output: $actual" >&2
        exit 1
    }
}

assert_dry_run() {
    local cuda="$1"
    local expected_extra="$2"
    local expected_index="${3:-}"
    local output
    output="$(DRY_RUN=1 CUDA="$cuda" "$INSTALLER")"

    assert_contains "$output" "nemo-safe-synthesizer\[engine\,${expected_extra}\]"
    if [[ -n "$expected_index" ]]; then
        assert_contains "$output" "--index"
        assert_contains "$output" "$expected_index"
        assert_contains "$output" "--index-strategy unsafe-best-match"
    else
        assert_not_contains "$output" "--index"
    fi
}

assert_dry_run 129 cu129 "https://wheels.vllm.ai/"
assert_dry_run 130 cu130 "https://pypi.nvidia.com"
if [[ "$(uname -s)" == "Linux" ]]; then
    assert_dry_run cpu cpu "https://download.pytorch.org/whl/cpu"
else
    assert_dry_run cpu cpu
fi

help_output="$(DRY_RUN=1 CUDA=help "$INSTALLER")"
assert_contains "$help_output" "Usage:"
assert_not_contains "$help_output" "Installing with:"

if DRY_RUN=1 CUDA=unsupported "$INSTALLER" >/dev/null 2>&1; then
    echo "Unsupported CUDA value unexpectedly succeeded" >&2
    exit 1
fi

test_dir="$(mktemp -d)"
readonly test_dir
trap 'rm -rf "$test_dir"' EXIT

fake_bin="${test_dir}/bin"
fake_uv_log="${test_dir}/uv.log"
mkdir -p "$fake_bin"
cat > "${fake_bin}/uv" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >> "${FAKE_UV_LOG:?}"
printf '\n' >> "$FAKE_UV_LOG"
if [[ "${1:-}" == "venv" ]]; then
    venv_path="${!#}"
    mkdir -p "$venv_path/bin"
    printf '#!/usr/bin/env bash\n' > "$venv_path/bin/python"
    chmod +x "$venv_path/bin/python"
fi
EOF
chmod +x "${fake_bin}/uv"

dry_venv="${test_dir}/dry-venv"
dry_output="$(
    PATH="${fake_bin}:$PATH" \
        FAKE_UV_LOG="$fake_uv_log" \
        UV_PROJECT_ENVIRONMENT="$dry_venv" \
        DRY_RUN=1 \
        CUDA=cpu \
        "$INSTALLER"
)"
[[ ! -e "$dry_venv" ]] || {
    echo "Dry run unexpectedly created: $dry_venv" >&2
    exit 1
}
[[ ! -e "$fake_uv_log" ]] || {
    echo "Dry run unexpectedly invoked uv" >&2
    exit 1
}
assert_contains "$dry_output" "uv venv --seed $dry_venv"
assert_contains "$dry_output" "--python $dry_venv/bin/python"

install_venv="${test_dir}/install-venv"
PATH="${fake_bin}:$PATH" \
    FAKE_UV_LOG="$fake_uv_log" \
    UV_PROJECT_ENVIRONMENT="$install_venv" \
    CUDA=cpu \
    PACKAGE_NAME=test-package \
    CONSTRAINTS_URL=/constraints.txt \
    "$INSTALLER" >/dev/null

uv_calls="$(<"$fake_uv_log")"
assert_contains "$uv_calls" "venv --seed $install_venv"
assert_contains "$uv_calls" "pip install"
assert_contains "$uv_calls" "--python $install_venv/bin/python"
