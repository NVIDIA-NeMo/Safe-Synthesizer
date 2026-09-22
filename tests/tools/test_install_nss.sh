#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
readonly REPO_ROOT INSTALLER="${REPO_ROOT}/install_nss.sh"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/nss-installer.XXXXXX")"
readonly test_dir
trap 'rm -rf "$test_dir"' EXIT
fake_bin="${test_dir}/bin"
mkdir -p "$fake_bin"

cat > "${fake_bin}/uv" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
count=0; [[ -f "${FAKE_UV_CALLS:?}" ]] && count="$(<"$FAKE_UV_CALLS")"
count=$((count + 1)); printf '%s' "$count" > "$FAKE_UV_CALLS"
printf '%s\0' "$@" >> "${FAKE_UV_LOG:?}"; printf '\0' >> "$FAKE_UV_LOG"
for arg in "$@"; do [[ "$arg" == "--overrides" ]] && cat > "${FAKE_UV_STDIN:?}"; done
if [[ "${1:-}" == "venv" ]]; then mkdir -p "${!#}/bin"; printf '#!/usr/bin/env bash\n' > "${!#}/bin/python"; chmod +x "${!#}/bin/python"; fi
[[ "${FAKE_UV_FAIL_CALL:-0}" != "$count" ]]
EOF
cat > "${fake_bin}/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
printf 'probe\n' >> "${FAKE_SMI_LOG:?}"
printf '%s' "${FAKE_SMI_OUTPUT:-}"
exit "${FAKE_SMI_STATUS:-0}"
EOF
chmod +x "${fake_bin}/uv" "${fake_bin}/nvidia-smi"

new_log() { FAKE_UV_LOG="${test_dir}/uv-$1.nul"; FAKE_UV_CALLS="${test_dir}/uv-$1.calls"; FAKE_UV_STDIN="${test_dir}/uv-$1.stdin"; FAKE_SMI_LOG="${test_dir}/smi-$1.log"; export FAKE_UV_LOG FAKE_UV_CALLS FAKE_UV_STDIN FAKE_SMI_LOG; }
uv_call_count() { local count=0; [[ -f "$FAKE_UV_CALLS" ]] && count="$(<"$FAKE_UV_CALLS")"; printf '%s' "$count"; }
assert_eq() { [[ "$1" == "$2" ]] || { echo "expected '$2', got '$1'" >&2; exit 1; }; }
assert_file_absent() { [[ ! -e "$1" ]] || { echo "unexpected file: $1" >&2; exit 1; }; }
read_call() { local -n target="$1"; target=(); [[ -f "$FAKE_UV_LOG" ]] && mapfile -d '' -t target < "$FAKE_UV_LOG"; }
make_installer_fixture() {
    local output="$1"
    local source="$2"
    local override="${3:-}"
    awk -v override="$override" '
        /readonly -a PACKAGE_OVERRIDES=\(/ {
            print
            if (override != "") printf "    \047%s\047\n", override
            in_overrides=1
            next
        }
        in_overrides && /^\)/ { print; in_overrides=0; next }
        !in_overrides { print }
    ' "$source" > "$output"
    chmod +x "$output"
}

readonly OVERRIDE_REQUIREMENT="test-override==1.2.3"
with_overrides_installer="${test_dir}/install_nss-with-overrides.sh"
without_overrides_installer="${test_dir}/install_nss-without-overrides.sh"
make_installer_fixture "$with_overrides_installer" "$INSTALLER" "$OVERRIDE_REQUIREMENT"
make_installer_fixture "$without_overrides_installer" "$INSTALLER"

# Dry-run is rendering-only: no driver probe and no uv invocation.
new_log dry
output="$(PATH="${fake_bin}:$PATH" DRY_RUN=1 CUDA=130 UV_PROJECT_ENVIRONMENT="${test_dir}/dry env" "$with_overrides_installer")"
assert_file_absent "$FAKE_SMI_LOG"; assert_file_absent "$FAKE_UV_LOG"
[[ "$output" != *"Installing with:"* && "$output" == *"--index https://pypi.nvidia.com"* ]]
[[ "$output" == *$'\n'"echo test-override==1.2.3 | uv pip install "*" --overrides - "* ]]

# CUDA 13 driver boundaries: reject below minimum before uv; warnings leave install available.
for case in equal above; do
    new_log "$case"; [[ "$case" == equal ]] && version=580.65.06 || version=581.0.0
    PATH="${fake_bin}:$PATH" FAKE_SMI_OUTPUT="$version" CUDA=130 UV_PROJECT_ENVIRONMENT="${test_dir}/$case" "$INSTALLER" >/dev/null
    assert_eq "$(uv_call_count)" 2
done
new_log below
if PATH="${fake_bin}:$PATH" FAKE_SMI_OUTPUT=580.65.05 CUDA=130 UV_PROJECT_ENVIRONMENT="${test_dir}/below" "$INSTALLER" >/dev/null 2>&1; then exit 1; fi
assert_eq "$(uv_call_count)" 0
for case in failed empty; do
    new_log "$case"; mkdir -p "${test_dir}/$case/bin"; printf '#!/usr/bin/env bash\n' > "${test_dir}/$case/bin/python"; chmod +x "${test_dir}/$case/bin/python"
    PATH="${fake_bin}:$PATH" FAKE_SMI_STATUS=$([[ "$case" == failed ]] && echo 1 || echo 0) FAKE_SMI_OUTPUT="" CUDA=130 UV_PROJECT_ENVIRONMENT="${test_dir}/$case" "$INSTALLER" >/dev/null 2>&1
    assert_eq "$(uv_call_count)" 1
done
new_log missing
mkdir -p "${test_dir}/missing/bin"; printf '#!/usr/bin/env bash\n' > "${test_dir}/missing/bin/python"; chmod +x "${test_dir}/missing/bin/python"
PATH="${fake_bin}:$PATH"; mv "${fake_bin}/nvidia-smi" "${test_dir}/nvidia-smi"
PATH="${fake_bin}:$PATH" CUDA=130 UV_PROJECT_ENVIRONMENT="${test_dir}/missing" "$INSTALLER" >/dev/null 2>&1
assert_eq "$(uv_call_count)" 1

# A failed venv prevents pip; failed pip follows one successful venv.
new_log venv-failure
if PATH="${fake_bin}:$PATH" FAKE_UV_FAIL_CALL=1 CUDA=cpu UV_PROJECT_ENVIRONMENT="${test_dir}/fail venv" "$INSTALLER" >/dev/null 2>&1; then exit 1; fi
assert_eq "$(uv_call_count)" 1
new_log pip-failure
if PATH="${fake_bin}:$PATH" FAKE_UV_FAIL_CALL=2 CUDA=cpu UV_PROJECT_ENVIRONMENT="${test_dir}/fail pip" "$INSTALLER" >/dev/null 2>&1; then exit 1; fi
assert_eq "$(uv_call_count)" 2

# Venv reuse has one argv-safe pip call; the path with spaces stays one argument.
new_log reuse
venv="${test_dir}/venv with spaces"; mkdir -p "$venv/bin"; printf '#!/usr/bin/env bash\n' > "$venv/bin/python"; chmod +x "$venv/bin/python"
PATH="${fake_bin}:$PATH" CUDA=cpu UV_PROJECT_ENVIRONMENT="$venv" PACKAGE_NAME=test-package CONSTRAINTS_URL=/constraints.txt "$with_overrides_installer" >/dev/null
assert_eq "$(uv_call_count)" 1
declare -a argv; read_call argv
expected=(pip install 'test-package[engine,cpu]' -c /constraints.txt --python "$venv/bin/python" --overrides - --index https://flashinfer.ai/whl/ --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
assert_eq "$(printf '%s\n' "${argv[@]}")" "$(printf '%s\n' "${expected[@]}")"
assert_eq "$(<"$FAKE_UV_STDIN")" "$OVERRIDE_REQUIREMENT"

# An installer without package overrides uses neither stdin nor --overrides.
new_log empty-overrides
output="$(PATH="${fake_bin}:$PATH" CUDA=cpu UV_PROJECT_ENVIRONMENT="$venv" PACKAGE_NAME=test-package CONSTRAINTS_URL=/constraints.txt "$without_overrides_installer")"
assert_eq "$(uv_call_count)" 1
read_call argv
empty_expected=(pip install 'test-package[engine,cpu]' -c /constraints.txt --python "$venv/bin/python" --index https://flashinfer.ai/whl/ --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
assert_eq "$(printf '%s\n' "${argv[@]}")" "$(printf '%s\n' "${empty_expected[@]}")"
assert_file_absent "$FAKE_UV_STDIN"
[[ "$output" != *" echo "* && "$output" != *" | "* && "$output" != *"--overrides"* ]]

# Private dependency groups participate in the same resolution as the package.
new_log dependency-groups
PATH="${fake_bin}:$PATH" CUDA=cpu PRIVATE_DEP_GROUPS="test docs" UV_PROJECT_ENVIRONMENT="$venv" PACKAGE_NAME=test-package CONSTRAINTS_URL=/constraints.txt "$with_overrides_installer" >/dev/null
assert_eq "$(uv_call_count)" 1
read_call argv
groups_expected=(pip install 'test-package[engine,cpu]' -c /constraints.txt --python "$venv/bin/python" --overrides - --group test --group docs --index https://flashinfer.ai/whl/ --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
assert_eq "$(printf '%s\n' "${argv[@]}")" "$(printf '%s\n' "${groups_expected[@]}")"

# Resolve-only executes uv's resolver with the full installer policy but does not install.
new_log resolve-only
resolve_venv="${test_dir}/resolve only"; mkdir -p "$resolve_venv/bin"; printf '#!/usr/bin/env bash\n' > "$resolve_venv/bin/python"; chmod +x "$resolve_venv/bin/python"
PATH="${fake_bin}:$PATH" CUDA=cpu NSS_INSTALLER_RESOLVE_ONLY=1 UV_PROJECT_ENVIRONMENT="$resolve_venv" PACKAGE_NAME=test-package CONSTRAINTS_URL=/constraints.txt "$with_overrides_installer" >/dev/null
assert_eq "$(uv_call_count)" 1
read_call argv
resolve_expected=(pip install 'test-package[engine,cpu]' -c /constraints.txt --python "$resolve_venv/bin/python" --overrides - --index https://flashinfer.ai/whl/ --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match --dry-run)
assert_eq "$(printf '%s\n' "${argv[@]}")" "$(printf '%s\n' "${resolve_expected[@]}")"

# A release installer applies its pinned package version and constraints artifact
# as actual uv argv, rather than merely rendering those values into its source.
new_log release
release_dir="${test_dir}/release"
bash "${REPO_ROOT}/tools/build_release_installer.sh" 1.2.3 "$release_dir"
release_venv="${test_dir}/release venv"; mkdir -p "$release_venv/bin"
printf '#!/usr/bin/env bash\n' > "$release_venv/bin/python"; chmod +x "$release_venv/bin/python"
release_installer="${test_dir}/install_nss-release-with-overrides.sh"
make_installer_fixture "$release_installer" "$release_dir/install_nss.sh" "$OVERRIDE_REQUIREMENT"
PATH="${fake_bin}:$PATH" CUDA=cpu UV_PROJECT_ENVIRONMENT="$release_venv" PACKAGE_NAME=test-package "$release_installer" >/dev/null
assert_eq "$(uv_call_count)" 1
read_call argv
release_constraints="https://raw.githubusercontent.com/NVIDIA-NeMo/Safe-Synthesizer/v1.2.3/constraints.txt"
release_expected=(pip install 'test-package[engine,cpu]==1.2.3' -c "$release_constraints" --python "$release_venv/bin/python" --overrides - --index https://flashinfer.ai/whl/ --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match)
assert_eq "$(printf '%s\n' "${argv[@]}")" "$(printf '%s\n' "${release_expected[@]}")"

# Docker consumes this narrow internal policy boundary; it has no effects.
new_log resolver
indexes="$(PATH="${fake_bin}:$PATH" NSS_INSTALLER_RESOLVE_INDEXES=1 CUDA=130 "$INSTALLER")"
assert_file_absent "$FAKE_UV_LOG"; assert_file_absent "$FAKE_SMI_LOG"
[[ "$indexes" == *"https://pypi.nvidia.com"* ]]
