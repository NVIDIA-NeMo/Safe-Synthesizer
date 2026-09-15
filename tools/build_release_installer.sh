#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

readonly REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VERSION="${1:?Usage: $0 <version> <output-dir>}"
readonly OUTPUT_DIR="${2:?Usage: $0 <version> <output-dir>}"
readonly CONSTRAINTS_URL="https://raw.githubusercontent.com/NVIDIA-NeMo/Safe-Synthesizer/v${VERSION}/constraints.txt"
readonly RELEASE_INSTALLER="${OUTPUT_DIR}/install_nss.sh"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ ! "$VERSION" =~ ^[0-9]+([.][0-9]+)*[a-zA-Z0-9.+-]*$ ]]; then
    die "invalid release version: ${VERSION}"
fi

case "$OUTPUT_DIR" in
    / | . | ..) die "refusing to write release installer to '${OUTPUT_DIR}'" ;;
esac

mkdir -p "$OUTPUT_DIR"
sed \
    -e "s|^readonly RELEASE_VERSION=\"\"$|readonly RELEASE_VERSION=\"${VERSION}\"|" \
    -e "s|https://raw.githubusercontent.com/NVIDIA-NeMo/Safe-Synthesizer/main/constraints.txt|${CONSTRAINTS_URL}|" \
    "${REPO_ROOT}/install_nss.sh" > "$RELEASE_INSTALLER"

grep -Fqx "readonly RELEASE_VERSION=\"${VERSION}\"" "$RELEASE_INSTALLER" ||
    die "failed to pin the installer package version"
grep -Fq "$CONSTRAINTS_URL" "$RELEASE_INSTALLER" ||
    die "failed to pin the installer constraints URL"
chmod 0755 "$RELEASE_INSTALLER"
