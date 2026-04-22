#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# install-mise.sh -- install the pinned mise version, preferring the
# GPG-verified path when the local toolchain supports it.
#
# Inputs (required, passed by the Makefile):
#   MISE_VERSION  mise version to install, e.g. "v2026.4.11"
#   MISE_GPG_KEY  fingerprint of the mise release signing key
#
# Behaviour:
#   - If mise is already on PATH at the pinned version, exit early.
#   - If mise is already on PATH at a different version, abort with an
#     actionable message (we don't silently clobber the user's install).
#   - If both gpg and dirmngr are available, fetch install.sh.sig, verify
#     its GPG signature, and run the embedded install script.
#   - Otherwise fall back to https://mise.run (no signature verification;
#     warn clearly).
#   - In either install path, pass MISE_VERSION through to the installer
#     via the documented MISE_VERSION env var so the installed binary
#     matches the value pinned in the Makefile.
#   - Assert the installed binary reports the pinned version before exit.
#
# See: https://mise.jdx.dev/installing-mise.html
#

set -euo pipefail

# ${VAR:?msg} aborts the script with the given message when VAR is unset or
# empty. Paired with `:` (the no-op builtin) it becomes an assert-and-die
# guard that surfaces a clearer error than `set -u`'s "unbound variable".
: "${MISE_VERSION:?MISE_VERSION is required (pass from Makefile)}"
: "${MISE_GPG_KEY:?MISE_GPG_KEY is required (pass from Makefile)}"

readonly MISE_SIG_URL="https://mise.jdx.dev/install.sh.sig"
readonly MISE_RUN_URL="https://mise.run"
readonly KEYSERVER="hkps://keys.openpgp.org"

expected="${MISE_VERSION#v}"

# mise --version prints "<semver> <target> (<date>)" to stdout plus "available
# update" nags to stderr, so take only the first field on stdout.
current_mise_version() {
    mise --version 2>/dev/null | awk '{print $1; exit}'
}

if command -v mise >/dev/null 2>&1; then
    installed="$(current_mise_version)"
    if [[ "$installed" == "$expected" ]]; then
        echo "mise ${installed} already installed"
        exit 0
    fi
    echo "ERROR: found mise ${installed} on PATH but this repo pins ${MISE_VERSION}" >&2
    echo "       run 'mise self-update --version ${expected}' or uninstall the current mise and rerun 'make setup'" >&2
    exit 1
fi

echo "mise not found -- installing ${MISE_VERSION}..."

if command -v gpg >/dev/null 2>&1 && command -v dirmngr >/dev/null 2>&1; then
    echo "Verifying installer signature..."
    gpg --batch --no-tty --keyserver "$KEYSERVER" --recv-keys "$MISE_GPG_KEY"

    tmpscript="$(mktemp)"
    trap 'rm -f "$tmpscript"' EXIT

    curl -fsSL "$MISE_SIG_URL" | gpg --batch --no-tty --decrypt >"$tmpscript"
    MISE_VERSION="$MISE_VERSION" sh "$tmpscript"
else
    echo "WARNING: gpg+dirmngr not available -- installing mise ${MISE_VERSION} via ${MISE_RUN_URL} without signature verification"
    curl -fsSL "$MISE_RUN_URL" | MISE_VERSION="$MISE_VERSION" sh
fi

if ! command -v mise >/dev/null 2>&1; then
    echo "ERROR: mise not found after install" >&2
    exit 1
fi

installed="$(current_mise_version)"

if [[ "$installed" != "$expected" ]]; then
    echo "ERROR: installed mise ${installed} does not match pinned ${MISE_VERSION}" >&2
    exit 1
fi

echo "mise ${installed} installed successfully"
