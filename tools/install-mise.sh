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
# Optional env:
#   MISE_REQUIRE_SIGNED_INSTALL=1  fail instead of falling back to the
#                                  unsigned mise.run installer when the
#                                  signed path can't be completed (missing
#                                  toolchain or keyserver/CDN flake).
#                                  Recommended for CI/release pipelines.
#
# Behaviour:
#   - If mise is already on PATH at the pinned version, exit early.
#   - If `mise --version` returns nothing, abort with a pointer to
#     MISE_VERBOSE=1 and the binary path (broken/partial install).
#   - If mise is already on PATH at a different version, abort with an
#     actionable message (we don't silently clobber the user's install).
#   - If the full gpg toolchain (gpg + gpg-agent + dirmngr) is available,
#     fetch install.sh.sig, verify its GPG signature against a temporary
#     GNUPGHOME (so we don't mutate the user's keyring), and run the
#     embedded install script. Keyserver recv and curl fetch are bounded
#     by timeouts and retried a few times on failure.
#   - If any of the above fails and MISE_REQUIRE_SIGNED_INSTALL != 1, fall
#     back to https://mise.run (no signature verification; warn loudly).
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

# Network knobs, applied to every keyserver/HTTP call so a flaky keyserver
# or CDN doesn't wedge `make setup` indefinitely in CI/container contexts.
readonly CURL_CONNECT_TIMEOUT=10
readonly CURL_MAX_TIME=60
readonly CURL_RETRIES=3
readonly CURL_RETRY_DELAY=2
readonly GPG_RECV_TIMEOUT=30
readonly GPG_RECV_RETRIES=3

# Set MISE_REQUIRE_SIGNED_INSTALL=1 to fail hard when the signed path can't
# be completed (missing toolchain or network failure fetching the key /
# installer) instead of falling back to the unsigned `curl | sh` path.
# Recommended for CI/release pipelines; default is off so local dev on slim
# images still succeeds with a loud warning.
REQUIRE_SIGNED_INSTALL="${MISE_REQUIRE_SIGNED_INSTALL:-0}"

curl_fetch() {
    curl -fsSL \
        --connect-timeout "$CURL_CONNECT_TIMEOUT" \
        --max-time "$CURL_MAX_TIME" \
        --retry "$CURL_RETRIES" \
        --retry-delay "$CURL_RETRY_DELAY" \
        --retry-connrefused \
        "$@"
}

# `timeout(1)` is GNU coreutils; not in the default macOS/BSD userland.
# When absent, rely on gpg's own `keyserver-options timeout=N` so we still
# get a bounded wait.
if command -v timeout >/dev/null 2>&1; then
    gpg_timeout() { timeout "$GPG_RECV_TIMEOUT" "$@"; }
else
    gpg_timeout() { "$@"; }
fi

gpg_recv_key() {
    local attempt
    for attempt in $(seq 1 "$GPG_RECV_RETRIES"); do
        if gpg_timeout gpg --batch --no-tty \
                --keyserver "$KEYSERVER" \
                --keyserver-options "timeout=${GPG_RECV_TIMEOUT}" \
                --recv-keys "$MISE_GPG_KEY"; then
            return 0
        fi
        echo "WARNING: gpg --recv-keys attempt ${attempt}/${GPG_RECV_RETRIES} failed" >&2
        sleep "$CURL_RETRY_DELAY"
    done
    return 1
}

unsigned_install_or_fail() {
    local reason="$1"
    if [[ "$REQUIRE_SIGNED_INSTALL" == "1" ]]; then
        echo "ERROR: ${reason}; MISE_REQUIRE_SIGNED_INSTALL=1 forbids unsigned fallback" >&2
        exit 1
    fi
    echo "WARNING: ${reason} -- installing mise ${MISE_VERSION} via ${MISE_RUN_URL} without signature verification" >&2
    curl_fetch "$MISE_RUN_URL" | MISE_VERSION="$MISE_VERSION" sh
}

expected="${MISE_VERSION#v}"

# mise --version prints "<semver> <target> (<date>)" to stdout plus "available
# update" nags to stderr, so take only the first field on stdout.
current_mise_version() {
    mise --version 2>/dev/null | awk '{print $1; exit}'
}

if command -v mise >/dev/null 2>&1; then
    installed="$(current_mise_version)"
    if [[ -z "$installed" ]]; then
        mise_path="$(command -v mise)"
        echo "ERROR: \`mise --version\` (at ${mise_path}) produced no output" >&2
        echo "       the install looks broken/partial; rerun with MISE_VERBOSE=1 for details" >&2
        echo "       or remove ${mise_path} and rerun 'make setup'" >&2
        exit 1
    fi
    if [[ "$installed" == "$expected" ]]; then
        echo "mise ${installed} already installed"
        exit 0
    fi
    echo "ERROR: found mise ${installed} on PATH but this repo pins ${MISE_VERSION}" >&2
    echo "       run 'mise self-update --version ${expected}' or uninstall the current mise and rerun 'make setup'" >&2
    exit 1
fi

echo "mise not found -- installing ${MISE_VERSION}..."

# gpg's keyserver + decrypt flow needs all three of gpg, gpg-agent, and
# dirmngr. Slim container images (e.g. debian:*-slim with
# --no-install-recommends) commonly ship only a subset, so require the full
# set before taking the signed path instead of tripping over a partial
# toolchain mid-run.
have_gpg_toolchain=false
if command -v gpg >/dev/null 2>&1 \
    && command -v gpg-agent >/dev/null 2>&1 \
    && command -v dirmngr >/dev/null 2>&1; then
    have_gpg_toolchain=true
fi

if [[ "$have_gpg_toolchain" == true ]]; then
    echo "Verifying installer signature..."

    # Isolate verification in an ephemeral GNUPGHOME so we don't mutate the
    # user's long-lived keyring (importing the mise release key, trust db
    # updates, spawning a persistent gpg-agent, etc.). The trap cleans up
    # the tmp script and the keyring directory (including the agent socket)
    # on any exit path.
    # `mktemp` isn't in POSIX and the no-arg form is a GNU/modern-BSD
    # extension. Passing an explicit template as a positional argument is the
    # form every implementation agrees on, and the named prefix keeps any
    # leaked temp paths self-identifying.
    tmp_prefix="${TMPDIR:-/tmp}/mise-install"
    gnupg_home="$(mktemp -d "${tmp_prefix}.gnupg.XXXXXXXX")"
    tmpscript="$(mktemp "${tmp_prefix}.sh.XXXXXXXX")"
    trap 'gpgconf --homedir "$gnupg_home" --kill all >/dev/null 2>&1 || true; rm -rf "$gnupg_home" "$tmpscript"' EXIT
    chmod 700 "$gnupg_home"
    export GNUPGHOME="$gnupg_home"

    if ! gpg_recv_key; then
        unsigned_install_or_fail "gpg --recv-keys from ${KEYSERVER} failed after ${GPG_RECV_RETRIES} attempts"
    elif ! curl_fetch "$MISE_SIG_URL" | gpg --batch --no-tty --decrypt >"$tmpscript"; then
        unsigned_install_or_fail "failed to fetch/verify ${MISE_SIG_URL}"
    else
        MISE_VERSION="$MISE_VERSION" sh "$tmpscript"
    fi
else
    unsigned_install_or_fail "full gpg toolchain (gpg + gpg-agent + dirmngr) not available"
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
