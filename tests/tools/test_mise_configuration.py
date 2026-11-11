# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALLER = REPO_ROOT / "tools" / "install-mise.sh"
MISE_VERSION = "v2026.7.5"
MISE_GPG_KEY = "24853EC9F655CE80B48E6C3A8B81C9D17413A06D"  # pragma: allowlist secret


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _write_curl_stub(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "curl",
        """#!/usr/bin/env bash
set -euo pipefail
output=
url=
while (($#)); do
    case "$1" in
        -o) output="$2"; shift 2 ;;
        -H|--connect-timeout|--max-time|--retry|--retry-delay) shift 2 ;;
        -*) shift ;;
        *) url="$1"; shift ;;
    esac
done
printf '%s\n' "$url" >> "$CURL_LOG"
if [[ "$url" == "https://mise.run" && "$UNSIGNED_EXIT" != 0 ]]; then
    printf 'partial installer' > "$output"
    exit "$UNSIGNED_EXIT"
fi
printf '# installer from %s\n' "$url" > "$output"
""",
    )


def _write_gpg_stubs(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "gpg",
        """#!/usr/bin/env bash
set -euo pipefail
case " $* " in
    *" --import "*) exit "$GPG_IMPORT_EXIT" ;;
    *" --list-keys "*)
        ((GPG_IMPORT_EXIT == 0)) || exit "$GPG_IMPORT_EXIT"
        printf 'fpr:::::::::%s:\n' "$MISE_GPG_KEY"
        ;;
    *" --decrypt "*)
        output=
        while (($#)); do
            case "$1" in
                --output) output="$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        printf '# verified signed installer\n' > "$output"
        printf '[GNUPG:] VALIDSIG %s 2026-01-01 0 0 4 0 22 10 00 %s\n' "$MISE_GPG_KEY" "$MISE_GPG_KEY"
        ;;
esac
""",
    )
    _write_executable(bin_dir / "gpgconf", "#!/usr/bin/env bash\nexit 0\n")


def _write_sh_stub(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "sh",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$1" > "$SH_PATH_LOG"
cp "$1" "$SH_CONTENT_LOG"
mkdir -p "$HOME/.local/bin"
cat > "$HOME/.local/bin/mise" <<'EOF'
#!/bin/bash
printf '2026.7.5 linux-x64 (test)\n'
EOF
chmod 755 "$HOME/.local/bin/mise"
""",
    )


def _installer_environment(tmp_path: Path, *, gpg_import_exit: int = 0, unsigned_exit: int = 0) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    _write_curl_stub(bin_dir)
    _write_gpg_stubs(bin_dir)
    _write_sh_stub(bin_dir)

    return os.environ | {
        "CURL_LOG": str(tmp_path / "curl.log"),
        "GPG_IMPORT_EXIT": str(gpg_import_exit),
        "HOME": str(home_dir),
        "MISE_GPG_KEY": MISE_GPG_KEY,
        "MISE_VERSION": MISE_VERSION,
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "SH_CONTENT_LOG": str(tmp_path / "sh-content.log"),
        "SH_PATH_LOG": str(tmp_path / "sh-path.log"),
        "TMPDIR": str(temp_dir),
        "UNSIGNED_EXIT": str(unsigned_exit),
    }


def _run_installer(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", str(INSTALLER)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_mise_installer_fetches_signature_for_pinned_version(tmp_path: Path) -> None:
    environment = _installer_environment(tmp_path)

    result = _run_installer(environment)

    assert result.returncode == 0, result.stderr
    requested_urls = Path(environment["CURL_LOG"]).read_text(encoding="utf-8").splitlines()
    assert f"https://github.com/jdx/mise/releases/download/{MISE_VERSION}/install.sh.sig" in requested_urls
    assert "https://mise.jdx.dev/install.sh.sig" not in requested_urls


def test_mise_installer_executes_downloaded_unsigned_file_then_cleans_it_up(tmp_path: Path) -> None:
    environment = _installer_environment(tmp_path, gpg_import_exit=1)

    result = _run_installer(environment)

    assert result.returncode == 0, result.stderr
    executed_path = Path(environment["SH_PATH_LOG"]).read_text(encoding="utf-8").strip()
    assert executed_path.startswith(f"{environment['TMPDIR']}/mise-install.unsigned.")
    assert Path(environment["SH_CONTENT_LOG"]).read_text(encoding="utf-8") == "# installer from https://mise.run\n"
    assert not Path(executed_path).exists()


def test_mise_installer_does_not_execute_or_retain_failed_unsigned_download(tmp_path: Path) -> None:
    environment = _installer_environment(tmp_path, gpg_import_exit=1, unsigned_exit=22)

    result = _run_installer(environment)

    assert result.returncode != 0
    assert not Path(environment["SH_PATH_LOG"]).exists()
    assert list(Path(environment["TMPDIR"]).glob("mise-install.unsigned.*")) == []


def test_mise_installer_executes_stub_verified_signed_installer(tmp_path: Path) -> None:
    environment = _installer_environment(tmp_path) | {"MISE_REQUIRE_SIGNED_INSTALL": "1"}

    result = _run_installer(environment)

    assert result.returncode == 0, result.stderr
    assert Path(environment["SH_CONTENT_LOG"]).read_text(encoding="utf-8") == "# verified signed installer\n"


def _gpg_supported() -> bool:
    """Return True when gpg and gpgconf exist and gpg is 2.2 or newer."""
    if shutil.which("gpg") is None or shutil.which("gpgconf") is None:
        return False
    first_line = subprocess.run(["gpg", "--version"], capture_output=True, text=True, check=False).stdout.splitlines()
    match = re.search(r"(\d+)\.(\d+)", first_line[0]) if first_line else None
    return match is not None and (int(match[1]), int(match[2])) >= (2, 2)


requires_gpg = pytest.mark.skipif(not _gpg_supported(), reason="gpg >= 2.2 with gpgconf is required")


@pytest.fixture
def short_dir() -> Iterator[Path]:
    """Short scratch directory: gpg-agent socket paths must fit the 104-byte unix limit on macOS."""
    path = Path(tempfile.mkdtemp(prefix="mi-", dir="/tmp"))
    try:
        yield path
    finally:
        for home in path.glob("gnupg*"):
            subprocess.run(["gpgconf", "--homedir", str(home), "--kill", "all"], capture_output=True, check=False)
        shutil.rmtree(path, ignore_errors=True)


def _run_installer_with_signed_script(tmp_path: Path, signer: str) -> tuple[subprocess.CompletedProcess[str], Path]:
    """Run install-mise.sh with real gpg against a keyring served by a stub curl.

    The served key bundle always holds the pinned key and a second key. The
    installer payload is signed by ``signer`` (``pinned`` or ``other``) and
    touches a marker file when executed.
    """
    gnupg = tmp_path / "gnupg"
    gnupg.mkdir(mode=0o700)
    env = {"GNUPGHOME": str(gnupg), "PATH": os.environ["PATH"]}

    def gpg(*args: str, stdin: bytes | None = None) -> bytes:
        result = subprocess.run(
            ["gpg", "--batch", "--no-tty", "--pinentry-mode", "loopback", "--passphrase", "", *args],
            input=stdin,
            capture_output=True,
            env=env,
            check=True,
        )
        return result.stdout

    fingerprints: dict[str, str] = {}
    for name in ("pinned", "other"):
        try:
            gpg("--quick-generate-key", f"{name} <{name}@example.invalid>", "ed25519", "sign", "1d")
        except subprocess.CalledProcessError as error:
            pytest.skip(f"gpg cannot generate keys here: {error.stderr.decode(errors='replace')[:200]}")
        listing = gpg("--with-colons", "--list-keys", f"{name}@example.invalid").decode()
        fingerprints[name] = next(line.split(":")[9] for line in listing.splitlines() if line.startswith("fpr:"))

    bundle = tmp_path / "bundle.asc"
    bundle.write_bytes(gpg("--armor", "--export", fingerprints["pinned"], fingerprints["other"]))

    marker = tmp_path / "payload-ran"
    payload = f"touch '{marker}'\n".encode()
    signature = tmp_path / "install.sh.sig"
    signature.write_bytes(gpg("--local-user", fingerprints[signer], "--sign", "--output", "-", stdin=payload))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "curl",
        "#!/bin/sh\n"
        'while [ "$#" -gt 0 ]; do\n'
        '  case "$1" in -o) out="$2"; shift 2;; *) url="$1"; shift;; esac\n'
        "done\n"
        'case "$url" in\n'
        f'  */by-fingerprint/*) cp "{bundle}" "$out";;\n'
        f'  */install.sh.sig) cp "{signature}" "$out";;\n'
        "  *) exit 22;;\n"
        "esac\n",
    )
    for tool in (
        "bash",
        "sh",
        "gpg",
        "gpgconf",
        "awk",
        "grep",
        "sed",
        "head",
        "mktemp",
        "chmod",
        "rm",
        "touch",
        "cp",
        "dirname",
    ):
        located = shutil.which(tool)
        assert located is not None, tool
        (bin_dir / tool).symlink_to(located)

    result = subprocess.run(
        [str(bin_dir / "bash"), str(INSTALLER)],
        capture_output=True,
        text=True,
        env={
            "PATH": str(bin_dir),
            "HOME": str(tmp_path / "home"),
            "TMPDIR": str(tmp_path),
            "MISE_GPG_KEY": fingerprints["pinned"],
            "MISE_REQUIRE_SIGNED_INSTALL": "1",
        },
        check=False,
    )
    return result, marker


@requires_gpg
def test_installer_runs_payload_signed_by_pinned_key(short_dir: Path) -> None:
    result, marker = _run_installer_with_signed_script(short_dir, signer="pinned")

    assert marker.exists(), result.stderr


@requires_gpg
def test_installer_rejects_payload_signed_by_extra_key_in_bundle(short_dir: Path) -> None:
    result, marker = _run_installer_with_signed_script(short_dir, signer="other")

    assert not marker.exists()
    assert result.returncode != 0
    assert "MISE_REQUIRE_SIGNED_INSTALL=1 forbids unsigned fallback" in result.stderr
