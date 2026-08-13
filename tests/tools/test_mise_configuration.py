# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

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
    *" --decrypt "*) printf '# verified signed installer\n' ;;
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
