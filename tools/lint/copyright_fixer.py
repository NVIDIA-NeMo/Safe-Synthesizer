# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env -S uv run --script
#
# /// script
# dependencies = [
#   "typer",
# ]
# ///


"""
Copyright header fixer for Safe Synthesizer.

Scans Python files and adds SPDX copyright headers where missing.
"""

import os
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(name="copyright-fixer", help="Copyright fixer tool for Safe Synthesizer.", no_args_is_help=True)

# --- constants ---

_CURRENT_YEAR = datetime.now().year
_HEADER = (
    f"# SPDX-FileCopyrightText: Copyright (c) 2025-{_CURRENT_YEAR} NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
    "\n"
)

_SKIP_DIRS = frozenset(
    {
        "__pycache__",
        ".git",
        ".pytest_cache",
        ".venv",
        ".mypy_cache",
        ".ruff_cache",
        ".egg-info",
        "node_modules",
        "venv",
        "site",
        ".uv_cache",
    }
)

_EXTENSIONS = frozenset({".py"})

# Cheap substring checks — no regex needed
_HEADER_MARKERS = (
    "SPDX-FileCopyrightText",
    "SPDX-License-Identifier",
    "Copyright (c)",
    "Copyright (C)",
)


# --- core helpers ---


def _has_header(head: str) -> bool:
    """Check the first ~512 bytes for any copyright marker."""
    for marker in _HEADER_MARKERS:
        if marker in head:
            return True
    return False


def _read_head(path: str, nbytes: int = 512) -> str:
    """Read the first *nbytes* of a file (fast, no full-file read)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read(nbytes)
    except OSError:
        return ""


def _collect_files_from_dir(root: str) -> list[str]:
    """Walk *root* with os.walk, pruning skip dirs, collecting .py files."""
    result: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune in-place so os.walk doesn't descend
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for fname in filenames:
            if os.path.splitext(fname)[1] in _EXTENSIONS:
                result.append(os.path.join(dirpath, fname))
    return result


def _add_header(filepath: str) -> bool:
    """Add the copyright header to *filepath*. Returns True if modified."""
    try:
        content = open(filepath, "r", encoding="utf-8").read()  # noqa: SIM115
    except (OSError, UnicodeDecodeError):
        return False

    if not content.strip():
        return False

    if _has_header(content[:512]):
        return False

    # Handle shebang
    if content.startswith("#!"):
        newline_pos = content.index("\n") + 1
        new_content = content[:newline_pos] + _HEADER + content[newline_pos:]
    else:
        new_content = _HEADER + content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)

    return True


def _resolve_targets(paths: list[Path]) -> tuple[list[str], str | None]:
    """Return (file_list, display_root)."""
    if len(paths) == 1 and paths[0].is_dir():
        root = str(paths[0].resolve())
        return _collect_files_from_dir(root), root

    files = [str(p.resolve()) for p in paths if p.is_file() and os.path.splitext(str(p))[1] in _EXTENSIONS]
    return files, None


# --- CLI ---


@app.command()
def update_license_headers(
    paths: list[Path] = typer.Argument(
        default=None,
        help="Directory to scan or individual files to process. Defaults to current directory.",
    ),
    check: bool = typer.Option(False, "--check", help="Check only, don't modify files. Exit 1 if headers are missing."),
) -> None:
    """Add SPDX copyright headers to Python files missing them.

    Accepts a single directory (scans recursively) or a list of individual
    files.  When no argument is provided, scans the current directory.
    """
    if not paths:
        paths = [Path(".")]

    for p in paths:
        if not p.exists():
            typer.echo(f"Error: {p} does not exist", err=True)
            raise typer.Exit(code=1)

    files, root = _resolve_targets(paths)

    def _rel(filepath: str) -> str:
        if root:
            return os.path.relpath(filepath, root)
        return filepath

    if check:
        missing = [f for f in files if _read_head(f) and not _has_header(_read_head(f))]
        if missing:
            typer.echo(f"Found {len(missing)} file(s) missing copyright headers:")
            for f in missing:
                typer.echo(f"  - {_rel(f)}")
            raise typer.Exit(code=1)
        typer.echo(f"All {len(files)} file(s) have copyright headers.")
    else:
        updated = 0
        for filepath in files:
            if _add_header(filepath):
                updated += 1
                typer.echo(f"  + {_rel(filepath)}")
        typer.echo(f"  Processed {len(files)} files, updated {updated}")
        if updated:
            typer.echo(f"Run 'git diff' to review {updated} changed file(s).")


if __name__ == "__main__":
    app()
