#!/usr/bin/env -S uv run
# /// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# /// SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.33.0",
#     "tomlkit>=0.13.0",
#     "packaging>=24",
#     "typing-extensions>=4.15.0",
# ]
# ///
# What is this?
# This script is used to patch CVE advisories from Dependabot into pyproject.toml.
#
# Direct dependencies (project.dependencies, optional-dependencies, dependency-groups)
# are bumped in-place by editing pyproject.toml directly.
#
# Transitive dependencies (not declared anywhere) are added to / merged into
# [tool.uv] constraint-dependencies so they don't widen the wheel's Requires-Dist.
#
# A single `uv lock` is run at the end to regenerate the lock file.
#
# Requirements:
# - A GitHub token with repo read access.
#   Obtain with: export GITHUB_TOKEN=$(gh auth token)
# Then run: uv run tools/patch_dependabot.py
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, TypeAlias

import requests
import tomlkit
import tomlkit.items
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version
from tomlkit.container import OutOfOrderTableProxy
from typing_extensions import TypeIs

TOO_HARD_TO_UPGRADE = frozenset({canonicalize_name(n) for n in ("torch", "transformers", "vllm")})
REPOSITORY_NAME = os.environ.get("REPOSITORY_NAME") or "NVIDIA-NeMo/Safe-Synthesizer"

TomlTable: TypeAlias = tomlkit.TOMLDocument | tomlkit.items.Table | OutOfOrderTableProxy


# ---------------------------------------------------------------------------
# PEP 508 / specifier helpers
# ---------------------------------------------------------------------------


def _format_requirement(name: str, spec: SpecifierSet, marker: Marker | None) -> str:
    s = str(spec).strip()
    out = f"{name}{s}" if s else str(name)
    if marker is not None:
        out = f"{out} ; {marker}"
    return out


def _merge_with_floor(pypi_name: str, floor: str, existing: str) -> str:
    """Return *existing* with a >=floor lower-bound merged in, collapsing redundant bounds."""
    try:
        r = Requirement(existing)
    except Exception:
        return f"{pypi_name}>={floor}"
    if canonicalize_name(r.name) != canonicalize_name(pypi_name):
        return existing
    f_v = Version(floor)
    kept: list[str] = []
    for s in r.specifier:
        # Drop exact/arbitrary-equality pins that are below the new floor (impossible).
        if s.operator in ("==", "===") and s.version and Version(s.version) < f_v:
            continue
        # Drop weaker lower bounds (>= or >) already superseded by the new floor.
        if s.operator in (">=", ">") and s.version and Version(s.version) <= f_v:
            continue
        kept.append(str(s))
    m = SpecifierSet(",".join([f">={floor}", *kept]))
    return _format_requirement(r.name, m, r.marker)


def _safe_req_name(line: str) -> str | None:
    line = line.strip()
    if not line:
        return None
    try:
        return canonicalize_name(Requirement(line).name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# In-place direct-dep bumping
# ---------------------------------------------------------------------------


def _bump_array_item(arr: tomlkit.items.Array, cname: str, floor: str, display_name: str) -> bool:
    """Find the item matching *cname* in a tomlkit Array and bump its floor. Returns True if changed."""
    for i, item in enumerate(arr):
        if not isinstance(item, str):
            continue
        if _safe_req_name(item) == cname:
            updated = _merge_with_floor(display_name, floor, item)
            if updated != item:
                arr[i] = updated
                return True
            return False  # already satisfied
    return False


def _bump_direct_deps_in_doc(
    doc: tomlkit.TOMLDocument,
    cname: str,
    floor: str,
    display_name: str,
) -> list[str]:
    """
    Edit every occurrence of *cname* in project.dependencies,
    project.optional-dependencies.*, and dependency-groups.* in-place.
    Returns list of human-readable locations that were changed.
    """
    changed: list[str] = []

    match _get_table(doc, "project"):
        case None:
            pass
        case proj:
            match _get_array(proj, "dependencies"):
                case tomlkit.items.Array() as main_deps:
                    if _bump_array_item(main_deps, cname, floor, display_name):
                        changed.append("project.dependencies")

            match _get_table(proj, "optional-dependencies"):
                case None:
                    pass
                case opt:
                    for extra_name, extra_arr in opt.items():
                        match extra_arr:
                            case tomlkit.items.Array() as extra_dependencies:
                                if _bump_array_item(extra_dependencies, cname, floor, display_name):
                                    changed.append(f"project.optional-dependencies.{extra_name}")

    match _get_table(doc, "dependency-groups"):
        case None:
            pass
        case dep_groups:
            for gname, garr in dep_groups.items():
                match garr:
                    case tomlkit.items.Array() as group_dependencies:
                        if _bump_array_item(group_dependencies, cname, floor, display_name):
                            changed.append(f"dependency-groups.{gname}")

    return changed


def _collect_direct_dep_names(doc: tomlkit.TOMLDocument) -> frozenset[str]:
    names: set[str] = set()
    match _get_table(doc, "project"):
        case None:
            pass
        case proj:
            for line in _get_array(proj, "dependencies") or []:
                n = _safe_req_name(str(line)) if isinstance(line, str) else None
                if n:
                    names.add(n)

            match _get_table(proj, "optional-dependencies"):
                case None:
                    pass
                case optional_dependencies:
                    for lines in optional_dependencies.values():
                        match lines:
                            case tomlkit.items.Array() as dependency_lines:
                                for line in dependency_lines:
                                    n = _safe_req_name(str(line)) if isinstance(line, str) else None
                                    if n:
                                        names.add(n)

    match _get_table(doc, "dependency-groups"):
        case None:
            pass
        case dep_groups:
            for items in dep_groups.values():
                match items:
                    case tomlkit.items.Array() as dependency_items:
                        for item in dependency_items:
                            n = _safe_req_name(str(item)) if isinstance(item, str) else None
                            if n:
                                names.add(n)
    return frozenset(names)


# ---------------------------------------------------------------------------
# constraint-dependencies helpers (for transitive deps)
# ---------------------------------------------------------------------------


def _index_constraints_by_name(lines: list[str]) -> dict[str, int]:
    by_name: dict[str, int] = {}
    for i, line in enumerate(lines):
        n = _safe_req_name(line)
        if n is not None:
            by_name[n] = i
    return by_name


def _apply_to_constraints(current: list[str], package_name: str, floor: str) -> list[str]:
    """Merge floor into existing constraint line, or append a new one."""
    out = list(current)
    by_name = _index_constraints_by_name(out)
    c = canonicalize_name(package_name)
    if c in by_name:
        out[by_name[c]] = _merge_with_floor(package_name, floor, out[by_name[c]])
    else:
        out.append(f"{package_name}>={floor}")
    return out


def _write_constraints_txt(path: Path, constraint_lines: list[str]) -> None:
    """Write a pip/uv -c compatible constraints file from constraint-dependencies."""
    header = (
        "# Security floor constraints -- generated from [tool.uv] constraint-dependencies in pyproject.toml\n"
        "# Pass to pip/uv with:  pip install <pkg> -c constraints.txt\n"
        "#                        uv pip install <pkg> -c constraints.txt\n"
    )
    path.write_text(header + "\n".join(sorted(constraint_lines)) + "\n", encoding="utf-8")


def _replace_constraint_dependencies_array(doc: tomlkit.TOMLDocument, lines: list[str]) -> None:
    match _get_table(doc, "tool"):
        case None:
            raise SystemExit("pyproject.toml: missing or invalid [tool] table")
        case tool:
            match _get_table(tool, "uv"):
                case None:
                    raise SystemExit("pyproject.toml: missing or invalid [tool.uv] table")
                case uv:
                    arr = tomlkit.array()
                    arr.multiline(True)
                    for line in sorted(lines):
                        arr.append(line)
                    uv["constraint-dependencies"] = arr


# ---------------------------------------------------------------------------
# Advisory collection
# ---------------------------------------------------------------------------


def _collect_max_floors(deps: list[dict[str, Any]], too_hard: frozenset[str]) -> dict[str, str]:
    """Map canonical name -> highest advisory floor version across all alerts."""
    out: dict[str, str] = {}
    for dep in deps:
        package_name = dep["dependency"]["package"]["name"]
        c = canonicalize_name(package_name)
        if c in too_hard:
            print(f"Skipping (pinned): {package_name}")
            continue
        ident = ((dep.get("security_vulnerability") or {}).get("first_patched_version") or {}).get("identifier")
        if not ident:
            print(f"Skipping (no fix): {package_name}")
            continue
        if c not in out or Version(ident) > Version(out[c]):
            out[c] = ident
    return out


def _advisory_spelling_by_canonical(deps: list[dict[str, Any]], cnames: frozenset[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for dep in deps:
        p = dep["dependency"]["package"]["name"]
        c = canonicalize_name(p)
        if c in cnames:
            out.setdefault(c, p)
    for c in cnames:
        out.setdefault(c, c)
    return out


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _is_table(x: object) -> TypeIs[TomlTable]:
    return isinstance(x, (tomlkit.TOMLDocument, tomlkit.items.Table, OutOfOrderTableProxy))


def _get_table(table: TomlTable, key: str) -> TomlTable | None:
    value = table.get(key)
    return value if _is_table(value) else None


def _get_array(table: TomlTable, key: str) -> tomlkit.items.Array | None:
    value = table.get(key)
    return value if isinstance(value, tomlkit.items.Array) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.is_file():
        raise SystemExit("Run from the repo root (pyproject.toml not found).")

    dependabot_file = "dependabot.json"
    if not os.path.exists(dependabot_file):
        url = f"https://api.github.com/repos/{REPOSITORY_NAME}/dependabot/alerts"
        github_token = os.getenv("GITHUB_TOKEN")
        if not github_token:
            raise SystemExit("GITHUB_TOKEN is not set, required to fetch dependabot alerts")
        headers = {"Authorization": f"Bearer {github_token}"}
        all_deps: list[dict[str, Any]] = []
        next_url: str | None = url
        params: dict[str, int] | None = {"per_page": 100}
        while next_url:
            response = requests.get(next_url, headers=headers, params=params)
            response.raise_for_status()
            all_deps.extend(response.json())
            nxt = response.links.get("next")
            next_url = nxt["url"] if nxt else None
            params = None
        with open(dependabot_file, "w", encoding="utf-8") as f:
            json.dump(all_deps, f)

    with open(dependabot_file, encoding="utf-8") as f:
        deps: list[dict[str, Any]] = json.load(f)

    doc = tomlkit.parse(pyproject_path.read_text(encoding="utf-8"))
    direct_dep_names = _collect_direct_dep_names(doc)

    floors = _collect_max_floors(deps, TOO_HARD_TO_UPGRADE)
    if not floors:
        print("No advisories to apply.")
        return
    display = _advisory_spelling_by_canonical(deps, frozenset(floors))

    changed = False

    # ------------------------------------------------------------------
    # Pass 1: bump direct deps in-place inside the tomlkit document
    # ------------------------------------------------------------------
    for cname, floor in sorted(floors.items()):
        if cname not in direct_dep_names:
            continue
        pypi_name = display.get(cname, cname)
        locations = _bump_direct_deps_in_doc(doc, cname, floor, pypi_name)
        if locations:
            print(f"bumped direct: {pypi_name}>={floor}  ({', '.join(locations)})")
            changed = True
        else:
            print(f"already satisfied: {pypi_name}>={floor}")

    # ------------------------------------------------------------------
    # Pass 2: transitive deps → constraint-dependencies
    # ------------------------------------------------------------------
    match _get_table(doc, "tool"):
        case None:
            raw_constraints = None
        case tool:
            match _get_table(tool, "uv"):
                case None:
                    raw_constraints = None
                case uv_section:
                    raw_constraints = uv_section.get("constraint-dependencies")

    match raw_constraints:
        case None:
            current: list[str] = []
        case tomlkit.items.Array() as constraints:
            current = [str(item) for item in constraints if str(item).strip()]
        case _:
            raise SystemExit("pyproject: [tool.uv] constraint-dependencies is not an array")

    updated = list(current)
    for cname, floor in sorted(floors.items()):
        if cname in direct_dep_names:
            continue
        pypi_name = display.get(cname, cname)
        updated = _apply_to_constraints(updated, pypi_name, floor)
        print(f"constraint (transitive): {pypi_name}>={floor}")

    if updated != current:
        _replace_constraint_dependencies_array(doc, updated)
        changed = True

    # ------------------------------------------------------------------
    # Write constraints.txt from the final constraint-dependencies list
    # (source of truth is pyproject.toml, not just the current advisory run)
    # ------------------------------------------------------------------
    constraints_path = pyproject_path.parent / "constraints.txt"
    _write_constraints_txt(constraints_path, updated)
    print(f"Wrote {constraints_path}")

    # ------------------------------------------------------------------
    # Single write + uv lock
    # ------------------------------------------------------------------
    if changed:
        pyproject_path.write_text(tomlkit.dumps(doc), encoding="utf-8")
        print("Running uv lock …")
        subprocess.check_call(["uv", "lock"])
    else:
        print("Nothing to update in pyproject.toml.")


if __name__ == "__main__":
    main()
