#!/usr/bin/env -S uv run
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""generate_sbom: build a dependency SBOM from uv.lock and diff new packages.

Parses ``uv.lock`` for direct and transitive dependencies, enriches each package
with license and project URL metadata from PyPI, and emits a machine-readable
JSON document or a markdown table.

License resolution order:

1. Manual overrides from ``tools/sbom_license_overrides.json`` (or ``--overrides``)
2. PyPI JSON API for the pinned version (``license_expression``, ``license``, classifiers)
3. Wheel ``METADATA`` and bundled license files from PyPI release wheels
4. Wheel ``METADATA`` from ``uv.lock`` wheel URLs (range-fetch for large wheels)
5. Latest PyPI project metadata when the pinned release omits license fields

Usage::

    uv run tools/generate_sbom.py generate -o sbom.json
    uv run tools/generate_sbom.py generate --markdown -o sbom.md
    uv run tools/generate_sbom.py diff sbom.json
    uv run tools/generate_sbom.py diff sbom.md --markdown

Models:
    SbomEntry    -- one dependency row (library, version, license, project_url)
    SbomDocument -- full SBOM snapshot (JSON canonical format)
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer>=0.15",
#     "pydantic>=2",
#     "packaging>=24",
#     "requests>=2.33.0",
# ]
# ///

from __future__ import annotations

import io
import json
import re
import struct
import sys
import tomllib
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import requests
import typer
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

app = typer.Typer(
    help="Generate an SBOM from uv.lock and diff new packages against a baseline.",
    no_args_is_help=True,
)

DEFAULT_LOCKFILE = "uv.lock"
DEFAULT_OVERRIDES = Path(__file__).resolve().parent / "sbom_license_overrides.json"
MAX_WHEEL_BYTES = 50 * 1024 * 1024
EOCD_SIGNATURE = b"PK\x05\x06"
CDFH_SIGNATURE = b"PK\x01\x02"
LFH_SIGNATURE = b"PK\x03\x04"
PYPI_PROJECT_URL = "https://pypi.org/project/{name}/"
MARKDOWN_ROW_RE = re.compile(
    r"^\|\s*(?P<library>[^|]+?)\s*\|\s*(?P<version>[^|]+?)\s*\|\s*(?P<license>[^|]+?)\s*\|\s*(?P<project_url>[^|]+?)\s*\|$",
)


class OutputFormat(str, Enum):
    """Supported SBOM output formats."""

    json = "json"
    markdown = "markdown"


class SbomEntry(BaseModel):
    """One dependency row in the SBOM."""

    library: str
    version: str
    license: str
    project_url: str


class SbomDocument(BaseModel):
    """Canonical SBOM snapshot."""

    generated_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    lockfile: str = DEFAULT_LOCKFILE
    packages: list[SbomEntry]


class LicenseOverride(BaseModel):
    """Manual license metadata for packages with incomplete upstream metadata."""

    license: str
    project_url: str | None = None


@dataclass(frozen=True)
class LockfilePackage:
    """One resolved package from ``uv.lock``."""

    version: str
    wheel_urls: tuple[str, ...]


def parse_lockfile_packages(content: str) -> dict[str, LockfilePackage]:
    """Parse ``uv.lock`` and return one entry per package name.

    When a package appears more than once (for example ``torch`` variants), the
    highest version string according to :class:`packaging.version.Version` wins.
    Workspace members without a pinned version are skipped.
    """
    data = tomllib.loads(content)
    packages: dict[str, LockfilePackage] = {}

    for pkg in data.get("package", []):
        name = pkg["name"]
        version = pkg.get("version")
        if version is None:
            continue

        wheel_urls = tuple(wheel["url"] for wheel in pkg.get("wheels", []))
        existing = packages.get(name)
        if existing is None or _version_key(version) > _version_key(existing.version):
            packages[name] = LockfilePackage(version=version, wheel_urls=wheel_urls)

    return packages


def _version_key(version: str) -> tuple[int, Version | str]:
    try:
        return (1, Version(version))
    except InvalidVersion:
        return (0, version)


def _license_from_classifiers(classifiers: list[str]) -> str:
    for classifier in classifiers:
        if classifier.startswith("License ::"):
            return classifier.split(" :: ", maxsplit=2)[-1].strip()
    return ""


def _normalize_license(info: dict[str, object]) -> str:
    license_expression = str(info.get("license_expression") or "").strip()
    if license_expression:
        return license_expression

    license_text = str(info.get("license") or "").strip()
    if license_text and "\n" not in license_text and len(license_text) <= 80:
        return license_text

    classifier_license = _license_from_classifiers(info.get("classifiers", []))
    if classifier_license:
        return classifier_license

    if license_text:
        first_line = license_text.splitlines()[0].strip()
        if first_line:
            return first_line

    return ""


def _parse_metadata_text(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    key: str | None = None
    for line in text.splitlines():
        if line.startswith(" ") and key:
            fields[key] += " " + line.strip()
        elif ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            fields[key] = value.strip()
    return fields


def _license_from_metadata_fields(fields: dict[str, str]) -> str:
    for field_name in ("License-Expression", "License"):
        value = fields.get(field_name, "").strip()
        if value:
            return value

    for field_name, value in fields.items():
        if field_name.startswith("Classifier") and "License ::" in value:
            return value.split(" :: ", maxsplit=2)[-1].strip()

    return ""


def _infer_license_from_text(text: str) -> str:
    normalized = text.casefold()
    if "apache license" in normalized and "version 2.0" in normalized:
        return "Apache-2.0"
    if "mit license" in normalized or (
        "permission is hereby granted" in normalized and "without warranty" in normalized
    ):
        return "MIT"
    if "bsd 3-clause" in normalized or "redistribution and use in source and binary forms" in normalized:
        return "BSD-3-Clause"
    if "nvidia proprietary" in normalized or "licenseref-nvidia-proprietary" in normalized:
        return "LicenseRef-NVIDIA-Proprietary"
    return ""


def _fetch_range(session: requests.Session, url: str, start: int, end: int) -> bytes:
    response = session.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
    response.raise_for_status()
    return response.content


def _read_wheel_metadata_via_ranges(session: requests.Session, url: str, file_size: int) -> str:
    tail_size = min(file_size, 65536 + 22)
    tail = _fetch_range(session, url, file_size - tail_size, file_size - 1)
    eocd_index = tail.rfind(EOCD_SIGNATURE)
    if eocd_index < 0:
        return ""

    central_directory_size, central_directory_offset = struct.unpack_from("<II", tail, eocd_index + 12)
    central_directory = _fetch_range(
        session,
        url,
        central_directory_offset,
        central_directory_offset + central_directory_size - 1,
    )

    position = 0
    metadata_entry: tuple[int, int, int, int] | None = None
    while position + 46 <= len(central_directory):
        if central_directory[position : position + 4] != CDFH_SIGNATURE:
            break

        compression_method = struct.unpack_from("<H", central_directory, position + 10)[0]
        compressed_size = struct.unpack_from("<I", central_directory, position + 20)[0]
        name_length = struct.unpack_from("<H", central_directory, position + 28)[0]
        extra_length = struct.unpack_from("<H", central_directory, position + 30)[0]
        comment_length = struct.unpack_from("<H", central_directory, position + 32)[0]
        local_header_offset = struct.unpack_from("<I", central_directory, position + 42)[0]
        entry_name = central_directory[position + 46 : position + 46 + name_length].decode()

        if entry_name.endswith(".dist-info/METADATA"):
            metadata_entry = (local_header_offset, compressed_size, compression_method, 0)
            break

        position += 46 + name_length + extra_length + comment_length

    if metadata_entry is None:
        return ""

    local_header_offset, compressed_size, compression_method, _ = metadata_entry
    local_header = _fetch_range(session, url, local_header_offset, local_header_offset + 4095)
    if local_header[:4] != LFH_SIGNATURE:
        return ""

    local_name_length = struct.unpack_from("<H", local_header, 26)[0]
    local_extra_length = struct.unpack_from("<H", local_header, 28)[0]
    data_offset = local_header_offset + 30 + local_name_length + local_extra_length
    compressed_data = _fetch_range(session, url, data_offset, data_offset + compressed_size - 1)

    if compression_method == 0:
        return compressed_data.decode()
    if compression_method == 8:
        return zlib.decompress(compressed_data, -zlib.MAX_WBITS).decode()
    return ""


def _license_from_wheel_archive(archive: zipfile.ZipFile) -> str:
    try:
        metadata_name = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        metadata_text = archive.read(metadata_name).decode()
    except (StopIteration, UnicodeDecodeError):
        return ""

    license_text = _license_from_metadata_fields(_parse_metadata_text(metadata_text))
    if license_text:
        return license_text

    for entry_name in archive.namelist():
        if ".dist-info/licenses/" in entry_name or entry_name.endswith(("/LICENSE", "/LICENSE.txt")):
            license_text = _infer_license_from_text(archive.read(entry_name).decode(errors="replace"))
            if license_text:
                return license_text

    return ""


def _wheel_content_length(session: requests.Session, url: str) -> int | None:
    try:
        response = session.head(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
    except requests.RequestException:
        return None

    content_length = response.headers.get("Content-Length")
    if content_length is None:
        return None
    return int(content_length)


def _license_from_wheel_url(session: requests.Session, url: str) -> str:
    content_length = _wheel_content_length(session, url)
    if content_length is not None and content_length > MAX_WHEEL_BYTES:
        try:
            metadata_text = _read_wheel_metadata_via_ranges(session, url, content_length)
        except requests.RequestException:
            return ""
        return _license_from_metadata_fields(_parse_metadata_text(metadata_text))

    try:
        response = session.get(url, timeout=120)
        response.raise_for_status()
    except requests.RequestException:
        return ""

    if len(response.content) > MAX_WHEEL_BYTES:
        return ""

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            return _license_from_wheel_archive(archive)
    except zipfile.BadZipFile:
        return ""


def _pick_wheel_urls(urls: list[dict[str, object]]) -> list[str]:
    wheels = [entry for entry in urls if entry.get("packagetype") == "bdist_wheel"]
    if not wheels:
        return []

    def sort_key(entry: dict[str, object]) -> tuple[int, int, str]:
        filename = str(entry.get("filename", ""))
        size = int(entry.get("size") or 0)
        universal = 0 if "py3-none-any" in filename or "py2.py3-none-any" in filename else 1
        return (universal, size, filename)

    return [str(entry["url"]) for entry in sorted(wheels, key=sort_key)]


def _license_from_wheel_urls(session: requests.Session, wheel_urls: list[str]) -> str:
    for wheel_url in wheel_urls:
        license_text = _license_from_wheel_url(session, wheel_url)
        if license_text:
            return license_text
    return ""


def _sanitize_table_cell(value: str) -> str:
    return " ".join(value.split())


def _pypi_version_candidates(version: str) -> list[str]:
    """Return PyPI version strings to try, from most to least specific."""
    candidates = [version]
    if "+" in version:
        base_version = version.split("+", 1)[0]
        if base_version not in candidates:
            candidates.append(base_version)
    return candidates


def _project_url_from_pypi_info(info: dict[str, object], name: str) -> str:
    project_url = str(info.get("project_url") or "").strip()
    if project_url:
        return project_url
    return PYPI_PROJECT_URL.format(name=name)


def _license_from_pypi_project(session: requests.Session, name: str) -> str:
    """Use the latest PyPI project metadata when a pinned release omits license fields."""
    try:
        response = session.get(f"https://pypi.org/pypi/{name}/json", timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return ""
    return _normalize_license(response.json().get("info", {}))


def load_license_overrides(path: Path | None) -> dict[str, LicenseOverride]:
    if path is None or not path.exists():
        return {}

    raw = json.loads(path.read_text(encoding="utf-8"))
    return {name: LicenseOverride.model_validate(value) for name, value in raw.items()}


def fetch_package_metadata(
    session: requests.Session,
    name: str,
    package: LockfilePackage,
    overrides: dict[str, LicenseOverride],
) -> tuple[str, str]:
    """Return ``(license, project_url)`` for a package."""
    override = overrides.get(name)
    if override is not None:
        project_url = override.project_url or PYPI_PROJECT_URL.format(name=name)
        return override.license, project_url

    project_url = PYPI_PROJECT_URL.format(name=name)
    for version_candidate in _pypi_version_candidates(package.version):
        url = f"https://pypi.org/pypi/{name}/{version_candidate}/json"
        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            continue

        payload = response.json()
        info = payload.get("info", {})
        project_url = _project_url_from_pypi_info(info, name)

        license_text = _normalize_license(info)
        if not license_text:
            license_text = _license_from_wheel_urls(session, _pick_wheel_urls(payload.get("urls", [])))
        if license_text:
            return license_text, project_url

    license_text = _license_from_wheel_urls(session, list(package.wheel_urls))
    if license_text:
        return license_text, project_url

    license_text = _license_from_pypi_project(session, name)
    if license_text:
        return license_text, project_url

    return "Unknown", project_url


def enrich_packages(
    packages: dict[str, LockfilePackage],
    *,
    overrides: dict[str, LicenseOverride],
    workers: int = 16,
    cache_path: Path | None = None,
    refresh_cache: bool = False,
) -> list[SbomEntry]:
    """Fetch package metadata and return sorted SBOM entries."""
    cache = _load_cache(cache_path) if cache_path else {}
    entries: list[SbomEntry] = []
    to_fetch: list[tuple[str, LockfilePackage]] = []

    for name, package in sorted(packages.items()):
        cache_key = f"{name}@{package.version}"
        if not refresh_cache and cache_key in cache:
            cached = cache[cache_key]
            entries.append(
                SbomEntry(
                    library=name,
                    version=package.version,
                    license=cached["license"],
                    project_url=cached["project_url"],
                ),
            )
            continue
        to_fetch.append((name, package))

    if to_fetch:
        typer.echo(f"Fetching metadata for {len(to_fetch)} packages...", err=True)
        session = requests.Session()
        session.headers.update({"Accept": "application/json", "User-Agent": "generate_sbom/1.0"})

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(fetch_package_metadata, session, name, package, overrides): (name, package)
                for name, package in to_fetch
            }
            fetched: dict[tuple[str, LockfilePackage], SbomEntry] = {}
            for future in as_completed(future_map):
                name, package = future_map[future]
                license_text, project_url = future.result()
                entry = SbomEntry(
                    library=name,
                    version=package.version,
                    license=license_text,
                    project_url=project_url,
                )
                fetched[(name, package)] = entry
                cache[f"{name}@{package.version}"] = {
                    "license": license_text,
                    "project_url": project_url,
                }

        if cache_path:
            _save_cache(cache_path, cache)

        entries.extend(fetched[name, package] for name, package in to_fetch)

    return sorted(entries, key=lambda entry: entry.library.lower())


def build_document(lockfile: Path, entries: list[SbomEntry]) -> SbomDocument:
    return SbomDocument(lockfile=str(lockfile), packages=entries)


def format_markdown(document: SbomDocument) -> str:
    lines = [
        "| Library | Version | License | Project URL |",
        "|---------|---------|---------|-------------|",
    ]
    for entry in document.packages:
        lines.append(
            "| {library} | {version} | {license} | {project_url} |".format(
                library=_sanitize_table_cell(entry.library),
                version=_sanitize_table_cell(entry.version),
                license=_sanitize_table_cell(entry.license),
                project_url=_sanitize_table_cell(entry.project_url),
            ),
        )
    return "\n".join(lines) + "\n"


def write_output(document: SbomDocument, output_format: OutputFormat, output_path: Path | None) -> None:
    if output_format is OutputFormat.json:
        payload = document.model_dump_json(indent=2) + "\n"
    else:
        payload = format_markdown(document)

    if output_path is None:
        sys.stdout.write(payload)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(payload, encoding="utf-8")


def load_document(path: Path) -> SbomDocument:
    content = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return SbomDocument.model_validate_json(content)

    packages = _parse_markdown_entries(content)
    return SbomDocument(lockfile="baseline", packages=packages)


def _parse_markdown_entries(content: str) -> list[SbomEntry]:
    entries: list[SbomEntry] = []
    for line in content.splitlines():
        match = MARKDOWN_ROW_RE.match(line.strip())
        if not match:
            continue
        groups = match.groupdict()
        library = groups["library"].strip()
        if library.lower() == "library" or set(library) <= {"-"}:
            continue
        entries.append(SbomEntry.model_validate(groups))
    if not entries:
        msg = "No SBOM table rows found in markdown baseline."
        raise typer.BadParameter(msg)
    return entries


def diff_documents(current: SbomDocument, baseline: SbomDocument) -> SbomDocument:
    baseline_names = {entry.library for entry in baseline.packages}
    new_entries = [entry for entry in current.packages if entry.library not in baseline_names]
    return SbomDocument(
        generated_at=current.generated_at,
        lockfile=current.lockfile,
        packages=sorted(new_entries, key=lambda entry: entry.library.lower()),
    )


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(path: Path, cache: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_lockfile(lockfile: Path) -> Path:
    if lockfile.exists():
        return lockfile
    msg = f"Lockfile not found: {lockfile}"
    raise typer.BadParameter(msg)


def _resolve_overrides(overrides: Path | None, no_overrides: bool) -> dict[str, LicenseOverride]:
    if no_overrides:
        return {}
    return load_license_overrides(overrides or DEFAULT_OVERRIDES)


def _generate_document(
    lockfile: Path,
    *,
    overrides: dict[str, LicenseOverride],
    cache_path: Path | None,
    refresh_cache: bool,
    workers: int,
) -> SbomDocument:
    packages = parse_lockfile_packages(lockfile.read_text(encoding="utf-8"))
    entries = enrich_packages(
        packages,
        overrides=overrides,
        workers=workers,
        cache_path=cache_path,
        refresh_cache=refresh_cache,
    )
    return build_document(lockfile, entries)


@app.command()
def generate(
    lockfile: Annotated[Path, typer.Option(help="Path to uv.lock.")] = Path(DEFAULT_LOCKFILE),
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Write output to this file.")] = None,
    markdown_output: Annotated[bool, typer.Option("--markdown", help="Emit a markdown table instead of JSON.")] = False,
    overrides: Annotated[
        Path | None,
        typer.Option(help="JSON file with manual license overrides."),
    ] = None,
    no_overrides: Annotated[
        bool,
        typer.Option("--no-overrides", help="Ignore license override file."),
    ] = False,
    cache: Annotated[
        Path | None,
        typer.Option(help="Optional JSON cache for metadata lookups."),
    ] = None,
    refresh_cache: Annotated[bool, typer.Option(help="Ignore cached metadata.")] = False,
    workers: Annotated[int, typer.Option(help="Concurrent metadata requests.")] = 16,
) -> None:
    """Generate an SBOM for all dependencies in uv.lock."""
    lockfile = _resolve_lockfile(lockfile)
    document = _generate_document(
        lockfile,
        overrides=_resolve_overrides(overrides, no_overrides),
        cache_path=cache,
        refresh_cache=refresh_cache,
        workers=workers,
    )

    if markdown_output:
        output_format = OutputFormat.markdown
    else:
        output_format = OutputFormat.json

    write_output(document, output_format, output)


@app.command()
def diff(
    baseline: Annotated[Path, typer.Argument(help="Previous SBOM (.json or markdown table).")],
    lockfile: Annotated[Path, typer.Option(help="Path to uv.lock.")] = Path(DEFAULT_LOCKFILE),
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Write output to this file.")] = None,
    markdown_output: Annotated[bool, typer.Option("--markdown", help="Emit a markdown table instead of JSON.")] = False,
    overrides: Annotated[
        Path | None,
        typer.Option(help="JSON file with manual license overrides."),
    ] = None,
    no_overrides: Annotated[
        bool,
        typer.Option("--no-overrides", help="Ignore license override file."),
    ] = False,
    cache: Annotated[
        Path | None,
        typer.Option(help="Optional JSON cache for metadata lookups."),
    ] = None,
    refresh_cache: Annotated[bool, typer.Option(help="Ignore cached metadata.")] = False,
    workers: Annotated[int, typer.Option(help="Concurrent metadata requests.")] = 16,
) -> None:
    """Report packages present in the current lockfile but absent from a baseline SBOM."""
    if not baseline.exists():
        msg = f"Baseline SBOM not found: {baseline}"
        raise typer.BadParameter(msg)

    lockfile = _resolve_lockfile(lockfile)
    current = _generate_document(
        lockfile,
        overrides=_resolve_overrides(overrides, no_overrides),
        cache_path=cache,
        refresh_cache=refresh_cache,
        workers=workers,
    )
    previous = load_document(baseline)
    delta = diff_documents(current, previous)

    typer.echo(
        f"Found {len(delta.packages)} new package(s) vs baseline ({len(previous.packages)} known).",
        err=True,
    )

    if markdown_output:
        output_format = OutputFormat.markdown
    else:
        output_format = OutputFormat.json

    write_output(delta, output_format, output)


if __name__ == "__main__":
    app()
