# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A CSV file read once, so its typed and text forms come from the same bytes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse
from urllib.request import url2pathname

import pandas as pd

__all__ = ["TEXT_READ_EXCLUDED_LOAD_ARGS", "CsvSource"]

TEXT_READ_EXCLUDED_LOAD_ARGS = frozenset(
    {"converters", "dtype", "keep_default_na", "na_filter", "na_values", "parse_dates"}
)
"""``pd.read_csv`` options dropped from the text read, because they would override its every-cell-as-text options."""

# The extensions pandas uses to infer compression from a file name, in the order it checks them.
_Compression: TypeAlias = Literal["tar", "gzip", "bz2", "zip", "xz", "zstd"]
_COMPRESSION_BY_EXTENSION: tuple[tuple[str, _Compression], ...] = (
    (".tar", "tar"),
    (".tar.gz", "tar"),
    (".tar.bz2", "tar"),
    (".tar.xz", "tar"),
    (".gz", "gzip"),
    (".bz2", "bz2"),
    (".zip", "zip"),
    (".xz", "xz"),
    (".zst", "zstd"),
)
# Options that only apply when pandas opens the location itself; memory_map needs a real file descriptor.
_LOCATION_ONLY_LOAD_ARGS = frozenset({"compression", "memory_map", "storage_options"})


@dataclass(frozen=True, slots=True)
class CsvSource:
    """A CSV location and its raw bytes, read once.

    ``read`` parses the file with type inference, as ``pd.read_csv`` does.
    ``read_text`` parses the same bytes again with every cell kept as the text
    in the file, so a ZIP code ``"02054"`` stays ``"02054"`` and ``"N/A"``
    stays ``"N/A"``. Keeping the bytes rather than a second dataframe means
    the text form costs no memory until it is needed, and the file can change
    on disk in between without the two forms disagreeing.

    Local paths, ``file://`` URLs, and ``http(s)`` URLs are read once with
    ``load``. Other locations, such as ``s3://`` URLs, keep no bytes:
    ``read`` and ``read_text`` each let pandas open the location, so the
    text form is only as current as the second read.

    Example:
        source = CsvSource.load("patients.csv.gz", {"sep": ";"})
        typed = source.read()
        text = source.read_text()  # Parsed from the same bytes as ``typed``.
    """

    location: str
    """Path or URL the file was read from."""

    content: bytes | None = field(repr=False)
    """Raw file bytes, or ``None`` when pandas reads ``location`` directly."""

    load_args: Mapping[str, Any] = field(default_factory=dict)
    """Extra ``pd.read_csv`` options for the typed read."""

    @classmethod
    def load(cls, location: str, load_args: Mapping[str, Any] | None = None) -> CsvSource:
        """Read the raw bytes of ``location`` once.

        HTTP(S) URLs are fetched with ``httpx``, following redirects, and a
        ``storage_options`` mapping is sent as request headers, as pandas does.

        Args:
            location: Local path, ``file://`` URL, or other URL that pandas can read.
            load_args: Extra ``pd.read_csv`` options.

        Raises:
            FileNotFoundError: If a local ``location`` does not exist.
            httpx.HTTPError: If an HTTP(S) request fails or returns an error status.
        """
        args = dict(load_args or {})
        return cls(location, _read_bytes(location, args.get("storage_options")), args)

    def read(self) -> pd.DataFrame:
        """Parse the file with type inference and the configured ``load_args``."""
        return self._read_csv(self.load_args)

    def read_text(self) -> pd.DataFrame:
        """Parse the file with every cell as the string written in it, including blanks and ``"N/A"``.

        Options in ``TEXT_READ_EXCLUDED_LOAD_ARGS`` are dropped; separators,
        encodings, and other parsing options still apply.
        """
        args = {key: value for key, value in self.load_args.items() if key not in TEXT_READ_EXCLUDED_LOAD_ARGS}
        return self._read_csv({**args, "dtype": str, "na_filter": False})

    def _read_csv(self, load_args: Mapping[str, Any]) -> pd.DataFrame:
        """Parse ``content``, or let pandas open ``location`` when no bytes were kept."""
        if self.content is None:
            return pd.read_csv(self.location, **load_args)
        args = {key: value for key, value in load_args.items() if key not in _LOCATION_ONLY_LOAD_ARGS}
        compression = load_args.get("compression", "infer")
        if compression == "infer":
            compression = _infer_compression(self.location)
        return pd.read_csv(BytesIO(self.content), compression=compression, **args)


def _read_bytes(location: str, storage_options: object) -> bytes | None:
    """Return the bytes at a local path or HTTP(S) URL, or ``None`` for locations pandas must open itself."""
    parsed = urlparse(location)
    if parsed.scheme in {"http", "https"}:
        import httpx

        headers = (
            {str(key): str(value) for key, value in storage_options.items()}
            if isinstance(storage_options, Mapping)
            else None
        )
        # Like pandas, set no read timeout: large files over slow links must not fail midway.
        response = httpx.get(location, headers=headers, follow_redirects=True, timeout=httpx.Timeout(None, connect=30))
        response.raise_for_status()
        return response.content
    if parsed.scheme == "file":
        return Path(url2pathname(parsed.path)).read_bytes()
    # A one-letter scheme is a Windows drive letter, as in "C:\data.csv".
    if len(parsed.scheme) <= 1:
        return Path(location).expanduser().read_bytes()
    return None


def _infer_compression(location: str) -> _Compression | None:
    """Return the compression pandas would infer from the file name of ``location``."""
    parsed = urlparse(location)
    lowered = (parsed.path if parsed.scheme in {"http", "https"} else location).lower()
    return next((method for extension, method in _COMPRESSION_BY_EXTENSION if lowered.endswith(extension)), None)
