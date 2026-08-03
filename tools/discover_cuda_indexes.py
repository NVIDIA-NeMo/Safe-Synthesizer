#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discover the package indexes associated with a CUDA extra."""

from __future__ import annotations

import argparse
import sys
import tomllib
from collections.abc import Sequence
from pathlib import Path


def discover_cuda_index_urls(pyproject_path: Path, cuda_extra: str, minimum: int) -> list[str]:
    """Return unique index URLs matching a CUDA extra, preserving file order."""
    if minimum < 0:
        raise ValueError(f"minimum expected index count must be nonnegative, got {minimum}")

    with pyproject_path.open("rb") as handle:
        indexes = tomllib.load(handle)["tool"]["uv"]["index"]

    urls: list[str] = []
    seen_urls: set[str] = set()
    for index in indexes:
        name = index["name"]
        url = index["url"]
        if (name.endswith(f"-{cuda_extra}") or f"/{cuda_extra}" in url) and url not in seen_urls:
            urls.append(url)
            seen_urls.add(url)

    if len(urls) < minimum:
        raise ValueError(f"expected at least {minimum} {cuda_extra} indexes in {pyproject_path}, found {len(urls)}")
    return urls


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pyproject", type=Path, help="Path to the pyproject.toml file")
    parser.add_argument("cuda_extra", help="CUDA extra to match, such as cu129")
    parser.add_argument("minimum", type=int, help="Minimum number of unique matching indexes")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Print matching index URLs, one per line."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        urls = discover_cuda_index_urls(args.pyproject, args.cuda_extra, args.minimum)
    except (KeyError, OSError, TypeError, ValueError, tomllib.TOMLDecodeError) as exc:
        sys.stderr.write(f"CUDA index discovery failed: {exc}\n")
        return 1

    sys.stdout.write("".join(f"{url}\n" for url in urls))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
