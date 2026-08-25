# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Folder CSV load/write for multi-table PII replacement."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ...errors import ParameterError
from .schema import DatabaseSchema

__all__ = ["load_csv_tables", "write_csv_table", "write_csv_tables"]


def load_csv_tables(input_dir: Path | str, schema: DatabaseSchema) -> dict[str, pd.DataFrame]:
    """Load top-level ``*.csv`` files; stems must match schema table names.

    Non-recursive. Extra CSVs not in the schema, or schema tables missing a CSV,
    are errors.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise ParameterError(f"input_dir must be a directory of CSV tables, got {input_dir!r}")

    csv_files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    stems = {p.stem: p for p in csv_files}
    schema_tables = set(schema.tables)

    missing = sorted(schema_tables - set(stems))
    extra = sorted(set(stems) - schema_tables)
    if missing:
        raise ParameterError(
            f"CSV folder missing tables required by schema: {', '.join(missing)}"
        )
    if extra:
        raise ParameterError(
            f"CSV folder has tables not listed in schema: {', '.join(extra)}"
        )

    frames: dict[str, pd.DataFrame] = {}
    for name in schema.table_order_names():
        path = stems[name]
        df = pd.read_csv(path)
        if not df.index.is_unique:
            df = df.reset_index(drop=True)
        frames[name] = df
    return frames


def write_csv_table(table_name: str, df: pd.DataFrame, output_dir: Path | str) -> Path:
    """Write one transformed table as ``{table_name}.csv`` under ``output_dir``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{table_name}.csv"
    df.to_csv(path, index=False)
    return path


def write_csv_tables(tables: dict[str, pd.DataFrame], output_dir: Path | str) -> list[Path]:
    """Write one CSV per table, mirroring ``{table}.csv`` filenames."""
    return [write_csv_table(name, df, output_dir) for name, df in tables.items()]
