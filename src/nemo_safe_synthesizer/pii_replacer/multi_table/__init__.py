# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-table (database-scope) PII replacement orchestrator package.

Import leaf modules directly when inside the replacement engine to avoid
circular imports with ``replacement.apply``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .map_io import PII_REPLACEMENT_MAP_FILENAME, load_replacement_map, save_replacement_map
from .schema import DatabaseSchema, load_schema
from .store import SharedRuntimeStore, TableRunContext

if TYPE_CHECKING:
    from .replacer import MultiTablePiiReplacer as MultiTablePiiReplacer

__all__ = [
    "DatabaseSchema",
    "MultiTablePiiReplacer",
    "PII_REPLACEMENT_MAP_FILENAME",
    "SharedRuntimeStore",
    "TableRunContext",
    "load_replacement_map",
    "load_schema",
    "save_replacement_map",
]


def __getattr__(name: str) -> Any:
    if name == "MultiTablePiiReplacer":
        from .replacer import MultiTablePiiReplacer

        return MultiTablePiiReplacer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
