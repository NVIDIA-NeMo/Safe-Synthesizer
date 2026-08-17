# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persist and load the shared database-scope replacement map."""

from __future__ import annotations

from pathlib import Path

import yaml

from ...errors import ParameterError
from .store import SharedRuntimeStore

PII_REPLACEMENT_MAP_FILENAME = "pii_replacement_map.yaml"

__all__ = [
    "PII_REPLACEMENT_MAP_FILENAME",
    "load_replacement_map",
    "save_replacement_map",
]


def save_replacement_map(store: SharedRuntimeStore, path: Path | str) -> Path:
    """Serialize ``store`` to YAML at ``path``."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = store.to_persist_dict()
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True, default_flow_style=False)
    out.write_text(text)
    return out


def load_replacement_map(path: Path | str) -> SharedRuntimeStore:
    """Load a previously persisted replacement map."""
    p = Path(path)
    try:
        text = p.read_text()
    except OSError as exc:
        raise ParameterError(f"Could not read PII replacement map file {path!r}: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ParameterError(f"Invalid YAML in PII replacement map file {path!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise ParameterError(f"replacement map file {path!r} must contain a mapping")
    return SharedRuntimeStore.from_persist_dict(data)
