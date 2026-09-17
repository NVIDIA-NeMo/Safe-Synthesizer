# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampler label catalogs used by dependency-mapping discovery."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from ..config.replace_pii import EntityType, PiiReplacementSettings, PiiSamplerBackend, PiiSamplerConfig
from ..observability import get_logger

logger = get_logger(__name__)

DependencyLabelCatalog = Mapping[EntityType, tuple[str, ...]]

MANAGED_DEPENDENCY_COLUMN_ALIASES: Mapping[EntityType, tuple[str, ...]] = {
    EntityType.GENDER: ("gender", "sex"),
    EntityType.ETHNIC_BACKGROUND: ("ethnic_background", "ethnicity"),
}

_FAKER_DEPENDENCY_LABELS: DependencyLabelCatalog = {
    EntityType.GENDER: ("female", "male"),
}


def dependency_label_catalog(
    settings: PiiReplacementSettings,
    sampler: PiiSamplerConfig,
) -> DependencyLabelCatalog:
    """Return the dependency labels understood by the selected sampler."""
    if sampler.backend is PiiSamplerBackend.FAKER:
        return dict(_FAKER_DEPENDENCY_LABELS)

    path = sampler.resolved_managed_assets_path() / "datasets" / f"{settings.locale}.parquet"
    if not path.is_file():
        return {}
    try:
        available_columns = _available_parquet_columns(path)
        selected = {
            entity_type: column
            for entity_type, aliases in MANAGED_DEPENDENCY_COLUMN_ALIASES.items()
            if (column := next((alias for alias in aliases if alias in available_columns), None)) is not None
        }
        dataframe = pd.read_parquet(path, columns=list(selected.values()), dtype_backend="pyarrow")
    except Exception:
        logger.runtime.warning(
            "Managed PII dependency labels could not be read; automatic label mappings will be unavailable",
            extra={"locale": settings.locale},
        )
        return {}

    return {
        entity_type: tuple(sorted({str(value).casefold() for value in dataframe[column].dropna().unique().tolist()}))
        for entity_type, column in selected.items()
    }


def _available_parquet_columns(path: Path) -> frozenset[str]:
    from pyarrow.parquet import ParquetFile

    return frozenset(ParquetFile(path).schema_arrow.names)
