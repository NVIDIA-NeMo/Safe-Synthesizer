# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampler value catalogs used by dependency-mapping discovery."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from ..config.replace_pii import EntityType, PiiSamplerBackend, PiiSamplerConfig
from ..errors import ParameterError

SamplerValueCatalog = Mapping[EntityType, tuple[str, ...]]

# The persona column each conditioning entity type reads. Every locale calls sex ``sex``; the ethnicity
# column is named differently by locale (en_US and en_SG, en_GB, pt_BR), and most locales have none.
PERSONA_DEPENDENCY_COLUMN_ALIASES: Mapping[EntityType, tuple[str, ...]] = {
    EntityType.GENDER: ("sex",),
    EntityType.ETHNIC_BACKGROUND: ("ethnic_background", "ethnicity", "race"),
}

_FAKER_DEPENDENCY_VALUES: SamplerValueCatalog = {
    EntityType.GENDER: ("female", "male"),
}

PERSONA_SEX_VALUES: Mapping[str, str] = {
    "female": "female",
    "male": "male",
    "女": "female",
    "男": "male",
    "여자": "female",
    "남자": "male",
    "महिला": "female",
    "पुरुष": "male",
    "mahila": "female",
    "purush": "male",
    "femme": "female",
    "homme": "male",
    "weiblich": "female",
    "männlich": "male",
    "feminino": "female",
    "masculino": "male",
}
"""Casefolded Nemotron-Personas ``sex`` values by locale and the gender each one means.

Only the English locales use ``Female`` and ``Male``; ja_JP uses ``女``, fr_FR ``Femme``, de_DE ``Weiblich``, and so on.
Every sampler accepts ``female`` and ``male``, so a value mapping need not depend on the locale.
"""


def gender_sampler_value(value: str) -> str:
    """Return ``female`` or ``male`` for a Nemotron-Personas ``sex`` value, else ``value`` casefolded.

    Example:
        ``Femme`` and ``female`` both give ``female``; a value missing from ``PERSONA_SEX_VALUES`` is kept.
    """
    key = value.casefold()
    return PERSONA_SEX_VALUES.get(key, key)


def nemotron_personas_dataset_path(sampler: PiiSamplerConfig, locale: str) -> Path:
    """Return where the Nemotron-Personas dataset for ``locale`` is read from, whether or not it exists."""
    return sampler.resolved_nemotron_personas_path() / "datasets" / f"{locale}.parquet"


def require_nemotron_personas_dataset(sampler: PiiSamplerConfig, locale: str) -> Path:
    """Return the Nemotron-Personas dataset for the configured locale, or raise when it is missing.

    Selecting the Nemotron-Personas sampler requires its data; NSS does not
    silently substitute Faker here. An unset ``replace_pii.sampler`` switches
    to Faker before this check; see ``resolve_replacement_config``.

    Raises:
        ParameterError: If the locale's dataset file does not exist.
    """
    path = nemotron_personas_dataset_path(sampler, locale)
    if not path.is_file():
        raise ParameterError(
            f"replace_pii.sampler.backend is {PiiSamplerBackend.NEMOTRON_PERSONAS.value}, but no "
            f"Nemotron-Personas data for locale {locale!r} was found at {path}. Download the "
            "Nemotron-Personas dataset, point replace_pii.sampler.nemotron_personas_path or "
            f"NSS_NEMOTRON_PERSONAS_PATH at it, or set replace_pii.sampler.backend to {PiiSamplerBackend.FAKER.value}"
        )
    return path


def load_sampler_value_catalog(sampler: PiiSamplerConfig, locale: str) -> SamplerValueCatalog:
    """Return the dependency values understood by the selected sampler.

    Nemotron-Personas gender values include ``female`` and ``male`` as well as the locale's own words, such as
    ``femme`` and ``homme`` in fr_FR, so a mapping written for Faker also works with personas.

    Raises:
        ParameterError: If the Nemotron-Personas dataset is missing or cannot be read.
    """
    if sampler.backend is PiiSamplerBackend.FAKER:
        return dict(_FAKER_DEPENDENCY_VALUES)

    path = require_nemotron_personas_dataset(sampler, locale)
    try:
        available_columns = available_parquet_columns(path)
        selected = {
            entity_type: column
            for entity_type, aliases in PERSONA_DEPENDENCY_COLUMN_ALIASES.items()
            if (column := next((alias for alias in aliases if alias in available_columns), None)) is not None
        }
        dataframe = pd.read_parquet(path, columns=list(selected.values()), dtype_backend="pyarrow")
    except Exception as exc:
        raise ParameterError(f"Nemotron-Personas data at {path} could not be read") from exc

    catalog: dict[EntityType, tuple[str, ...]] = {}
    for entity_type, column in selected.items():
        values = {str(value).casefold() for value in dataframe[column].dropna().unique().tolist()}
        if entity_type is EntityType.GENDER:
            values |= {gender_sampler_value(value) for value in values}
        catalog[entity_type] = tuple(sorted(values))
    return catalog


def available_parquet_columns(path: Path) -> frozenset[str]:
    """Return the column names in a parquet file's schema without reading its data."""
    from pyarrow.parquet import ParquetFile

    return frozenset(ParquetFile(path).schema_arrow.names)
