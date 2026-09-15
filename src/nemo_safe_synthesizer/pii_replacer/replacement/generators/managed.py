# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed-asset structured PII replacement generation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, ClassVar, Never

import numpy as np
import pandas as pd

from ....config.replace_pii import EntityType, PiiReplacementSettings, PiiSamplerBackend, PiiSamplerConfig
from ....errors import GenerationError
from ....observability import get_logger
from ...dependency_labels import MANAGED_DEPENDENCY_COLUMN_ALIASES
from ...planning.patterns import render_name_pattern
from ._common import (
    _dependency_values,
    _email_domain,
    _name_values,
    _render_address,
    _render_name,
    _required_pattern_values,
    _resolve_dependency_labels,
    normalize_organization_domain,
)
from .faker import FakerReplacementGenerator

if TYPE_CHECKING:
    from ..generation import ReplacementGenerationRequest

logger = get_logger(__name__)

__all__ = ["ManagedReplacementGenerator"]


class ManagedReplacementGenerator:
    """Generate structured replacements using managed person-sampling assets.

    Args:
        settings: Locale and seed configuration shared by replacement
            generators.
        sampler: Managed sampler configuration, including its asset path.

    """

    backend: ClassVar[PiiSamplerBackend] = PiiSamplerBackend.MANAGED

    def __init__(self, *, settings: PiiReplacementSettings, sampler: PiiSamplerConfig) -> None:
        if sampler.backend is not self.backend:
            raise ValueError("ManagedReplacementGenerator requires the managed sampler backend")
        self._settings = settings
        self._sampler = sampler
        self._fallback_generator = FakerReplacementGenerator(
            settings=settings,
            sampler=sampler.model_copy(update={"backend": PiiSamplerBackend.FAKER}),
        )
        self._managed_people: pd.DataFrame | None = None
        self._managed_value_columns: dict[str, str] = {}
        self._candidate_indexes: dict[EntityType, dict[str, np.ndarray]] = {}
        self._candidate_position_cache: dict[
            tuple[tuple[EntityType, tuple[str, ...]], ...],
            np.ndarray,
        ] = {}
        self._load_attempted = False
        self._warned_fallbacks: set[EntityType] = set()

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Generate a managed-asset replacement for ``request``."""
        if request.entity_type not in _MANAGED_ENTITY_TYPES:
            return self._fallback_generator.generate(request)

        managed_values = self._managed_values(request)
        replacement = None if managed_values is None else _generate_managed_value(request, managed_values)
        if replacement is not None:
            # The executor rejects unchanged values and retries this generator
            # with a new seed, producing another managed sample.
            return replacement
        self._warn_fallback(request.entity_type)
        return self._fallback_generator.generate(request)

    def _managed_values(self, request: ReplacementGenerationRequest) -> dict[str, str] | None:
        people = self._load_managed_people()
        if people is None or people.empty:
            return None

        candidate_positions = self._candidate_positions(request)
        if candidate_positions is None:
            row_position = request.seed % len(people)
        else:
            row_position = int(candidate_positions[request.seed % len(candidate_positions)])
        return _managed_row_values(people, row_position, self._managed_value_columns)

    def _candidate_positions(self, request: ReplacementGenerationRequest) -> np.ndarray | None:
        dependencies = _dependency_values(request.effective_dependency_tuple)
        resolved: list[tuple[EntityType, tuple[str, ...]]] = []
        for entity_type in MANAGED_DEPENDENCY_COLUMN_ALIASES:
            dependency_value = dependencies.get(entity_type)
            if dependency_value is None:
                continue
            labels = _resolve_dependency_labels(
                request,
                entity_type,
                dependency_value,
            )
            if labels is None:
                continue
            resolved.append((entity_type, labels))

        if not resolved:
            return None
        cache_key = tuple(resolved)
        cached = self._candidate_position_cache.get(cache_key)
        if cached is not None:
            return cached

        candidates: np.ndarray | None = None
        for entity_type, labels in resolved:
            label_index = self._candidate_indexes.get(entity_type)
            matching = _matching_positions(label_index, labels)
            if matching is None:
                _raise_no_managed_candidates(entity_type)
            candidates = matching if candidates is None else np.intersect1d(candidates, matching, assume_unique=True)
            if not len(candidates):
                _raise_no_managed_candidates(entity_type)
        if candidates is None:
            return None
        self._candidate_position_cache[cache_key] = candidates
        return candidates

    def _load_managed_people(self) -> pd.DataFrame | None:
        if self._load_attempted:
            return self._managed_people
        self._load_attempted = True
        path = self._sampler.resolved_managed_assets_path() / "datasets" / f"{self._settings.locale}.parquet"
        if not path.is_file():
            return None
        try:
            people = _read_managed_people(path)
            self._candidate_indexes, dependency_columns = _build_candidate_indexes(people)
            self._managed_people = people.drop(columns=dependency_columns)
            self._managed_value_columns = _managed_value_columns(self._managed_people)
        except Exception:
            logger.runtime.warning(
                "Managed PII replacement assets could not be read; affected entities will use Faker",
                extra={"locale": self._settings.locale},
            )
        return self._managed_people

    def _warn_fallback(self, entity_type: EntityType) -> None:
        if entity_type in self._warned_fallbacks:
            return
        self._warned_fallbacks.add(entity_type)
        logger.user.warning(
            "Managed PII sampling is unavailable for an entity category; using Faker",
            extra={"entity_type": entity_type.value, "locale": self._settings.locale},
        )


_MANAGED_ENTITY_TYPES = frozenset(
    {
        EntityType.FIRST_NAME,
        EntityType.MIDDLE_NAME,
        EntityType.LAST_NAME,
        EntityType.FULL_NAME,
        EntityType.EMAIL,
        EntityType.PHONE_NUMBER,
        EntityType.STREET_ADDRESS,
    }
)
_MANAGED_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "first": ("first_name", "given_name", "firstname"),
    "middle": ("middle_name", "middlename"),
    "last": ("last_name", "family_name", "surname", "lastname"),
    "email": ("email", "email_address"),
    "phone": ("phone_number", "phone"),
    "street": ("street_address", "address"),
    "street_number": ("street_number", "building_number"),
    "street_name": ("street_name",),
}
_MANAGED_READ_COLUMNS = tuple(
    dict.fromkeys(
        column
        for aliases in (*_MANAGED_COLUMN_ALIASES.values(), *MANAGED_DEPENDENCY_COLUMN_ALIASES.values())
        for column in aliases
    )
)


def _read_managed_people(path: Path) -> pd.DataFrame:
    try:
        available_columns = _available_parquet_columns(path)
    except Exception:
        # Tests and non-pyarrow parquet engines may not expose lightweight
        # schema inspection. The ordinary read still produces the same result.
        return pd.read_parquet(path)
    columns = [column for column in _MANAGED_READ_COLUMNS if column in available_columns]
    return pd.read_parquet(path, columns=columns, dtype_backend="pyarrow")


def _available_parquet_columns(path: Path) -> frozenset[str]:
    from pyarrow.parquet import ParquetFile

    return frozenset(ParquetFile(path).schema_arrow.names)


def _build_candidate_indexes(
    people: pd.DataFrame,
) -> tuple[dict[EntityType, dict[str, np.ndarray]], frozenset[str]]:
    indexes: dict[EntityType, dict[str, np.ndarray]] = {}
    dependency_columns: set[str] = set()
    for entity_type, aliases in MANAGED_DEPENDENCY_COLUMN_ALIASES.items():
        column = _first_existing_column(people, aliases)
        if column is None:
            continue
        dependency_columns.add(column)
        indexes[entity_type] = _build_label_index(people[column])
    return indexes, frozenset(dependency_columns)


def _build_label_index(values: pd.Series) -> dict[str, np.ndarray]:
    if isinstance(values.dtype, pd.ArrowDtype):
        return _build_arrow_label_index(values)

    # Preserve compatibility with tests and alternate parquet engines that
    # return ordinary pandas dtypes instead of Arrow-backed columns.
    normalized = values.astype("string").str.casefold()
    return {
        str(label): np.asarray(positions, dtype=np.int64)
        for label, positions in normalized.groupby(normalized, sort=False, dropna=True).indices.items()
    }


def _build_arrow_label_index(values: pd.Series) -> dict[str, np.ndarray]:
    import pyarrow as pa
    import pyarrow.compute as pc

    arrow_values = pa.array(values)
    if isinstance(arrow_values, pa.ChunkedArray):
        arrow_values = arrow_values.combine_chunks()
    encoded = arrow_values.dictionary_encode()
    codes = pc.fill_null(encoded.indices, -1).to_numpy(zero_copy_only=False)
    if not len(codes):
        return {}

    # Sort integer dictionary codes once instead of materializing and grouping
    # one Python string per managed row. Stable sorting keeps each label's row
    # positions in their original order, preserving deterministic sampling.
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    starts = np.concatenate(([0], np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1))
    stops = np.concatenate((starts[1:], [len(codes)]))
    dictionary = encoded.dictionary.to_pylist()
    grouped: dict[str, list[np.ndarray]] = {}
    for start, stop in zip(starts, stops, strict=True):
        code = int(sorted_codes[start])
        if code < 0:
            continue
        label = str(dictionary[code]).casefold()
        grouped.setdefault(label, []).append(order[start:stop])

    return {
        label: positions[0] if len(positions) == 1 else np.sort(np.concatenate(positions))
        for label, positions in grouped.items()
    }


def _matching_positions(
    label_index: Mapping[str, np.ndarray] | None,
    labels: tuple[str, ...],
) -> np.ndarray | None:
    if label_index is None:
        return None
    matches = [label_index[label] for label in labels if label in label_index]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return np.sort(np.concatenate(matches))


def _raise_no_managed_candidates(entity_type: EntityType) -> Never:
    raise GenerationError(
        "managed PII sampling found no candidates after applying dependency "
        f"entity type {entity_type.value!r} (candidate_count=0)"
    )


def _first_existing_column(dataframe: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    return next((column for column in aliases if column in dataframe.columns), None)


def _managed_value_columns(dataframe: pd.DataFrame) -> dict[str, str]:
    return {
        key: column
        for key, aliases in _MANAGED_COLUMN_ALIASES.items()
        if (column := _first_existing_column(dataframe, aliases)) is not None
    }


def _managed_row_values(
    dataframe: pd.DataFrame,
    row_position: int,
    columns: Mapping[str, str],
) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, column in columns.items():
        value = dataframe[column].iat[row_position]
        if pd.isna(value):
            continue
        values[key] = str(value)
    return values


def _generate_managed_value(
    request: ReplacementGenerationRequest,
    sampled: Mapping[str, str],
) -> str | None:
    dependencies = _dependency_values(request.effective_dependency_tuple)
    entity_type = request.entity_type
    rng = Random(request.seed)

    if entity_type in {
        EntityType.FIRST_NAME,
        EntityType.MIDDLE_NAME,
        EntityType.LAST_NAME,
        EntityType.FULL_NAME,
    }:
        values = _name_values(dependencies, sampled)
        required = (
            _required_pattern_values(entity_type, request.pattern)
            if request.pattern is not None
            else _required_name_values(entity_type)
        )
        if not _has_values(values, required):
            return None
        return _render_name(entity_type, request.pattern, rng, values)
    if entity_type is EntityType.EMAIL:
        if request.pattern is None:
            return sampled.get("email") or None
        values = _name_values(dependencies, sampled)
        values["domain"] = _email_domain(request.original_value) or ""
        organization = dependencies.get(EntityType.ORGANIZATION)
        values["organization"] = normalize_organization_domain(organization or "")
        required = _required_pattern_values(EntityType.EMAIL, request.pattern)
        if not _has_values(values, required):
            return None
        return render_name_pattern(EntityType.EMAIL, request.pattern, values, rng)
    if entity_type is EntityType.PHONE_NUMBER:
        if request.pattern is not None:
            return None
        return sampled.get("phone") or None
    if entity_type is EntityType.STREET_ADDRESS:
        street = sampled.get("street") or _managed_street_components(sampled)
        return _render_address(street, dependencies) if street else None
    return None


def _required_name_values(entity_type: EntityType) -> frozenset[str]:
    if entity_type is EntityType.FULL_NAME:
        return frozenset({"first", "last"})
    key = {
        EntityType.FIRST_NAME: "first",
        EntityType.MIDDLE_NAME: "middle",
        EntityType.LAST_NAME: "last",
    }[entity_type]
    return frozenset({key})


def _has_values(values: Mapping[str, str], required: frozenset[str]) -> bool:
    return all(values.get(key) for key in required)


def _managed_street_components(values: Mapping[str, str]) -> str:
    parts = [values.get("street_number", ""), values.get("street_name", "")]
    return " ".join(part for part in parts if part)
