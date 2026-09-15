# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-driven structured replacement execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from ...config.replace_pii import DependencyValueMappings, EntityType, PiiColumnPlan, PiiReplacementPlan
from ...errors import GenerationError, InternalError, ParameterError
from ...observability import get_logger
from ..transform_result import ColumnStatistics, ReplacementGenerationStatistics
from .canonicalization import canonicalize_scalar, is_missing_scalar
from .compiler import compile_plan
from .detection import FreeTextDetector
from .free_text import FreeTextColumnStatistics, FreeTextReplacementExecutor
from .generation import ReplacementGenerationRequest, ReplacementGenerator, generated_value_is_valid
from .seeding import derive_seed, resolve_base_seed
from .types import (
    CanonicalValue,
    EffectiveDependencyTuple,
    GroupDependencyDrift,
    GroupMappingKey,
    GroupMappingProvenance,
    RecordMappingKey,
)

logger = get_logger(__name__)

__all__ = ["ReplacementExecutionResult", "StructuredReplacementExecutor", "resolve_base_seed"]

_MAX_GENERATION_ATTEMPTS = 10
_CompiledDependencyValueMappings = dict[str, dict[str, tuple[str, ...] | None]]
_COLLISION_SENSITIVE_ENTITY_TYPES = frozenset(
    {
        EntityType.EMAIL,
        EntityType.PHONE_NUMBER,
        EntityType.SSN,
        EntityType.NATIONAL_ID,
        EntityType.CREDIT_DEBIT_CARD,
        EntityType.API_KEY,
        EntityType.IPV4,
        EntityType.IPV6,
        EntityType.UNIQUE_IDENTIFIER,
    }
)


@dataclass(frozen=True, slots=True)
class ReplacementExecutionResult:
    """Internal structured-execution result returned to the public facade."""

    dataframe: pd.DataFrame
    column_statistics: dict[str, ColumnStatistics]
    generation_statistics: ReplacementGenerationStatistics
    dependency_drifts: tuple[GroupDependencyDrift, ...]


@dataclass(frozen=True, slots=True)
class _CachedReplacement:
    value: str = field(repr=False)
    provenance: GroupMappingProvenance | None = field(default=None, repr=False)


class StructuredReplacementExecutor:
    """Execute structured targets in dependency order on a dataframe copy."""

    def __init__(
        self,
        plan: PiiReplacementPlan,
        generator: ReplacementGenerator,
        *,
        group_column: str | None,
        base_seed: int,
        dependency_value_mappings: DependencyValueMappings | None = None,
        free_text_detector: FreeTextDetector | None = None,
    ) -> None:
        self._plan = plan
        self._generator = generator
        self._group_column = group_column
        self._base_seed = base_seed
        self._dependency_value_mappings = _compile_dependency_value_mappings(dependency_value_mappings or {})
        self._cache: dict[RecordMappingKey | GroupMappingKey, _CachedReplacement] = {}
        self._free_text = FreeTextReplacementExecutor(generator, free_text_detector, base_seed=base_seed)
        self._reserved_by_target: dict[str, set[str]] = {}
        self._dependency_conflicts: dict[tuple[str, frozenset[EntityType]], int] = {}
        self._generation_elapsed = 0.0
        self._generated_count = 0

    def execute(self, dataframe: pd.DataFrame) -> ReplacementExecutionResult:
        """Return a copy with every structured plan target replaced."""
        self._reset_execution_state()
        free_text_specs = [spec for spec in self._plan.columns_to_replace if spec.entity_type is EntityType.FREE_TEXT]

        working = dataframe.copy(deep=True)
        original_group_identities = self._snapshot_group_identities(dataframe)
        self._seed_reserved_values(dataframe)
        free_text_spans = self._free_text.detect(dataframe, free_text_specs)
        ordered_specs = compile_plan(self._plan)
        for spec in ordered_specs:
            if spec.entity_type is not EntityType.FREE_TEXT:
                self._execute_target(working, spec, original_group_identities)
        free_text_result = self._free_text.execute(
            working,
            free_text_specs,
            original_group_identities,
            free_text_spans,
        )
        _verify_result(dataframe, working, self._plan)

        drifts = self._dependency_drift_results()
        for drift in drifts:
            logger.user.warning(
                "PII group replacement reused the first value despite dependency drift",
                extra=drift.as_log_extra(),
            )
        return ReplacementExecutionResult(
            dataframe=working,
            column_statistics=_column_statistics(
                dataframe,
                working,
                self._plan,
                self._generator.backend.value,
                free_text_result.column_statistics,
            ),
            generation_statistics=ReplacementGenerationStatistics(
                generated_replacement_count=(self._generated_count + free_text_result.generated_replacement_count),
                elapsed_time_seconds=self._generation_elapsed + free_text_result.elapsed_time_seconds,
            ),
            dependency_drifts=drifts,
        )

    def _reset_execution_state(self) -> None:
        self._cache.clear()
        self._free_text.reset()
        self._reserved_by_target.clear()
        self._dependency_conflicts.clear()
        self._generation_elapsed = 0.0
        self._generated_count = 0

    def _snapshot_group_identities(self, dataframe: pd.DataFrame) -> tuple[CanonicalValue, ...] | None:
        if self._group_column is None:
            return None
        return tuple(_group_identity(value) for value in dataframe[self._group_column].tolist())

    def _seed_reserved_values(self, dataframe: pd.DataFrame) -> None:
        for spec in self._plan.columns_to_replace:
            if spec.entity_type not in _COLLISION_SENSITIVE_ENTITY_TYPES:
                continue
            self._reserved_by_target[spec.column_name] = {
                canonicalize_scalar(value).normalized_value
                for value in dataframe[spec.column_name].tolist()
                if not is_missing_scalar(value)
            }

    def _execute_target(
        self,
        working: pd.DataFrame,
        spec: PiiColumnPlan,
        group_identities: tuple[CanonicalValue, ...] | None,
    ) -> None:
        column_position = working.columns.get_loc(spec.column_name)
        if not isinstance(column_position, int):
            raise ParameterError(f"replacement column {spec.column_name!r} is duplicated in the dataframe")
        if working.iloc[:, column_position].notna().any():
            working[spec.column_name] = working[spec.column_name].astype(object)

        for row_position in range(len(working)):
            original = working.iat[row_position, column_position]
            if is_missing_scalar(original):
                continue
            canonical_original = canonicalize_scalar(original)
            dependencies = _effective_dependencies(working, row_position, spec)
            key = self._mapping_key(spec.column_name, row_position, canonical_original, group_identities)
            replacement = self._replacement_for(key, spec, canonical_original, dependencies)
            working.iat[row_position, column_position] = replacement

    def _mapping_key(
        self,
        target_column: str,
        row_position: int,
        original: CanonicalValue,
        group_identities: tuple[CanonicalValue, ...] | None,
    ) -> RecordMappingKey | GroupMappingKey:
        if group_identities is None:
            return RecordMappingKey(target_column, row_position, original)
        return GroupMappingKey(target_column, group_identities[row_position], original)

    def _replacement_for(
        self,
        key: RecordMappingKey | GroupMappingKey,
        spec: PiiColumnPlan,
        original: CanonicalValue,
        dependencies: EffectiveDependencyTuple,
    ) -> str:
        cached = self._cache.get(key)
        if cached is not None:
            self._record_dependency_drift(key, cached, dependencies)
            self._free_text.register_structured_mapping(key, spec, original, cached.value, dependencies)
            return cached.value

        replacement = self._generate_fresh(key, spec, original, dependencies)
        provenance = GroupMappingProvenance(dependencies) if isinstance(key, GroupMappingKey) else None
        self._cache[key] = _CachedReplacement(replacement, provenance)
        if (reserved_values := self._reserved_by_target.get(spec.column_name)) is not None:
            reserved_values.add(replacement)
        self._free_text.register_structured_mapping(key, spec, original, replacement, dependencies)
        self._generated_count += 1
        return replacement

    def _generate_fresh(
        self,
        key: RecordMappingKey | GroupMappingKey,
        spec: PiiColumnPlan,
        original: CanonicalValue,
        dependencies: EffectiveDependencyTuple,
    ) -> str:
        for attempt in range(_MAX_GENERATION_ATTEMPTS):
            request = ReplacementGenerationRequest(
                entity_type=spec.entity_type,
                original_value=original.normalized_value,
                effective_dependency_tuple=dependencies,
                pattern=spec.pattern,
                seed=derive_seed(self._base_seed, key, purpose="replacement", attempt=attempt),
                resolved_dependency_labels=_resolved_dependency_labels(
                    spec,
                    dependencies,
                    self._dependency_value_mappings,
                ),
            )
            started = time.perf_counter()
            try:
                candidate = self._generator.generate(request)
            except GenerationError:
                raise
            except Exception as exc:
                raise GenerationError(
                    f"PII replacement generation failed for column {spec.column_name!r} "
                    f"at row position {key.row_position if isinstance(key, RecordMappingKey) else 'group'}"
                ) from exc
            finally:
                self._generation_elapsed += time.perf_counter() - started
            reserved_values = self._reserved_by_target.get(spec.column_name)
            if generated_value_is_valid(request, candidate) and (
                reserved_values is None or candidate not in reserved_values
            ):
                return candidate
        raise GenerationError(
            f"PII replacement could not generate a distinct value for column {spec.column_name!r} "
            f"after {_MAX_GENERATION_ATTEMPTS} attempts"
        )

    def _record_dependency_drift(
        self,
        key: RecordMappingKey | GroupMappingKey,
        cached: _CachedReplacement,
        dependencies: EffectiveDependencyTuple,
    ) -> None:
        if not isinstance(key, GroupMappingKey) or cached.provenance is None:
            return
        if cached.provenance.effective_dependency_tuple == dependencies:
            return
        conditioner_types = frozenset(entity_type for entity_type, _ in dependencies)
        drift_key = (key.target_column, conditioner_types)
        self._dependency_conflicts[drift_key] = self._dependency_conflicts.get(drift_key, 0) + 1

    def _dependency_drift_results(self) -> tuple[GroupDependencyDrift, ...]:
        return tuple(
            GroupDependencyDrift(target, conditioner_types, count)
            for (target, conditioner_types), count in sorted(
                self._dependency_conflicts.items(),
                key=lambda item: (item[0][0], sorted(entity.value for entity in item[0][1])),
            )
        )


def _effective_dependencies(
    dataframe: pd.DataFrame,
    row_position: int,
    spec: PiiColumnPlan,
) -> EffectiveDependencyTuple:
    dependencies: list[tuple[EntityType, CanonicalValue | None]] = []
    for dependency in spec.depends_on:
        if dependency.entity_type is None:
            raise InternalError("resolved replacement dependency is missing its entity type")
        column_position = dataframe.columns.get_loc(dependency.column_name)
        if not isinstance(column_position, int):
            raise ParameterError(f"dependency column {dependency.column_name!r} is duplicated in the dataframe")
        value = dataframe.iat[row_position, column_position]
        canonical = None if is_missing_scalar(value) else canonicalize_scalar(value)
        dependencies.append((dependency.entity_type, canonical))
    return tuple(dependencies)


def _compile_dependency_value_mappings(
    mappings: DependencyValueMappings,
) -> _CompiledDependencyValueMappings:
    return {
        column_name: {
            source.casefold(): None if targets is None else tuple(target.casefold() for target in targets)
            for source, targets in value_mappings.items()
        }
        for column_name, value_mappings in mappings.items()
    }


def _resolved_dependency_labels(
    spec: PiiColumnPlan,
    dependencies: EffectiveDependencyTuple,
    mappings: _CompiledDependencyValueMappings,
) -> tuple[tuple[EntityType, tuple[str, ...] | None], ...]:
    resolved: list[tuple[EntityType, tuple[str, ...] | None]] = []
    for dependency, (entity_type, value) in zip(spec.depends_on, dependencies, strict=True):
        if value is None:
            continue
        source_mappings = mappings.get(dependency.column_name, {})
        normalized = value.normalized_value.casefold()
        resolved.append((entity_type, source_mappings.get(normalized, (normalized,))))
    return tuple(resolved)


def _group_identity(value: object) -> CanonicalValue:
    if is_missing_scalar(value):
        return CanonicalValue("missing_group", "")
    return canonicalize_scalar(value)


def _column_statistics(
    original: pd.DataFrame,
    transformed: pd.DataFrame,
    plan: PiiReplacementPlan,
    backend: str,
    free_text_statistics: dict[str, FreeTextColumnStatistics],
) -> dict[str, ColumnStatistics]:
    statistics: dict[str, ColumnStatistics] = {}
    for spec in plan.columns_to_replace:
        source = original[spec.column_name]
        entity = spec.entity_type.value
        if spec.entity_type is EntityType.FREE_TEXT:
            detected = free_text_statistics.get(spec.column_name, FreeTextColumnStatistics())
            changed = sum(
                not _equal_scalars(before, after)
                for before, after in zip(source.tolist(), transformed[spec.column_name].tolist())
            )
            statistics[spec.column_name] = ColumnStatistics(
                assigned_type="text",
                assigned_entity=entity,
                detected_entity_counts=detected.counts,
                detected_entity_values=detected.values,
                is_transformed=changed > 0,
                transform_functions={backend, *detected.sources} if changed else set(),
            )
            continue
        non_missing = [value for value in source.tolist() if not is_missing_scalar(value)]
        changed = sum(
            not _equal_scalars(before, after)
            for before, after in zip(source.tolist(), transformed[spec.column_name].tolist())
        )
        statistics[spec.column_name] = ColumnStatistics(
            assigned_type="text" if spec.entity_type is EntityType.FREE_TEXT else "structured",
            assigned_entity=entity,
            detected_entity_counts={entity: len(non_missing)},
            detected_entity_values={entity: {str(value) for value in non_missing}},
            is_transformed=changed > 0,
            transform_functions={"pattern" if spec.pattern is not None else backend} if changed else set(),
        )
    return statistics


def _equal_scalars(left: object, right: object) -> bool:
    if is_missing_scalar(left) and is_missing_scalar(right):
        return True
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _verify_result(original: pd.DataFrame, transformed: pd.DataFrame, plan: PiiReplacementPlan) -> None:
    """Defensively verify structural and no-unplanned-edit invariants."""
    if original.shape != transformed.shape:
        raise InternalError("PII replacement changed dataframe shape")
    if not original.index.equals(transformed.index):
        raise InternalError("PII replacement changed dataframe index")
    if not original.columns.equals(transformed.columns):
        raise InternalError("PII replacement changed dataframe column order")

    planned = {spec.column_name for spec in plan.columns_to_replace}
    for column in original.columns:
        if column not in planned and not original[column].equals(transformed[column]):
            raise InternalError(f"PII replacement changed unplanned column {column!r}")
        if column not in planned:
            continue
        missing_before = [is_missing_scalar(value) for value in original[column].tolist()]
        missing_after = [is_missing_scalar(value) for value in transformed[column].tolist()]
        if missing_before != missing_after:
            raise InternalError(f"PII replacement changed missing-value positions in column {column!r}")
