# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plan-driven structured replacement execution."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from ...config.replace_pii import DataToSamplerValueMapping, EntityType, PiiColumnPlan, PiiReplacementPlan
from ...errors import DataError, GenerationError, InternalError, ParameterError, SafeSynthesizerError
from ...observability import get_logger
from ..transform_result import (
    ColumnStatistics,
    ReplacementGenerationStatistics,
    ReplacementMap,
    StructuredReplacementRecord,
)
from .addresses import AddressParts, detect_address_parts
from .canonicalization import canonicalize_scalar, is_missing_scalar, readable_value
from .compiler import compile_plan
from .detection import FreeTextDetector
from .free_text import FreeTextColumnStatistics, FreeTextReplacementExecutor
from .generation import ReplacementGenerationRequest, ReplacementGenerator
from .identifier_space import IdentifierAllocator, NumberSpace, identifier_allocator
from .seeding import derive_seed, resolve_base_seed
from .types import (
    CanonicalValue,
    EffectiveDependencyTuple,
    GroupDependencyDrift,
    GroupMappingKey,
    GroupMappingProvenance,
    RecordMappingKey,
    StructuredMappingKey,
)
from .value_checks import generated_value_is_valid

logger = get_logger(__name__)

__all__ = ["ReplacementExecutionResult", "StructuredReplacementExecutor", "resolve_base_seed"]

_MAX_GENERATION_ATTEMPTS = 10
_CompiledDataToSamplerValueMapping = dict[str, dict[str, tuple[str, ...]]]
# Identifier-like values: replacements avoid every original value in the column, so real identifiers
# do not reappear on other rows, and stay unique while the column's format has unused values. Both
# are best effort: when retries cannot avoid them, a replacement repeats an earlier one or, failing
# that, equals another row's original, with a warning. A replacement never equals its own original.
_IDENTIFIER_ENTITY_TYPES = frozenset(
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
    """Copy of the input with every structured target replaced."""

    column_statistics: dict[str, ColumnStatistics]
    """Per-column replacement metadata, keyed by column name."""

    generation_statistics: ReplacementGenerationStatistics
    """Aggregate generation counts and timing."""

    dependency_drifts: tuple[GroupDependencyDrift, ...]
    """Group mappings reused although their dependency values changed."""

    replacement_map: ReplacementMap | None
    """Per-occurrence original and replacement values when capture was requested, otherwise ``None``."""


@dataclass(frozen=True, slots=True)
class _CachedReplacement:
    value: str = field(repr=False)
    """Replacement reused for every occurrence with the same mapping key."""

    provenance: GroupMappingProvenance | None = field(default=None, repr=False)
    """Dependencies of the group occurrence that generated the value, or ``None`` in record scope."""


class StructuredReplacementExecutor:
    """Execute structured targets in dependency order on a dataframe copy."""

    def __init__(
        self,
        plan: PiiReplacementPlan,
        generator: ReplacementGenerator,
        *,
        group_column: str | None,
        base_seed: int,
        free_text_detector: FreeTextDetector | None = None,
        capture_replacement_map: bool = False,
        find_and_replace: bool = False,
    ) -> None:
        self._plan = plan
        self._generator = generator
        self._group_column = group_column
        self._base_seed = base_seed
        self._data_to_sampler_value_mapping = _compile_data_to_sampler_value_mapping(plan.data_to_sampler_value_mapping)
        self._capture_replacement_map = capture_replacement_map
        self._cache: dict[StructuredMappingKey, _CachedReplacement] = {}
        self._detector = free_text_detector
        self._address_parts: dict[str, dict[str, AddressParts]] = {}
        self._free_text = FreeTextReplacementExecutor(
            generator,
            free_text_detector,
            base_seed=base_seed,
            capture_replacement_map=capture_replacement_map,
            find_and_replace=find_and_replace,
        )
        self._replacement_records: list[StructuredReplacementRecord] = []
        self._originals_by_target: dict[str, set[str]] = {}
        self._generated_by_target: dict[str, set[str]] = {}
        self._repeated_by_target: dict[str, int] = {}
        self._reused_originals_by_target: dict[str, int] = {}
        self._identifier_allocators: dict[str, IdentifierAllocator] = {}
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
        self._prepare_identifier_allocators(dataframe)
        self._address_parts = detect_address_parts(
            self._detector,
            dataframe,
            [spec for spec in self._plan.columns_to_replace if spec.entity_type is EntityType.STREET_ADDRESS],
        )
        free_text_detections = self._free_text.detect(dataframe, free_text_specs)
        ordered_specs = compile_plan(self._plan)
        for spec in ordered_specs:
            if spec.entity_type is not EntityType.FREE_TEXT:
                self._execute_target(working, spec, original_group_identities)
        free_text_result = self._free_text.execute(
            working,
            free_text_specs,
            original_group_identities,
            free_text_detections,
        )
        _verify_result(dataframe, working, self._plan)

        for column_name, repeated in sorted(self._repeated_by_target.items()):
            logger.user.warning(
                "Some identifier replacements repeat because the column's format has no unused values left",
                extra={"column": column_name, "repeated_count": repeated},
            )
        for column_name, reused in sorted(self._reused_originals_by_target.items()):
            logger.user.warning(
                "Some identifier replacements equal another row's original value because no other value was found",
                extra={"column": column_name, "reused_original_count": reused},
            )
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
                generator_time_seconds=self._generation_elapsed + free_text_result.generator_time_seconds,
                dependency_drift_count=sum(drift.conflict_count for drift in drifts),
            ),
            dependency_drifts=drifts,
            replacement_map=(
                ReplacementMap(
                    structured=tuple(self._replacement_records),
                    free_text=free_text_result.replacement_records,
                )
                if self._capture_replacement_map
                else None
            ),
        )

    def _reset_execution_state(self) -> None:
        self._cache.clear()
        self._address_parts = {}
        self._free_text.reset()
        self._replacement_records.clear()
        self._originals_by_target.clear()
        self._generated_by_target.clear()
        self._repeated_by_target.clear()
        self._reused_originals_by_target.clear()
        self._identifier_allocators.clear()
        self._dependency_conflicts.clear()
        self._generation_elapsed = 0.0
        self._generated_count = 0

    def _snapshot_group_identities(self, dataframe: pd.DataFrame) -> tuple[CanonicalValue, ...] | None:
        if self._group_column is None:
            return None
        values = dataframe[self._group_column].tolist()
        if missing := sum(1 for value in values if is_missing_scalar(value)):
            raise DataError(
                f"group column {self._group_column!r} has {missing} missing value(s); PII replacement needs a group "
                "identifier on every row so unrelated rows do not share replacements"
            )
        return tuple(canonicalize_scalar(value) for value in values)

    def _seed_reserved_values(self, dataframe: pd.DataFrame) -> None:
        # Originals are recorded in the same text form generators receive and return,
        # so a replacement can never reuse another row's original value.
        for spec in self._plan.columns_to_replace:
            if spec.entity_type not in _IDENTIFIER_ENTITY_TYPES:
                continue
            column = dataframe[spec.column_name]
            self._originals_by_target[spec.column_name] = {
                text
                for value, text in zip(column.tolist(), column.astype(str).tolist(), strict=True)
                if not is_missing_scalar(value)
            }
            self._generated_by_target[spec.column_name] = set()

    def _prepare_identifier_allocators(self, dataframe: pd.DataFrame) -> None:
        for spec in self._plan.columns_to_replace:
            if spec.entity_type not in _IDENTIFIER_ENTITY_TYPES:
                continue
            column = dataframe[spec.column_name]
            originals = [
                text
                for value, text in zip(column.tolist(), column.astype(str).tolist(), strict=True)
                if not is_missing_scalar(value)
            ]
            allocator = identifier_allocator(spec.entity_type, spec.pattern, originals)
            if allocator is None:
                continue
            self._identifier_allocators[spec.column_name] = allocator
            longest = max((len(text.split(".", 1)[0]) for text in originals), default=0)
            if isinstance(allocator.space, NumberSpace) and allocator.space.digits > longest:
                logger.user.warning(
                    "Widened replacement identifiers so every distinct value can get an unused replacement",
                    extra={
                        "column": spec.column_name,
                        "original_digits": longest,
                        "replacement_digits": allocator.space.digits,
                    },
                )

    def _execute_target(
        self,
        working: pd.DataFrame,
        spec: PiiColumnPlan,
        group_identities: tuple[CanonicalValue, ...] | None,
    ) -> None:
        column_position = working.columns.get_loc(spec.column_name)
        if not isinstance(column_position, int):
            raise ParameterError(f"replacement column {spec.column_name!r} is duplicated in the dataframe")
        # Generators receive the same text form that plan validation checked patterns against;
        # the typed canonical identity is kept for cache keys and seeds.
        original_texts = working.iloc[:, column_position].astype(str).tolist()
        if working.iloc[:, column_position].notna().any():
            working[spec.column_name] = working[spec.column_name].astype(object)
        address_parts = self._address_parts.get(spec.column_name, {})

        for row_position in range(len(working)):
            original = working.iat[row_position, column_position]
            if is_missing_scalar(original):
                continue
            canonical_original = canonicalize_scalar(original)
            dependencies = _effective_dependencies(working, row_position, spec)
            key = self._mapping_key(spec, row_position, canonical_original, group_identities)
            replacement = self._replacement_for(
                key,
                spec,
                original_texts[row_position],
                dependencies,
                address_parts.get(original_texts[row_position]),
            )
            working.iat[row_position, column_position] = replacement
            if self._capture_replacement_map:
                self._replacement_records.append(
                    StructuredReplacementRecord(
                        row_position=row_position,
                        column_name=spec.column_name,
                        entity_type=spec.entity_type,
                        scope="record" if group_identities is None else "group",
                        original_value=original_texts[row_position],
                        replacement_value=replacement,
                    )
                )

    def _mapping_key(
        self,
        spec: PiiColumnPlan,
        row_position: int,
        original: CanonicalValue,
        group_identities: tuple[CanonicalValue, ...] | None,
    ) -> StructuredMappingKey:
        if group_identities is None:
            return RecordMappingKey(spec.column_name, row_position, original)
        return GroupMappingKey(spec.column_name, group_identities[row_position], original)

    def _replacement_for(
        self,
        key: StructuredMappingKey,
        spec: PiiColumnPlan,
        original_text: str,
        dependencies: EffectiveDependencyTuple,
        address_parts: AddressParts | None,
    ) -> str:
        cached = self._cache.get(key)
        if cached is not None:
            self._record_dependency_drift(key, cached, dependencies)
            self._free_text.register_structured_mapping(
                key, spec, original_text, cached.value, dependencies, address_parts
            )
            return cached.value

        if address_parts is None:
            replacement = self._generate_fresh(key, spec, original_text, dependencies)
        else:
            # Only the detected street is generated; the original locality text after it is kept as written.
            street = self._generate_fresh(key, spec, address_parts.street, dependencies, keep_original_locality=True)
            replacement = street + address_parts.locality
        provenance = None if isinstance(key, RecordMappingKey) else GroupMappingProvenance(dependencies)
        self._cache[key] = _CachedReplacement(replacement, provenance)
        if (generated_values := self._generated_by_target.get(spec.column_name)) is not None:
            generated_values.add(replacement)
        self._free_text.register_structured_mapping(key, spec, original_text, replacement, dependencies, address_parts)
        self._generated_count += 1
        return replacement

    def _generate_fresh(
        self,
        key: StructuredMappingKey,
        spec: PiiColumnPlan,
        original_text: str,
        dependencies: EffectiveDependencyTuple,
        *,
        keep_original_locality: bool = False,
    ) -> str:
        if (allocator := self._identifier_allocators.get(spec.column_name)) is not None:
            return self._allocate_identifier(key, spec, original_text, allocator)
        originals = self._originals_by_target.get(spec.column_name, set())
        generated = self._generated_by_target.get(spec.column_name, set())
        repeat: str | None = None
        reused_original: str | None = None
        for attempt in range(_MAX_GENERATION_ATTEMPTS):
            request = ReplacementGenerationRequest(
                entity_type=spec.entity_type,
                original_value=original_text,
                effective_dependency_tuple=dependencies,
                pattern=spec.pattern,
                seed=derive_seed(self._base_seed, key, purpose="replacement", attempt=attempt),
                resolved_dependency_values=_resolved_dependency_values(
                    spec,
                    dependencies,
                    self._data_to_sampler_value_mapping,
                ),
                keep_original_locality=keep_original_locality,
            )
            started = time.perf_counter()
            try:
                candidate = self._generator.generate(request)
            except SafeSynthesizerError:
                raise
            except Exception as exc:
                raise GenerationError(
                    f"PII replacement generation failed for column {spec.column_name!r} "
                    f"at row position {key.row_position if isinstance(key, RecordMappingKey) else 'group'}"
                ) from exc
            finally:
                self._generation_elapsed += time.perf_counter() - started
            # A replacement must differ from its own original; that rule is strict.
            if not generated_value_is_valid(request, candidate):
                continue
            if candidate in originals:
                reused_original = reused_original or candidate
            elif candidate in generated:
                repeat = repeat or candidate
            else:
                return candidate
        # No new value was found. Prefer repeating an earlier replacement over using another row's original.
        if repeat is not None:
            self._repeated_by_target[spec.column_name] = self._repeated_by_target.get(spec.column_name, 0) + 1
            return repeat
        if reused_original is not None:
            self._reused_originals_by_target[spec.column_name] = (
                self._reused_originals_by_target.get(spec.column_name, 0) + 1
            )
            return reused_original
        raise GenerationError(
            f"PII replacement could not generate a value that differs from the original for column "
            f"{spec.column_name!r} after {_MAX_GENERATION_ATTEMPTS} attempts"
        )

    def _allocate_identifier(
        self,
        key: StructuredMappingKey,
        spec: PiiColumnPlan,
        original_text: str,
        allocator: IdentifierAllocator,
    ) -> str:
        """Take the next unused value from the column's identifier space, starting at a seeded position."""
        started = time.perf_counter()
        seed = derive_seed(self._base_seed, key, purpose="replacement", attempt=0)
        allocation = allocator.allocate(seed, original_text)
        self._generation_elapsed += time.perf_counter() - started
        if allocation is None:
            raise ParameterError(
                f"column {spec.column_name!r}: its identifier format cannot produce any value other than the "
                "original; use a pattern with more generated positions"
            )
        replacement, kind = allocation
        if kind == "repeat":
            self._repeated_by_target[spec.column_name] = self._repeated_by_target.get(spec.column_name, 0) + 1
        elif kind == "original":
            self._reused_originals_by_target[spec.column_name] = (
                self._reused_originals_by_target.get(spec.column_name, 0) + 1
            )
        return replacement

    def _record_dependency_drift(
        self,
        key: StructuredMappingKey,
        cached: _CachedReplacement,
        dependencies: EffectiveDependencyTuple,
    ) -> None:
        if isinstance(key, RecordMappingKey) or cached.provenance is None:
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


def _compile_data_to_sampler_value_mapping(
    mappings: DataToSamplerValueMapping,
) -> _CompiledDataToSamplerValueMapping:
    return {
        column_name: {
            source.casefold(): tuple(target.casefold() for target in targets)
            for source, targets in value_mappings.items()
        }
        for column_name, value_mappings in mappings.items()
    }


def _resolved_dependency_values(
    spec: PiiColumnPlan,
    dependencies: EffectiveDependencyTuple,
    mappings: _CompiledDataToSamplerValueMapping,
) -> tuple[tuple[EntityType, tuple[str, ...]], ...]:
    resolved: list[tuple[EntityType, tuple[str, ...]]] = []
    for dependency, (entity_type, value) in zip(spec.depends_on, dependencies, strict=True):
        if value is None:
            continue
        source_mappings = mappings.get(dependency.column_name, {})
        normalized = readable_value(value).casefold()
        resolved.append((entity_type, source_mappings.get(normalized, (normalized,))))
    return tuple(resolved)


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
