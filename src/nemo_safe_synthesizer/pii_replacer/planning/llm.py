# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible two-pass LLM enhancement for PII replacement plans."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence, Set
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Literal, TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from ...config.replace_pii import (
    ENTITIES,
    ENTITY_BY_TYPE,
    EXCLUSIVE_DEPENDS_ON_GROUPS,
    EntityType,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
)
from ...errors import GenerationError, InternalError, ParameterError
from ...observability import get_logger
from ..llm_client import (
    InvalidInferenceResponse,
    LLMTransport,
    OpenAICompatibleTransport,
    TransientInferenceError,
    resolve_inference_settings,
)
from .patterns import pattern_grammar_catalog
from .plan_builder import (
    ColumnClassification,
    DependencyCandidate,
    apply_dependencies,
    derive_dependency_candidates,
    plan_from_classifications,
)
from .resolver import ColumnProfile, PlanDiscoveryInput, PlanEnhancer
from .validation import column_pattern_issue, column_pattern_syntax_issue

__all__ = [
    "LLMPlanEnhancer",
]

# Every request is split into batches of at most this many entries (column
# profiles or dependency targets) and this many bytes of compact JSON evidence.
MAX_BATCH_ENTRIES = 32
MAX_BATCH_BYTES = 48 * 1024
MAX_REQUEST_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 30.0

logger = get_logger(__name__)


class _StructuredResponse(BaseModel):
    """Strict base for LLM-authored response envelopes."""

    model_config = ConfigDict(extra="forbid")


class _ProposedClassification(_StructuredResponse):
    """One LLM classification before NSS drops a pattern it cannot use.

    ``ColumnClassification`` rejects ineligible patterns, which is right for
    heuristic and user-authored input. An LLM pattern that cannot apply is
    harmless and unambiguous to drop, so it must not fail the whole batch.
    """

    column_name: str
    entity_type: EntityType | None
    pattern: str | None


class _ClassificationResponse(_StructuredResponse):
    classifications: list[_ProposedClassification]


class _PatternRepairResponse(_StructuredResponse):
    pattern: str


class _InvalidStructuredOutputError(GenerationError):
    """Structured output stayed invalid after every allowed attempt."""


ResponseT = TypeVar("ResponseT", bound=_StructuredResponse)
ResultT = TypeVar("ResultT")
BatchT = TypeVar("BatchT")


def _profile_payload(profile: ColumnProfile) -> dict[str, object]:
    return {
        "column_name": profile.column_name,
        "dtype": profile.dtype,
        "non_null_count": profile.non_null_count,
        "unique_count": profile.unique_count,
        "unique_ratio": profile.unique_ratio,
        "samples": list(profile.samples),
    }


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _json_bytes(value: object) -> int:
    return len(_compact_json(value).encode())


def _bounded_batches(
    payloads: Sequence[dict[str, object]],
    *,
    kind: str,
    name_key: str,
) -> list[list[dict[str, object]]]:
    """Split request entries into batches within ``MAX_BATCH_ENTRIES`` and ``MAX_BATCH_BYTES``.

    Args:
        payloads: JSON-ready entries, kept in order.
        kind: Entry description used in the oversize error, such as ``"Column profile"``.
        name_key: Payload key naming the entry in the oversize error.

    Returns:
        Consecutive, non-empty batches covering every entry.

    Raises:
        ParameterError: If a single entry exceeds ``MAX_BATCH_BYTES`` on its own.
    """
    batches: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for payload in payloads:
        if _json_bytes([payload]) > MAX_BATCH_BYTES:
            raise ParameterError(f"{kind} for {payload[name_key]!r} exceeds the 48 KiB LLM evidence limit")
        candidate = [*current, payload]
        if current and (len(candidate) > MAX_BATCH_ENTRIES or _json_bytes(candidate) > MAX_BATCH_BYTES):
            batches.append(current)
            current = [payload]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _profile_batches(profiles: Sequence[ColumnProfile]) -> list[list[dict[str, object]]]:
    return _bounded_batches(
        [_profile_payload(profile) for profile in profiles],
        kind="Column profile",
        name_key="column_name",
    )


def _entity_catalog() -> list[dict[str, object]]:
    return [
        {
            "entity_type": entity.entity_type.value,
            "pattern_syntax": entity.pattern_syntax.name.lower() if entity.pattern_syntax is not None else None,
        }
        for entity in ENTITIES
    ]


def _entity_types_without_pattern() -> str:
    return ", ".join(entity.entity_type.value for entity in ENTITIES if entity.pattern_syntax is None)


def _heuristic_classifications_payload(
    baseline: PiiReplacementPlan,
    batch: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    submitted = {str(profile["column_name"]) for profile in batch}
    return [
        {
            "column_name": spec.column_name,
            "entity_type": spec.entity_type.value,
            "pattern": spec.pattern,
        }
        for spec in baseline.columns_to_replace
        if spec.column_name in submitted
    ]


def _heuristic_dependency_edges(baseline: PiiReplacementPlan) -> frozenset[tuple[str, str]]:
    return frozenset(
        (spec.column_name, dependency.column_name)
        for spec in baseline.columns_to_replace
        for dependency in spec.depends_on
    )


def _exclusive_dependency_groups_payload() -> list[list[list[str]]]:
    return [
        [sorted(entity_type.value for entity_type in group) for group in family]
        for family in EXCLUSIVE_DEPENDS_ON_GROUPS
    ]


def _validation_feedback(exc: Exception) -> str:
    """Render a bounded, input-free description of why a response was rejected.

    Pydantic errors are rendered without their inputs so model-authored text
    and samples are not echoed back. The result is capped because it is
    appended to the next attempt's prompt.
    """
    if isinstance(exc, ValidationError):
        details = exc.errors(include_input=False, include_url=False)[:5]
        rendered = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'response'}: {item['msg']}" for item in details
        )
    else:
        rendered = str(exc)
    return rendered[:800]


def _quoted(names: Sequence[str]) -> str:
    return ", ".join(repr(name) for name in names)


def _usable_pattern(item: _ProposedClassification, protected_columns: Set[str]) -> str | None:
    """Return the proposed pattern, or ``None`` when NSS could not use it for this column.

    Unclassified columns, protected columns, and entity types without a pattern
    syntax never take a pattern, so dropping one loses nothing.
    """
    pattern = item.pattern
    if (
        pattern is None
        or not pattern.strip()
        or item.entity_type is None
        or item.column_name in protected_columns
        or ENTITY_BY_TYPE[item.entity_type].pattern_syntax is None
    ):
        if pattern is not None:
            logger.debug("Ignoring an LLM-proposed pattern the column cannot use", extra={"column": item.column_name})
        return None
    return pattern


def _classification_coverage_issue(expected: Sequence[str], actual: Sequence[str]) -> str | None:
    """Name the columns that make ``actual`` differ from exactly one entry per ``expected`` column."""
    counts = Counter(actual)
    expected_names = set(expected)
    problems = [
        f"{label}: {_quoted(names)}"
        for label, names in (
            ("missing", [name for name in expected if name not in counts]),
            ("duplicated", sorted(name for name, count in counts.items() if count > 1)),
            ("not submitted", sorted(set(counts) - expected_names)),
        )
        if names
    ]
    if not problems:
        return None
    return "classifications must contain every submitted column exactly once; " + "; ".join(problems)


def _classification_messages(
    discovery_input: PlanDiscoveryInput,
    batch: Sequence[Mapping[str, object]],
    baseline: PiiReplacementPlan,
) -> list[dict[str, str]]:
    system = (
        "You are helping NVIDIA NeMo Safe Synthesizer (NSS) de-identify a table before it is used to train a "
        "synthetic-data model. NSS replaces sensitive values with realistic fake values of the same kind. Your job is "
        "only to say what kind of data each column holds, and optionally describe its format so the fake values look "
        "like the real ones. NSS decides on its own which columns to replace.\n"
        "\n"
        "Input\n"
        "- column_profiles: one entry per column, with its name, dtype, counts, and a few sample values.\n"
        "- entity_catalog: the entity types you may choose from, each with the pattern grammar it supports.\n"
        "- discovery_context.group_column: the column that groups rows (for example, one patient's events).\n"
        "- heuristic_classifications: guesses from a rule-based detector, only for columns it flagged. They can be "
        "wrong, and a column missing from this list may still hold sensitive data.\n"
        "- pattern_grammars: the pattern languages you may use.\n"
        "\n"
        "For each column\n"
        "1. Set entity_type to the catalog entry that matches what the values mean, or null if none fits. Classify "
        "every column, including the group column.\n"
        "2. Set pattern to null unless all of these are true:\n"
        "   - entity_type is not null and that entity has a pattern_syntax. These entity types never take a "
        f"pattern: {_entity_types_without_pattern()};\n"
        "   - every sample value shares one format that the entity type alone does not capture. If the values mix "
        "formats, set pattern to null.\n"
        "3. A pattern must use exactly the grammar named by the entity's pattern_syntax and describe the whole cell "
        "value. Patterns are not regular expressions. Name placeholders such as {first} or {last} do not require "
        "name columns in the table; NSS fills them with a generated name when no related column exists.\n"
        "\n"
        "Output\n"
        "Return a JSON object with one key, classifications: a list with one object per submitted column, in the "
        "same order. Do not skip, repeat, or add columns. Each object has exactly the keys column_name, entity_type "
        "(a catalog value or null), and pattern (a string or null). Example:\n"
        '{"classifications":[{"column_name":"phone","entity_type":"phone_number","pattern":"###-###-####"},'
        '{"column_name":"visit_count","entity_type":null,"pattern":null}]}'
    )
    user = _compact_json(
        {
            "discovery_context": {"group_column": discovery_input.group_column},
            "entity_catalog": _entity_catalog(),
            "pattern_grammars": pattern_grammar_catalog(),
            "heuristic_classifications": _heuristic_classifications_payload(baseline, batch),
            "column_profiles": batch,
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _pattern_syntax_name(entity_type: EntityType, pattern: str | None) -> str | None:
    pattern_syntax = ENTITY_BY_TYPE[entity_type].pattern_syntax
    if pattern is None or pattern_syntax is None:
        return None
    return pattern_syntax.name.lower()


DependencyOptions = dict[str, dict[EntityType, list[str]]]
"""Target column -> source entity type -> permitted source columns, in candidate order."""


def _dependency_options(
    candidates: Sequence[DependencyCandidate],
    entity_types: Mapping[str, EntityType],
) -> DependencyOptions:
    options: DependencyOptions = {}
    for candidate in candidates:
        source_entity_type = entity_types[candidate.source_column]
        options.setdefault(candidate.target_column, {}).setdefault(source_entity_type, []).append(
            candidate.source_column
        )
    return options


def _create_response_model(name: str, fields: Mapping[str, tuple[object, object]]) -> type[_StructuredResponse]:
    """Create a strict response model from ``(type, Field(...))`` field definitions.

    ``create_model`` accepts fields only as ``**field_definitions`` typed ``Any``.
    Typed values would also be checked against its own keyword parameters, such
    as ``__doc__`` and ``__module__``, so the definitions cross that boundary as
    ``Any`` here and nowhere else.
    """
    field_definitions: dict[str, Any] = dict(fields)
    return create_model(name, __base__=_StructuredResponse, **field_definitions)


def _dependency_response_model(options: DependencyOptions) -> type[_StructuredResponse]:
    """Build the response schema for one dependency-selection request.

    The response has one object per target column and, inside it, one field per
    permitted source entity type whose value is one permitted source column or
    null. A target therefore cannot select two sources of the same entity type,
    and cannot name a column that is not a candidate; with schema-constrained
    decoding the model cannot even produce such output. Field names are
    positional because column names need not be Python identifiers; aliases
    carry the real names into the schema and the parsed JSON.
    """
    target_fields: dict[str, tuple[object, object]] = {}
    for target_index, (target_column, sources_by_type) in enumerate(options.items()):
        source_fields: dict[str, tuple[object, object]] = {
            f"source_{type_index}": (
                # The permitted columns are only known at runtime, so the Literal is built from them.
                Literal[tuple(source_columns)] | None,  # ty: ignore[invalid-type-form]
                Field(default=None, alias=entity_type.value),
            )
            for type_index, (entity_type, source_columns) in enumerate(sources_by_type.items())
        }
        target_model = _create_response_model(f"_DependencyTarget{target_index}", source_fields)
        target_fields[f"target_{target_index}"] = (
            target_model,
            Field(default_factory=target_model, alias=target_column),
        )
    return _create_response_model("_DependencySelectionResponse", target_fields)


def _selected_dependencies(response: _StructuredResponse, options: DependencyOptions) -> list[DependencyCandidate]:
    selected: list[DependencyCandidate] = []
    for target_index, (target_column, sources_by_type) in enumerate(options.items()):
        choices = getattr(response, f"target_{target_index}")
        for type_index in range(len(sources_by_type)):
            source_column = getattr(choices, f"source_{type_index}")
            if source_column is not None:
                selected.append(DependencyCandidate(target_column=target_column, source_column=source_column))
    return selected


def _dependency_target_payloads(
    plan: PiiReplacementPlan,
    options: DependencyOptions,
    baseline: PiiReplacementPlan,
) -> list[dict[str, object]]:
    """Describe each dependency target and its permitted sources, in ``options`` order."""
    heuristic_edges = _heuristic_dependency_edges(baseline)
    specs = {spec.column_name: spec for spec in plan.columns_to_replace}
    targets: list[dict[str, object]] = []
    for target_column, sources_by_type in options.items():
        # Targets are replacement columns, so read their repaired patterns from the plan.
        spec = specs[target_column]
        targets.append(
            {
                "target_column": target_column,
                "target_entity_type": spec.entity_type.value,
                "target_pattern": spec.pattern,
                "target_pattern_syntax": _pattern_syntax_name(spec.entity_type, spec.pattern),
                "source_options": {
                    entity_type.value: {
                        "columns": source_columns,
                        "heuristic_choice": next(
                            (source for source in source_columns if (target_column, source) in heuristic_edges),
                            None,
                        ),
                    }
                    for entity_type, source_columns in sources_by_type.items()
                },
            }
        )
    return targets


def _dependency_selection_messages(targets: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    system = (
        "You are helping NVIDIA NeMo Safe Synthesizer (NSS) de-identify a table before it is used to train a "
        "synthetic-data model. NSS replaces sensitive values with realistic fake values. A dependency tells NSS to "
        "generate a target column's fake value using a source column in the same row as context, so related fake "
        "values stay consistent. For example, a fake first name can match the gender in the same row, and a fake "
        "email can be built from the fake first and last name in the same row.\n"
        "\n"
        "Input\n"
        "- dependency_targets: one entry per target column, with its entity type and target_pattern, the format "
        "proposed for its fake values (or null; target_pattern_syntax names its grammar in pattern_grammars). "
        "source_options lists, for each source entity type the target may depend on, the columns of that type. "
        "heuristic_choice is the column a rule-based detector chose for that type, or null; it can be wrong, so "
        "treat it as a hint, not a requirement. When the pattern uses a name part such as {first} or {last}, a "
        "source column holding that name part lets the fake value match the fake name in the same row.\n"
        "- exclusive_dependency_groups: families of source entity types that must not be mixed (see rule 3).\n"
        "\n"
        "Rules\n"
        "1. For each target and each source entity type, choose at most one column: the one that describes the "
        "same person or record as the target. Tables can describe several people per row, such as a customer and "
        "a spouse or several children; match each target to its own "
        "person by column name, for example a spouse's first name to the spouse's gender, never to another person's.\n"
        "2. Choose null when no column of that type describes the same person or record, or when the source adds "
        "nothing beyond the other sources chosen for the same target.\n"
        "3. Check exclusivity separately for each target column. Each outer list in exclusive_dependency_groups is a "
        "family of inner groups. Within a family, the source entity types chosen for one target may come from at "
        "most one inner group; several types from the same inner group are fine. For example, with the family "
        "[[first_name, last_name, middle_name], [full_name]], a target may depend on first_name and last_name, or on "
        "full_name, but not on both first_name and full_name.\n"
        "\n"
        "Output\n"
        "Return a JSON object with one key per target column in dependency_targets. Each value is an object with one "
        "key per source entity type listed in that target's source_options, set to one of the listed columns or null. "
        "For example, for a target column email whose source_options are first_name and last_name:\n"
        '{"email":{"first_name":"given_name","last_name":null}}'
    )
    used_syntaxes = {str(target["target_pattern_syntax"]) for target in targets if target["target_pattern_syntax"]}
    user = _compact_json(
        {
            "dependency_targets": list(targets),
            "exclusive_dependency_groups": _exclusive_dependency_groups_payload(),
            # Only document the grammars that the targets' patterns actually use.
            "pattern_grammars": {
                name: grammar for name, grammar in pattern_grammar_catalog().items() if name in used_syntaxes
            },
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _apply_dependency_selection(
    plan: PiiReplacementPlan,
    selected: Sequence[DependencyCandidate],
    classifications: Sequence[ColumnClassification],
) -> PiiReplacementPlan:
    """Apply selected dependencies, raising ``ValueError`` with specific feedback when invalid."""
    try:
        return apply_dependencies(plan, selected, classifications=classifications)
    except (ParameterError, ValidationError) as exc:
        raise ValueError(
            "selected dependencies do not form a valid replacement plan: " + _validation_feedback(exc)
        ) from exc


def _pattern_repair_messages(
    profile: ColumnProfile,
    spec: PiiColumnPlan,
    issue: str,
) -> list[dict[str, str]]:
    pattern_syntax = ENTITY_BY_TYPE[spec.entity_type].pattern_syntax
    if pattern_syntax is None:
        raise InternalError(f"entity_type {spec.entity_type.value!r} has a pattern but no pattern syntax")
    system = (
        "Repair only the optional whole-column pattern. Return one non-empty pattern that follows the supplied pattern "
        "grammar exactly and describes the complete cell values represented by the samples. Patterns are not regular "
        "expressions. Do not change the column or entity type. Return a JSON object with one key, pattern, for "
        'example {"pattern":"###-###-####"}.'
    )
    syntax_name = pattern_syntax.name.lower()
    user = _compact_json(
        {
            "column_profile": _profile_payload(profile),
            "entity_type": spec.entity_type.value,
            "pattern_syntax": syntax_name,
            "pattern_grammar": pattern_grammar_catalog()[syntax_name],
            "invalid_pattern": spec.pattern,
            "validation_issue": issue,
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _with_feedback(messages: Sequence[Mapping[str, str]], feedback: str | None) -> list[dict[str, str]]:
    result = [dict(message) for message in messages]
    if feedback is not None:
        result.append(
            {
                "role": "user",
                "content": "The previous structured response was invalid. Correct it using this validation feedback: "
                + feedback,
            }
        )
    return result


def _retry_delay(attempt: int, exc: TransientInferenceError) -> float:
    """Return the wait before retrying after transient failure ``attempt``.

    A server-supplied ``Retry-After`` wins. Otherwise the delay doubles per
    attempt with equal jitter, so concurrent batches that failed together do
    not retry in lockstep. Both forms are capped.
    """
    if exc.retry_after is not None:
        return min(exc.retry_after, RETRY_MAX_DELAY_SECONDS)
    ceiling = min(RETRY_BASE_DELAY_SECONDS * 2 ** (attempt - 1), RETRY_MAX_DELAY_SECONDS)
    return ceiling / 2 + random.uniform(0, ceiling / 2)


def _raise_if_invalid_output_exhausted(purpose: str, attempt: int) -> None:
    if attempt == MAX_REQUEST_ATTEMPTS:
        # ``from None`` keeps model-authored output out of the exception chain.
        raise _InvalidStructuredOutputError(
            f"{purpose} returned invalid structured output after {MAX_REQUEST_ATTEMPTS} attempts"
        ) from None


class LLMPlanEnhancer(PlanEnhancer):
    """Enhance a heuristic plan with classification and dependency-selection passes.

    Pass one classifies every column's entity type and optional pattern. NSS
    then derives replacement membership and all permitted dependency
    candidates deterministically, and pass two only selects useful candidate
    IDs. Invalid optional patterns get focused repair requests.

    Args:
        config: Persisted LLM behavior; the endpoint, key, and model resolve
            through ``resolve_inference_settings``.
        transport: Structured-response transport; defaults to an
            ``OpenAICompatibleTransport`` for the resolved settings.
        environ: Environment mapping for settings resolution; defaults to ``os.environ``.
        sleep: Function used to wait between transient retries.

    Attributes:
        settings: Resolved inference settings.

    Raises:
        ParameterError: If the inference settings are invalid.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: LLMTransport | None = None,
        environ: Mapping[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = resolve_inference_settings(config, environ=environ)
        self._transport = transport or OpenAICompatibleTransport(self.settings)
        self._sleep = sleep

    def enhance(
        self,
        discovery_input: PlanDiscoveryInput,
        baseline: PiiReplacementPlan,
    ) -> PiiReplacementPlan:
        """Return a semantically classified, deterministically assembled plan."""
        classifications = self._classify_columns(discovery_input, baseline)
        expected_columns = {profile.column_name for profile in discovery_input.column_profiles}
        plan = plan_from_classifications(
            classifications,
            expected_columns=expected_columns,
            protected_columns=discovery_input.protected_columns,
        )
        # Repair first so dependency selection sees each target's final pattern.
        plan = self._repair_invalid_patterns(discovery_input, plan)
        candidates = derive_dependency_candidates(
            classifications,
            expected_columns=expected_columns,
            protected_columns=discovery_input.protected_columns,
        )
        if candidates:
            plan = self._select_dependencies(plan, candidates, baseline, classifications)
        return plan

    def _classify_columns(
        self,
        discovery_input: PlanDiscoveryInput,
        baseline: PiiReplacementPlan,
    ) -> list[ColumnClassification]:
        batches = _profile_batches(discovery_input.column_profiles)
        if not batches:
            return []
        classify = partial(self._classify_batch, discovery_input, baseline)
        return [classification for batch in self._map_batches(classify, batches) for classification in batch]

    def _map_batches(self, request: Callable[[BatchT], ResultT], batches: Sequence[BatchT]) -> list[ResultT]:
        """Run one request per batch, concurrently up to ``max_workers``, preserving batch order."""
        if len(batches) == 1:
            return [request(batches[0])]
        with ThreadPoolExecutor(max_workers=min(self.settings.max_workers, len(batches))) as executor:
            return list(executor.map(request, batches))

    def _classify_batch(
        self,
        discovery_input: PlanDiscoveryInput,
        baseline: PiiReplacementPlan,
        batch: list[dict[str, object]],
    ) -> list[ColumnClassification]:
        expected = [str(profile["column_name"]) for profile in batch]
        protected = discovery_input.protected_columns.intersection(expected)

        def parse(response: _ClassificationResponse) -> list[ColumnClassification]:
            actual = [classification.column_name for classification in response.classifications]
            if issue := _classification_coverage_issue(expected, actual):
                raise ValueError(issue)
            by_name = {
                item.column_name: ColumnClassification(
                    column_name=item.column_name,
                    entity_type=item.entity_type,
                    pattern=_usable_pattern(item, protected),
                )
                for item in response.classifications
            }
            return [by_name[name] for name in expected]

        return self._request_structured(
            purpose="PII column classification",
            messages=_classification_messages(discovery_input, batch, baseline),
            response_model=_ClassificationResponse,
            parse=parse,
        )

    def _select_dependencies(
        self,
        plan: PiiReplacementPlan,
        candidates: Sequence[DependencyCandidate],
        baseline: PiiReplacementPlan,
        classifications: Sequence[ColumnClassification],
    ) -> PiiReplacementPlan:
        entity_types = {
            classification.column_name: classification.entity_type
            for classification in classifications
            if classification.entity_type is not None
        }
        options = _dependency_options(candidates, entity_types)
        batches = _bounded_batches(
            _dependency_target_payloads(plan, options, baseline),
            kind="Dependency options",
            name_key="target_column",
        )
        select = partial(self._select_batch_dependencies, plan, options, classifications)
        selected = [dependency for batch in self._map_batches(select, batches) for dependency in batch]
        try:
            return apply_dependencies(plan, selected, classifications=classifications)
        except (ParameterError, ValidationError) as exc:
            # Every selection rule applies to one target, and the entity catalog's
            # dependency graph is acyclic, so batches valid on their own stay valid together.
            raise InternalError("Dependency batches that were valid individually failed together") from exc

    def _select_batch_dependencies(
        self,
        plan: PiiReplacementPlan,
        options: DependencyOptions,
        classifications: Sequence[ColumnClassification],
        targets: Sequence[Mapping[str, object]],
    ) -> list[DependencyCandidate]:
        batch_options = {str(target["target_column"]): options[str(target["target_column"])] for target in targets}

        def parse(response: _StructuredResponse) -> list[DependencyCandidate]:
            selected = _selected_dependencies(response, batch_options)
            # Validate this batch on its own so retry feedback names its specific conflicts.
            _apply_dependency_selection(plan, selected, classifications)
            return selected

        return self._request_structured(
            purpose="PII dependency selection",
            messages=_dependency_selection_messages(targets),
            response_model=_dependency_response_model(batch_options),
            parse=parse,
        )

    def _request_structured(
        self,
        *,
        purpose: str,
        messages: Sequence[Mapping[str, str]],
        response_model: type[ResponseT],
        parse: Callable[[ResponseT], ResultT],
    ) -> ResultT:
        """Request, validate, and parse one structured response with bounded retries.

        Every request path shares this loop. Each attempt resends the original
        messages plus feedback describing only the latest invalid response, so
        the prompt does not grow across attempts. A transient failure keeps that
        feedback for the next attempt.

        Args:
            purpose: Human-readable request name used in error messages.
            messages: Original chat messages for every attempt.
            response_model: Strict response envelope to validate against.
            parse: Converts a validated response into the result; raises
                ``ValueError`` with specific feedback when the response is
                semantically invalid.

        Returns:
            The value returned by ``parse``.

        Raises:
            ParameterError: Immediately, on permanent configuration or authentication failures.
            GenerationError: When transient failures persist through every attempt.
            _InvalidStructuredOutputError: When every attempt returns invalid output.
        """
        feedback: str | None = None
        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                raw = self._transport.complete(
                    messages=_with_feedback(messages, feedback),
                    response_model=response_model,
                )
                return parse(response_model.model_validate_json(raw))
            except ParameterError:
                raise
            except TransientInferenceError as exc:
                # Keep the latest validation feedback: the failed request never reached the model.
                self._wait_before_transient_retry(purpose, attempt, exc)
            except (InvalidInferenceResponse, ValidationError, ValueError) as exc:
                _raise_if_invalid_output_exhausted(purpose, attempt)
                feedback = _validation_feedback(exc)
        raise InternalError(f"{purpose} retry loop ended without a result")

    def _wait_before_transient_retry(self, purpose: str, attempt: int, exc: TransientInferenceError) -> None:
        if attempt == MAX_REQUEST_ATTEMPTS:
            raise GenerationError(f"{purpose} failed after {MAX_REQUEST_ATTEMPTS} attempts") from exc
        self._sleep(_retry_delay(attempt, exc))

    def _repair_invalid_patterns(
        self,
        discovery_input: PlanDiscoveryInput,
        plan: PiiReplacementPlan,
    ) -> PiiReplacementPlan:
        dataframe = discovery_input.dataframe
        profiles = {profile.column_name: profile for profile in discovery_input.column_profiles}
        repaired_specs: list[PiiColumnPlan] = []
        for spec in plan.columns_to_replace:
            issue = column_pattern_issue(dataframe, spec)
            if issue is None:
                repaired_specs.append(spec)
                continue
            if column_pattern_syntax_issue(spec) is None:
                # The grammar is fine but too few values match, so the column mixes
                # formats. One pattern cannot describe it, and asking again would not help.
                logger.user.warning(
                    "Dropping an LLM-proposed pattern that does not cover the column's values",
                    extra={"column": spec.column_name},
                )
                repaired_specs.append(spec.model_copy(update={"pattern": None}))
                continue
            pattern = self._repair_pattern(dataframe, profiles[spec.column_name], spec, issue)
            repaired_specs.append(spec.model_copy(update={"pattern": pattern}))
        return plan.model_copy(update={"columns_to_replace": repaired_specs})

    def _repair_pattern(
        self,
        dataframe: pd.DataFrame,
        profile: ColumnProfile,
        spec: PiiColumnPlan,
        issue: str,
    ) -> str | None:
        """Return a repaired pattern, or ``None`` after repair attempts are exhausted.

        Transient failures that persist still raise ``GenerationError``; only
        invalid repair output drops the optional pattern.
        """

        def parse(response: _PatternRepairResponse) -> str:
            if not response.pattern.strip():
                raise ValueError("pattern must be non-empty")
            candidate = spec.model_copy(update={"pattern": response.pattern})
            if (next_issue := column_pattern_issue(dataframe, candidate)) is not None:
                raise ValueError(next_issue)
            return response.pattern

        try:
            return self._request_structured(
                purpose="PII pattern repair",
                messages=_pattern_repair_messages(profile, spec, issue),
                response_model=_PatternRepairResponse,
                parse=parse,
            )
        except _InvalidStructuredOutputError:
            logger.user.warning(
                "Dropping an invalid LLM-proposed pattern after repair attempts",
                extra={"column": spec.column_name, "attempts": MAX_REQUEST_ATTEMPTS},
            )
            return None
