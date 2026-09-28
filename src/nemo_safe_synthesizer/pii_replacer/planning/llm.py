# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible two-pass LLM enhancement for PII replacement plans."""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

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
from .validation import column_pattern_issue

__all__ = [
    "LLMPlanEnhancer",
]

MAX_CLASSIFICATION_PROFILES = 32
MAX_CLASSIFICATION_PROFILE_BYTES = 48 * 1024
MAX_REQUEST_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 30.0

logger = get_logger(__name__)


class _StructuredResponse(BaseModel):
    """Strict base for LLM-authored response envelopes."""

    model_config = ConfigDict(extra="forbid")


class _ClassificationResponse(_StructuredResponse):
    classifications: list[ColumnClassification]


class _DependencySelectionResponse(_StructuredResponse):
    selected_dependency_ids: list[str]


class _PatternRepairResponse(_StructuredResponse):
    pattern: str


class _InvalidStructuredOutputError(GenerationError):
    """Structured output stayed invalid after every allowed attempt."""


ResponseT = TypeVar("ResponseT", bound=_StructuredResponse)
ResultT = TypeVar("ResultT")


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


def _profile_batches(profiles: Sequence[ColumnProfile]) -> list[list[dict[str, object]]]:
    batches: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for profile in profiles:
        payload = _profile_payload(profile)
        if _json_bytes([payload]) > MAX_CLASSIFICATION_PROFILE_BYTES:
            raise ParameterError(f"Column profile for {profile.column_name!r} exceeds the 48 KiB LLM evidence limit")
        candidate = [*current, payload]
        if current and (
            len(candidate) > MAX_CLASSIFICATION_PROFILES or _json_bytes(candidate) > MAX_CLASSIFICATION_PROFILE_BYTES
        ):
            batches.append(current)
            current = [payload]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _entity_catalog() -> list[dict[str, object]]:
    return [
        {
            "entity_type": entity.entity_type.value,
            "action": entity.action.name.lower(),
            "can_condition": entity.can_condition,
            "pattern_syntax": entity.pattern_syntax.name.lower() if entity.pattern_syntax is not None else None,
        }
        for entity in ENTITIES
    ]


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
        "Classify every submitted dataframe column by its semantic entity type and, when useful, propose an "
        "optional whole-column replacement pattern. Return exactly one classification for every submitted column, "
        "in the same order. entity_type must be one of the values in entity_catalog, or null when no catalog entity "
        "accurately describes the column. Classify semantic meaning only. Do not decide whether a column should be "
        "replaced, used as a conditioner, or ignored; NSS derives those roles deterministically. pattern must be null "
        "when entity_type is null, the selected entity has no pattern_syntax, or the column is listed in "
        "discovery_context.protected_columns. Protected columns must still receive a semantic classification, but NSS "
        "will not replace them. The grouping column is not automatically protected and must still be classified. "
        "Otherwise, "
        "emit a pattern only when the observed values have a consistent format worth preserving; do not emit a "
        "redundant pattern that adds no useful formatting information beyond the entity type. A non-null pattern must "
        "follow exactly the grammar named by the entity's pattern_syntax and describe the complete cell value. "
        "Patterns are not regular expressions. heuristic_classifications is a sparse projection containing only "
        "columns that the heuristic baseline selected for replacement; treat it as fallible prior evidence, not an "
        "ignore decision for absent columns. Do not omit, duplicate, or invent columns."
    )
    user = _compact_json(
        {
            "discovery_context": {
                "group_column": discovery_input.group_column,
                "protected_columns": sorted(discovery_input.protected_columns),
            },
            "entity_catalog": _entity_catalog(),
            "pattern_grammars": pattern_grammar_catalog(),
            "heuristic_classifications": _heuristic_classifications_payload(baseline, batch),
            "column_profiles": batch,
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _dependency_candidate_id(index: int) -> str:
    return f"dependency_{index}"


def _dependency_candidate_payload(
    index: int,
    candidate: DependencyCandidate,
    *,
    entity_types: Mapping[str, EntityType],
    selected_by_heuristic: bool,
) -> dict[str, str | bool]:
    return {
        "id": _dependency_candidate_id(index),
        "target_column": candidate.target_column,
        "target_entity_type": entity_types[candidate.target_column].value,
        "source_column": candidate.source_column,
        "source_entity_type": entity_types[candidate.source_column].value,
        "selected_by_heuristic": selected_by_heuristic,
    }


def _dependency_selection_messages(
    candidates: Sequence[DependencyCandidate],
    baseline: PiiReplacementPlan,
    classifications: Sequence[ColumnClassification],
) -> list[dict[str, str]]:
    system = (
        "Select the contextually useful replacement dependencies from the submitted candidates. A selected dependency "
        "means that the target column's replacement should be conditioned on the source column. Every submitted "
        "candidate is permitted by the entity catalog, but permission alone does not make a dependency useful. Select "
        "a candidate only when the source column provides meaningful semantic context for generating the target "
        "column. selected_by_heuristic indicates that the heuristic baseline chose the same target/source edge; treat "
        "it as fallible prior evidence, not a requirement. Return only IDs from dependency_candidates. Do not invent "
        "IDs, replacement columns, entity types, patterns, or dependency relationships. Apply "
        "exclusive_dependency_groups independently to every target column: within each outer family, the selected "
        "source_entity_types for one target may intersect at most one inner group. Multiple source entity types from "
        "the same inner group are allowed. Do not select redundant dependencies."
    )
    heuristic_edges = _heuristic_dependency_edges(baseline)
    entity_types = {
        classification.column_name: classification.entity_type
        for classification in classifications
        if classification.entity_type is not None
    }
    user = _compact_json(
        {
            "dependency_candidates": [
                _dependency_candidate_payload(
                    index,
                    candidate,
                    entity_types=entity_types,
                    selected_by_heuristic=(candidate.target_column, candidate.source_column) in heuristic_edges,
                )
                for index, candidate in enumerate(candidates)
            ],
            "exclusive_dependency_groups": _exclusive_dependency_groups_payload(),
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _apply_dependency_selection(
    plan: PiiReplacementPlan,
    candidates_by_id: Mapping[str, DependencyCandidate],
    classifications: Sequence[ColumnClassification],
    selected_ids: Sequence[str],
) -> PiiReplacementPlan:
    """Apply selected candidate IDs, raising ``ValueError`` with specific feedback when invalid."""
    if duplicates := sorted(selected_id for selected_id, count in Counter(selected_ids).items() if count > 1):
        raise ValueError("selected_dependency_ids must not contain duplicates: " + _quoted(duplicates))
    if unknown := sorted(set(selected_ids) - set(candidates_by_id)):
        raise ValueError("selected_dependency_ids contains unknown IDs: " + _quoted(unknown))

    try:
        return apply_dependencies(
            plan,
            [candidates_by_id[selected_id] for selected_id in selected_ids],
            classifications=classifications,
        )
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
        "expressions. Do not change the column or entity type."
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
        candidates = derive_dependency_candidates(
            classifications,
            expected_columns=expected_columns,
            protected_columns=discovery_input.protected_columns,
        )
        if candidates:
            plan = self._select_dependencies(plan, candidates, baseline, classifications)
        return self._repair_invalid_patterns(discovery_input, plan)

    def _classify_columns(
        self,
        discovery_input: PlanDiscoveryInput,
        baseline: PiiReplacementPlan,
    ) -> list[ColumnClassification]:
        batches = _profile_batches(discovery_input.column_profiles)
        if not batches:
            return []
        classify = partial(self._classify_batch, discovery_input, baseline)
        if len(batches) == 1:
            return classify(batches[0])
        with ThreadPoolExecutor(max_workers=min(self.settings.max_workers, len(batches))) as executor:
            results = list(executor.map(classify, batches))
        return [classification for batch_result in results for classification in batch_result]

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
            if protected_with_pattern := sorted(
                item.column_name
                for item in response.classifications
                if item.pattern is not None and item.column_name in protected
            ):
                raise ValueError(
                    "protected columns cannot include replacement patterns: " + _quoted(protected_with_pattern)
                )
            by_name = {classification.column_name: classification for classification in response.classifications}
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
        candidates_by_id = {_dependency_candidate_id(index): candidate for index, candidate in enumerate(candidates)}
        return self._request_structured(
            purpose="PII dependency selection",
            messages=_dependency_selection_messages(candidates, baseline, classifications),
            response_model=_DependencySelectionResponse,
            parse=lambda response: _apply_dependency_selection(
                plan, candidates_by_id, classifications, response.selected_dependency_ids
            ),
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
        messages plus, after an invalid response, feedback describing only the
        latest rejection, so the prompt does not grow across attempts.

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
                self._wait_before_transient_retry(purpose, attempt, exc)
                feedback = None
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
