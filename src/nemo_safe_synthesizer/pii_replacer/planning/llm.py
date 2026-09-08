# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible two-pass LLM enhancement for PII replacement plans."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, ValidationError

from ...config.replace_pii import (
    ENTITIES,
    ENTITY_BY_TYPE,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
)
from ...errors import GenerationError, ParameterError
from ...observability import get_logger
from ..llm_client import (
    InvalidInferenceResponse,
    LLMTransport,
    OpenAICompatibleTransport,
    TransientInferenceError,
    resolve_inference_settings,
)
from .assembly import (
    ColumnClassification,
    DependencyCandidate,
    apply_dependencies,
    derive_dependency_candidates,
    plan_from_classifications,
)
from .patterns import pattern_grammar_catalog
from .resolver import ColumnProfile, PlanDiscoveryInput, PlanEnhancer
from .validation import _cycle_columns, _iter_pattern_issues

__all__ = [
    "LLMPlanEnhancer",
]

MAX_CLASSIFICATION_PROFILES = 32
MAX_CLASSIFICATION_PROFILE_BYTES = 48 * 1024
MAX_REQUEST_ATTEMPTS = 3

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


ResponseT = TypeVar("ResponseT", bound=_StructuredResponse)
ResponseValidator = Callable[[ResponseT], ResponseT]


def _profile_payload(profile: ColumnProfile) -> dict[str, object]:
    return {
        "column_name": profile.column_name,
        "dtype": profile.dtype,
        "non_null_count": profile.non_null_count,
        "unique_count": profile.unique_count,
        "unique_ratio": profile.unique_ratio,
        "samples": list(profile.samples),
    }


def _json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


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


def _compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


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


def _validation_feedback(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        details = exc.errors(include_input=False, include_url=False)[:5]
        rendered = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'response'}: {item['msg']}" for item in details
        )
    else:
        rendered = str(exc)
    return rendered[:800]


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
                "scope": discovery_input.scope.value,
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
    selected_by_heuristic: bool,
) -> dict[str, str | bool]:
    return {
        "id": _dependency_candidate_id(index),
        "target_column": candidate.target_column,
        "target_entity_type": candidate.target_entity_type.value,
        "source_column": candidate.source_column,
        "source_entity_type": candidate.source_entity_type.value,
        "selected_by_heuristic": selected_by_heuristic,
    }


def _dependency_selection_messages(
    candidates: Sequence[DependencyCandidate],
    baseline: PiiReplacementPlan,
) -> list[dict[str, str]]:
    system = (
        "Select the contextually useful replacement dependencies from the submitted candidates. A selected dependency "
        "means that the target column's replacement should be conditioned on the source column. Every submitted "
        "candidate is permitted by the entity catalog, but permission alone does not make a dependency useful. Select "
        "a candidate only when the source column provides meaningful semantic context for generating the target "
        "column. selected_by_heuristic indicates that the heuristic baseline chose the same target/source edge; treat "
        "it as fallible prior evidence, not a requirement. Return only IDs from dependency_candidates. Do not invent "
        "IDs, replacement columns, entity types, patterns, or dependency relationships. Do not select redundant or "
        "conflicting dependencies."
    )
    heuristic_edges = _heuristic_dependency_edges(baseline)
    user = _compact_json(
        {
            "dependency_candidates": [
                _dependency_candidate_payload(
                    index,
                    candidate,
                    selected_by_heuristic=(candidate.target_column, candidate.source_column) in heuristic_edges,
                )
                for index, candidate in enumerate(candidates)
            ]
        }
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _pattern_repair_messages(
    profile: ColumnProfile,
    spec: PiiColumnPlan,
    issue: str,
) -> list[dict[str, str]]:
    pattern_syntax = ENTITY_BY_TYPE[spec.entity_type].pattern_syntax
    assert pattern_syntax is not None
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


class LLMPlanEnhancer(PlanEnhancer):
    """Enhance a heuristic plan with classification and dependency-selection passes."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        endpoint_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        transport: LLMTransport | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.settings = resolve_inference_settings(
            config,
            endpoint_url=endpoint_url,
            model_id=model_id,
            api_key=api_key,
            environ=environ,
        )
        self._transport = transport or OpenAICompatibleTransport(self.settings)

    def enhance(
        self,
        discovery_input: PlanDiscoveryInput,
        baseline: PiiReplacementPlan,
    ) -> PiiReplacementPlan:
        """Return a semantically classified, deterministically assembled plan."""
        classifications = self._classify_columns(discovery_input, baseline)
        plan = plan_from_classifications(
            discovery_input.scope,
            classifications,
            protected_columns=discovery_input.protected_columns,
        )
        candidates = derive_dependency_candidates(plan, classifications)
        if candidates:
            plan = self._select_dependencies(plan, candidates, baseline)
        return self._repair_invalid_patterns(discovery_input, plan)

    def _classify_columns(
        self,
        discovery_input: PlanDiscoveryInput,
        baseline: PiiReplacementPlan,
    ) -> list[ColumnClassification]:
        batches = _profile_batches(discovery_input.column_profiles)
        if not batches:
            return []

        def classify(batch: list[dict[str, object]]) -> list[ColumnClassification]:
            expected = [str(profile["column_name"]) for profile in batch]
            protected = discovery_input.protected_columns.intersection(expected)

            def validate(response: _ClassificationResponse) -> _ClassificationResponse:
                actual = [classification.column_name for classification in response.classifications]
                if len(actual) != len(expected) or set(actual) != set(expected):
                    raise ValueError("classifications must contain every submitted column exactly once")
                by_name = {classification.column_name: classification for classification in response.classifications}
                response.classifications = [by_name[name] for name in expected]
                if any(item.pattern is not None and item.column_name in protected for item in response.classifications):
                    raise ValueError("protected columns cannot include replacement patterns")
                return response

            response = self._request_structured(
                purpose="PII column classification",
                messages=_classification_messages(discovery_input, batch, baseline),
                response_model=_ClassificationResponse,
                validate=validate,
            )
            return response.classifications

        if len(batches) == 1:
            return classify(batches[0])
        worker_count = min(self.settings.max_workers, len(batches))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(classify, batches))
        return [classification for batch_result in results for classification in batch_result]

    def _select_dependencies(
        self,
        plan: PiiReplacementPlan,
        candidates: Sequence[DependencyCandidate],
        baseline: PiiReplacementPlan,
    ) -> PiiReplacementPlan:
        selected_plan: PiiReplacementPlan | None = None

        def validate(response: _DependencySelectionResponse) -> _DependencySelectionResponse:
            nonlocal selected_plan
            selected_plan = self._apply_dependency_selection(plan, candidates, response)
            return response

        self._request_structured(
            purpose="PII dependency selection",
            messages=_dependency_selection_messages(candidates, baseline),
            response_model=_DependencySelectionResponse,
            validate=validate,
        )
        assert selected_plan is not None
        return selected_plan

    @staticmethod
    def _apply_dependency_selection(
        plan: PiiReplacementPlan,
        candidates: Sequence[DependencyCandidate],
        response: _DependencySelectionResponse,
    ) -> PiiReplacementPlan:
        selected_ids = response.selected_dependency_ids
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("selected_dependency_ids must not contain duplicates")

        by_id = {_dependency_candidate_id(index): candidate for index, candidate in enumerate(candidates)}
        if unknown := sorted(set(selected_ids) - set(by_id)):
            raise ValueError("selected_dependency_ids contains unknown IDs: " + ", ".join(unknown))

        try:
            selected_plan = apply_dependencies(plan, [by_id[selected_id] for selected_id in selected_ids])
        except (ParameterError, ValidationError) as exc:
            raise ValueError("selected dependencies do not form a valid replacement plan") from exc
        if _cycle_columns(selected_plan):
            raise ValueError("selected dependencies form a cycle")
        return selected_plan

    def _request_structured(
        self,
        *,
        purpose: str,
        messages: Sequence[Mapping[str, str]],
        response_model: type[ResponseT],
        validate: ResponseValidator[ResponseT] | None = None,
    ) -> ResponseT:
        feedback: str | None = None
        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                raw = self._transport.complete(
                    messages=_with_feedback(messages, feedback),
                    response_model=response_model,
                )
                response = response_model.model_validate_json(raw)
                return validate(response) if validate is not None else response
            except ParameterError:
                raise
            except TransientInferenceError as exc:
                if attempt == MAX_REQUEST_ATTEMPTS:
                    raise GenerationError(f"{purpose} failed after {MAX_REQUEST_ATTEMPTS} attempts") from exc
                feedback = None
            except (InvalidInferenceResponse, ValidationError, ValueError) as exc:
                if attempt == MAX_REQUEST_ATTEMPTS:
                    raise GenerationError(
                        f"{purpose} returned invalid structured output after {MAX_REQUEST_ATTEMPTS} attempts"
                    ) from None
                feedback = _validation_feedback(exc)
        raise AssertionError("unreachable")

    def _repair_invalid_patterns(
        self,
        discovery_input: PlanDiscoveryInput,
        plan: PiiReplacementPlan,
    ) -> PiiReplacementPlan:
        repaired = plan.model_copy(deep=True)
        profiles = {profile.column_name: profile for profile in discovery_input.column_profiles}
        for spec in repaired.columns_to_replace:
            issue = self._pattern_issue(discovery_input, repaired, spec.column_name)
            if issue is None:
                continue
            spec.pattern = self._repair_pattern(profiles[spec.column_name], spec, discovery_input, repaired, issue)
        return repaired

    @staticmethod
    def _pattern_issue(
        discovery_input: PlanDiscoveryInput,
        plan: PiiReplacementPlan,
        column_name: str,
    ) -> str | None:
        prefix = f"column {column_name!r}:"
        return next(
            (issue for issue in _iter_pattern_issues(discovery_input.dataframe, plan) if issue.startswith(prefix)),
            None,
        )

    def _repair_pattern(
        self,
        profile: ColumnProfile,
        spec: PiiColumnPlan,
        discovery_input: PlanDiscoveryInput,
        plan: PiiReplacementPlan,
        issue: str,
    ) -> str | None:
        messages = _pattern_repair_messages(profile, spec, issue)
        feedback: str | None = issue
        for attempt in range(1, MAX_REQUEST_ATTEMPTS + 1):
            try:
                raw = self._transport.complete(
                    messages=_with_feedback(messages, feedback),
                    response_model=_PatternRepairResponse,
                )
                response = _PatternRepairResponse.model_validate_json(raw)
                candidate_plan = plan.model_copy(deep=True)
                candidate = next(
                    item for item in candidate_plan.columns_to_replace if item.column_name == spec.column_name
                )
                candidate.pattern = response.pattern
                next_issue = self._pattern_issue(discovery_input, candidate_plan, spec.column_name)
                if next_issue is None:
                    return response.pattern
                feedback = next_issue
            except ParameterError:
                raise
            except TransientInferenceError as exc:
                if attempt == MAX_REQUEST_ATTEMPTS:
                    raise GenerationError(f"PII pattern repair failed after {MAX_REQUEST_ATTEMPTS} attempts") from exc
                feedback = None
            except (InvalidInferenceResponse, ValidationError, ValueError) as exc:
                feedback = _validation_feedback(exc)

        logger.user.warning(
            "Dropping an invalid LLM-proposed pattern after repair attempts",
            extra={"column": spec.column_name, "attempts": MAX_REQUEST_ATTEMPTS},
        )
        return None
