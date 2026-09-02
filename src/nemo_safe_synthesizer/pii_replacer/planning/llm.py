# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenAI-compatible two-pass LLM enhancement for PII replacement plans."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
import json
import os
from typing import Protocol, TypeVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from typing_extensions import Self

from ...config.replace_pii import (
    ALLOWED_DEPENDS_ON,
    ENTITIES,
    ENTITY_BY_TYPE,
    EntityType,
    LLMConfig,
    PiiColumnPlan,
    PiiReplacementPlan,
    can_condition,
    is_columns_to_replace_type,
)
from ...defaults import DEFAULT_NSS_INFERENCE_ENDPOINT, DEFAULT_NSS_INFERENCE_MODEL
from ...errors import GenerationError, ParameterError
from ...observability import get_logger
from .resolver import ColumnProfile, PlanDiscoveryInput, PlanEnhancer
from .validation import _cycle_columns, _iter_pattern_issues

__all__ = [
    "InferenceSettings",
    "LLMPlanEnhancer",
    "OpenAICompatibleTransport",
    "resolve_inference_settings",
]

MAX_ASSESSMENT_PROFILES = 32
MAX_ASSESSMENT_PROFILE_BYTES = 48 * 1024
MAX_REQUEST_ATTEMPTS = 3
_CHAT_COMPLETIONS_PATH = "/chat/completions"
_TRANSIENT_STATUS_CODES = frozenset({408, 409, 425, 429})

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    """Resolved runtime settings for the OpenAI-compatible adapter."""

    endpoint_url: str
    model_id: str
    max_workers: int
    api_key: str | None = field(default=None, repr=False)


class _TransientInferenceError(RuntimeError):
    """Retryable inference transport failure without response content."""


class _InvalidInferenceResponse(RuntimeError):
    """Retryable malformed inference envelope without response content."""


class _StructuredResponse(BaseModel):
    """Strict base for LLM-authored response envelopes."""

    model_config = ConfigDict(extra="forbid")


class _AssessmentDisposition(StrEnum):
    replace = "replace"
    conditioner = "conditioner"
    ignore = "ignore"


class _ColumnAssessment(_StructuredResponse):
    column_name: str
    disposition: _AssessmentDisposition
    entity_type: EntityType | None
    pattern: str | None = None

    @model_validator(mode="after")
    def _validate_disposition(self) -> Self:
        if self.disposition is _AssessmentDisposition.replace:
            if self.entity_type is None or not is_columns_to_replace_type(self.entity_type):
                raise ValueError("replace assessments require a replaceable entity_type")
            return self
        if self.disposition is _AssessmentDisposition.conditioner:
            if self.entity_type is None or not can_condition(self.entity_type):
                raise ValueError("conditioner assessments require an entity_type that can condition")
            if self.pattern is not None:
                raise ValueError("conditioner assessments cannot include a pattern")
            return self
        if self.pattern is not None:
            raise ValueError("ignore assessments cannot include a pattern")
        return self


class _AssessmentResponse(_StructuredResponse):
    assessments: list[_ColumnAssessment]


class _SynthesisDependency(_StructuredResponse):
    column_name: str
    entity_type: EntityType | None = None


class _SynthesisColumn(_StructuredResponse):
    column_name: str
    entity_type: EntityType
    pattern: str | None = None
    depends_on: list[_SynthesisDependency] = Field(default_factory=list)


class _SynthesisResponse(_StructuredResponse):
    columns_to_replace: list[_SynthesisColumn]


class _PatternRepairResponse(_StructuredResponse):
    pattern: str


ResponseT = TypeVar("ResponseT", bound=_StructuredResponse)
ResponseValidator = Callable[[ResponseT], ResponseT]


class _LLMTransport(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_model: type[_StructuredResponse],
    ) -> str:
        """Return the assistant response text without logging it."""


def _nonblank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _validate_endpoint(endpoint_url: str) -> None:
    parsed = urlparse(endpoint_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ParameterError("The PII inference endpoint must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ParameterError("The PII inference endpoint URL must not contain credentials")


def _is_default_hosted_endpoint(endpoint_url: str) -> bool:
    return endpoint_url.rstrip("/") == DEFAULT_NSS_INFERENCE_ENDPOINT.rstrip("/")


def resolve_inference_settings(
    config: LLMConfig,
    *,
    endpoint_url: str | None = None,
    model_id: str | None = None,
    api_key: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> InferenceSettings:
    """Resolve CLI overrides, config, environment, and NSS defaults in order.

    The API key is deliberately absent from persisted configuration. It is
    accepted only as a runtime override or from ``NSS_INFERENCE_KEY``.
    """
    runtime_env = os.environ if environ is None else environ
    resolved_endpoint = (
        _nonblank(endpoint_url)
        or _nonblank(config.endpoint_url)
        or _nonblank(runtime_env.get("NSS_INFERENCE_ENDPOINT"))
        or DEFAULT_NSS_INFERENCE_ENDPOINT
    )
    resolved_model = (
        _nonblank(model_id)
        or _nonblank(config.model_id)
        or _nonblank(runtime_env.get("NSS_INFERENCE_MODEL"))
        or DEFAULT_NSS_INFERENCE_MODEL
    )
    resolved_key = _nonblank(api_key) or _nonblank(runtime_env.get("NSS_INFERENCE_KEY"))

    _validate_endpoint(resolved_endpoint)
    if _is_default_hosted_endpoint(resolved_endpoint) and resolved_key is None:
        raise ParameterError(
            "NSS_INFERENCE_KEY or --inference-api-key is required for the default hosted NVIDIA inference endpoint"
        )

    return InferenceSettings(
        endpoint_url=resolved_endpoint.rstrip("/"),
        model_id=resolved_model,
        api_key=resolved_key,
        max_workers=config.max_workers,
    )


class OpenAICompatibleTransport:
    """Minimal privacy-preserving OpenAI chat-completions transport."""

    def __init__(self, settings: InferenceSettings, *, timeout: float = 60.0) -> None:
        self._settings = settings
        self._timeout = timeout

    def complete(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        response_model: type[_StructuredResponse],
    ) -> str:
        """Return one structured assistant response.

        Errors intentionally exclude response bodies because they may contain
        prompts, samples, or model-authored sensitive text.
        """
        headers = {"Content-Type": "application/json"}
        if self._settings.api_key is not None:
            headers["Authorization"] = f"Bearer {self._settings.api_key}"
        payload = {
            "model": self._settings.model_id,
            "messages": list(messages),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            },
            "temperature": 0,
        }
        try:
            response = httpx.post(
                self._settings.endpoint_url + _CHAT_COMPLETIONS_PATH,
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except (httpx.NetworkError, httpx.TimeoutException) as exc:
            raise _TransientInferenceError("PII inference transport failed") from exc
        except httpx.HTTPError as exc:
            raise _TransientInferenceError("PII inference transport failed") from exc

        if response.status_code in {401, 403}:
            raise ParameterError(f"PII inference authentication or authorization failed (HTTP {response.status_code})")
        if response.status_code in _TRANSIENT_STATUS_CODES or response.status_code >= 500:
            raise _TransientInferenceError(f"PII inference service returned HTTP {response.status_code}")
        if response.status_code >= 400:
            raise ParameterError(f"PII inference request was rejected (HTTP {response.status_code})")

        try:
            envelope = response.json()
            content = envelope["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise _InvalidInferenceResponse("PII inference service returned an invalid response envelope") from exc
        if not isinstance(content, str):
            raise _InvalidInferenceResponse("PII inference service returned non-text response content")
        return content


def _profile_payload(profile: ColumnProfile) -> dict[str, object]:
    return {
        "column_name": profile.column_name,
        "dtype": profile.dtype,
        "non_null_count": profile.non_null_count,
        "unique_count": profile.unique_count,
        "unique_ratio": profile.unique_ratio,
        "group_constancy": profile.group_constancy,
        "grain": profile.grain.value,
        "protected": profile.protected,
        "samples": list(profile.samples),
    }


def _json_bytes(value: object) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def _profile_batches(profiles: Sequence[ColumnProfile]) -> list[list[dict[str, object]]]:
    batches: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    for profile in profiles:
        payload = _profile_payload(profile)
        if _json_bytes([payload]) > MAX_ASSESSMENT_PROFILE_BYTES:
            raise ParameterError(f"Column profile for {profile.column_name!r} exceeds the 48 KiB LLM evidence limit")
        candidate = [*current, payload]
        if current and (
            len(candidate) > MAX_ASSESSMENT_PROFILES or _json_bytes(candidate) > MAX_ASSESSMENT_PROFILE_BYTES
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


def _dependency_catalog() -> dict[str, list[str]]:
    return {
        target.value: sorted(source.value for source in sources)
        for target, sources in ALLOWED_DEPENDS_ON.items()
    }


def _validation_feedback(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        details = exc.errors(include_input=False, include_url=False)[:5]
        rendered = "; ".join(
            f"{'.'.join(str(part) for part in item['loc']) or 'response'}: {item['msg']}" for item in details
        )
    else:
        rendered = str(exc)
    return rendered[:800]


def _assessment_messages(batch: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    system = (
        "Assess every submitted dataframe column for a PII replacement plan. "
        "Return exactly one assessment per column, using disposition replace, conditioner, or ignore. "
        "A replace assessment needs a replaceable entity type; a conditioner needs an entity type with "
        "can_condition=true; ignore may use null when no catalog entity applies. Optional patterns must describe "
        "the observed whole-column format. Do not omit, duplicate, or invent columns."
    )
    user = _compact_json({"entity_catalog": _entity_catalog(), "column_profiles": batch})
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _synthesis_messages(
    discovery_input: PlanDiscoveryInput,
    assessments: Sequence[_ColumnAssessment],
    baseline: PiiReplacementPlan,
) -> list[dict[str, str]]:
    system = (
        "Synthesize the final PII replacement columns and dependency edges. You may add or remove baseline columns "
        "and revise entity types, patterns, and dependencies. Use only submitted dataframe columns. Never replace a "
        "protected column. The scope is supplied by NSS and must not appear in your response. Dependencies must follow "
        "the permitted relationship catalog and form an acyclic graph."
    )
    user = _compact_json(
        {
            "scope": discovery_input.scope.value,
            "protected_columns": sorted(discovery_input.protected_columns),
            "assessments": [assessment.model_dump(mode="json") for assessment in assessments],
            "heuristic_baseline": baseline.model_dump(mode="json"),
            "entity_catalog": _entity_catalog(),
            "permitted_dependencies": _dependency_catalog(),
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
        "Repair only the optional whole-column pattern. Return one non-empty pattern compatible with the supplied "
        "entity pattern syntax and samples. Do not change the column or entity type."
    )
    user = _compact_json(
        {
            "column_profile": _profile_payload(profile),
            "entity_type": spec.entity_type.value,
            "pattern_syntax": pattern_syntax.name.lower(),
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
    """Enhance a heuristic plan with bounded assessment and synthesis passes."""

    def __init__(
        self,
        config: LLMConfig,
        *,
        endpoint_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        transport: _LLMTransport | None = None,
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
        """Return an LLM-authored plan or fail without heuristic fallback."""
        assessments = self._assess_columns(discovery_input.column_profiles)
        plan: PiiReplacementPlan | None = None

        def validate_synthesis(response: _SynthesisResponse) -> _SynthesisResponse:
            nonlocal plan
            plan = self._validate_synthesis(discovery_input, response)
            return response

        self._request_structured(
            purpose="PII plan synthesis",
            messages=_synthesis_messages(discovery_input, assessments, baseline),
            response_model=_SynthesisResponse,
            validate=validate_synthesis,
        )
        assert plan is not None
        return self._repair_invalid_patterns(discovery_input, plan)

    def _assess_columns(self, profiles: Sequence[ColumnProfile]) -> list[_ColumnAssessment]:
        batches = _profile_batches(profiles)
        if not batches:
            return []

        def assess(batch: list[dict[str, object]]) -> list[_ColumnAssessment]:
            expected = [str(profile["column_name"]) for profile in batch]

            def validate(response: _AssessmentResponse) -> _AssessmentResponse:
                actual = [assessment.column_name for assessment in response.assessments]
                if len(actual) != len(expected) or set(actual) != set(expected):
                    raise ValueError("assessments must contain every submitted column exactly once")
                by_name = {assessment.column_name: assessment for assessment in response.assessments}
                response.assessments = [by_name[name] for name in expected]
                return response

            response = self._request_structured(
                purpose="PII column assessment",
                messages=_assessment_messages(batch),
                response_model=_AssessmentResponse,
                validate=validate,
            )
            return response.assessments

        if len(batches) == 1:
            return assess(batches[0])
        worker_count = min(self.settings.max_workers, len(batches))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(assess, batches))
        return [assessment for batch_result in results for assessment in batch_result]

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
            except _TransientInferenceError as exc:
                if attempt == MAX_REQUEST_ATTEMPTS:
                    raise GenerationError(f"{purpose} failed after {MAX_REQUEST_ATTEMPTS} attempts") from exc
                feedback = None
            except (_InvalidInferenceResponse, ValidationError, ValueError) as exc:
                if attempt == MAX_REQUEST_ATTEMPTS:
                    raise GenerationError(
                        f"{purpose} returned invalid structured output after {MAX_REQUEST_ATTEMPTS} attempts"
                    ) from None
                feedback = _validation_feedback(exc)
        raise AssertionError("unreachable")

    @staticmethod
    def _validate_synthesis(
        discovery_input: PlanDiscoveryInput,
        response: _SynthesisResponse,
    ) -> PiiReplacementPlan:
        try:
            plan = PiiReplacementPlan(
                scope=discovery_input.scope,
                columns_to_replace=[
                    PiiColumnPlan.model_validate(spec.model_dump()) for spec in response.columns_to_replace
                ],
            )
        except (ParameterError, ValidationError) as exc:
            raise ValueError("synthesized columns do not form a valid replacement plan") from exc
        available = {profile.column_name for profile in discovery_input.column_profiles}
        for spec in plan.columns_to_replace:
            if spec.column_name not in available:
                raise ValueError("synthesis invented a dataframe column")
            if spec.column_name in discovery_input.protected_columns:
                raise ValueError("synthesis attempted to replace a protected structural column")
            if any(dependency.column_name not in available for dependency in spec.depends_on):
                raise ValueError("synthesis invented a dependency column")
        if _cycle_columns(plan):
            raise ValueError("synthesis returned cyclic replacement dependencies")
        return plan

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
                candidate = next(item for item in candidate_plan.columns_to_replace if item.column_name == spec.column_name)
                candidate.pattern = response.pattern
                next_issue = self._pattern_issue(discovery_input, candidate_plan, spec.column_name)
                if next_issue is None:
                    return response.pattern
                feedback = next_issue
            except ParameterError:
                raise
            except _TransientInferenceError as exc:
                if attempt == MAX_REQUEST_ATTEMPTS:
                    raise GenerationError(
                        f"PII pattern repair failed after {MAX_REQUEST_ATTEMPTS} attempts"
                    ) from exc
                feedback = None
            except (_InvalidInferenceResponse, ValidationError, ValueError) as exc:
                feedback = _validation_feedback(exc)

        logger.user.warning(
            "Dropping an invalid LLM-proposed pattern after repair attempts",
            extra={"column": spec.column_name, "attempts": MAX_REQUEST_ATTEMPTS},
        )
        return None
