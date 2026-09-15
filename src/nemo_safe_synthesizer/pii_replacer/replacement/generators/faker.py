# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Faker-backed structured PII replacement generator adapter."""

from __future__ import annotations

from random import Random
from typing import TYPE_CHECKING, ClassVar

from faker import Faker

from ....config.replace_pii import EntityType, PiiReplacementSettings, PiiSamplerBackend, PiiSamplerConfig
from ....errors import GenerationError
from ....observability import get_logger
from ...planning.patterns import render_character_mask, render_name_pattern
from ._common import (
    _dependency_values,
    _email_domain,
    _looks_like_uuid,
    _name_values,
    _render_address,
    _render_luhn_pattern,
    _render_name,
    _required_pattern_values,
    _resolve_dependency_labels,
    _shape_preserving_value,
    _shift_birth_date,
    normalize_organization_domain,
)

if TYPE_CHECKING:
    from ..generation import ReplacementGenerationRequest

__all__ = ["FakerReplacementGenerator"]

logger = get_logger(__name__)


class FakerReplacementGenerator:
    """Generate structured replacements using Faker.

    Args:
        settings: Locale and seed configuration shared by replacement
            generators.
        sampler: Faker sampler configuration.

    """

    backend: ClassVar[PiiSamplerBackend] = PiiSamplerBackend.FAKER

    def __init__(self, *, settings: PiiReplacementSettings, sampler: PiiSamplerConfig) -> None:
        if sampler.backend is not self.backend:
            raise ValueError("FakerReplacementGenerator requires the faker sampler backend")
        self._settings = settings
        self._sampler = sampler
        self._warned_generation_fallbacks: set[str] = set()

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Generate a Faker-backed replacement for ``request``."""
        rng = Random(request.seed)
        entity_type = request.entity_type
        if entity_type is EntityType.PHONE_NUMBER and request.pattern is not None:
            return render_character_mask(request.pattern, rng)
        if entity_type is EntityType.DATE_OF_BIRTH:
            return _shift_birth_date(request.original_value, request.pattern, rng)
        if entity_type is EntityType.CREDIT_DEBIT_CARD and request.pattern is not None:
            return _render_luhn_pattern(request.pattern, rng)
        if entity_type in {EntityType.API_KEY, EntityType.UNIQUE_IDENTIFIER}:
            if request.pattern is not None:
                return render_character_mask(request.pattern, rng)
            if entity_type is not EntityType.UNIQUE_IDENTIFIER or not _looks_like_uuid(request.original_value):
                shaped = _shape_preserving_value(request.original_value, rng)
                if shaped != request.original_value:
                    return shaped

        fake = self._faker(request.seed)
        dependencies = _dependency_values(request.effective_dependency_tuple)
        if entity_type in {
            EntityType.FIRST_NAME,
            EntityType.MIDDLE_NAME,
            EntityType.LAST_NAME,
            EntityType.FULL_NAME,
        }:
            gender_value = dependencies.get(EntityType.GENDER)
            gender_labels = (
                _resolve_dependency_labels(request, EntityType.GENDER, gender_value)
                if gender_value is not None
                else None
            )
            values = _name_values(
                dependencies,
                _faker_name_values(fake, _select_dependency_label(gender_labels, request.seed)),
            )
            return _render_name(entity_type, request.pattern, rng, values)
        if entity_type is EntityType.EMAIL:
            return self._generate_email(request, fake, rng, dependencies)
        if entity_type is EntityType.PHONE_NUMBER:
            return fake.phone_number()
        if entity_type is EntityType.STREET_ADDRESS:
            return _render_address(fake.street_address(), dependencies)
        if entity_type in {EntityType.SSN, EntityType.NATIONAL_ID}:
            return fake.ssn()
        if entity_type is EntityType.CREDIT_DEBIT_CARD:
            return fake.credit_card_number()
        if entity_type in {EntityType.API_KEY, EntityType.UNIQUE_IDENTIFIER}:
            if entity_type is EntityType.UNIQUE_IDENTIFIER and _looks_like_uuid(request.original_value):
                return str(fake.uuid4())
            fallback = str(fake.uuid4()).replace("-", "")
            return fallback if entity_type is EntityType.UNIQUE_IDENTIFIER else f"key-{fallback}"
        if entity_type is EntityType.IPV4:
            return fake.ipv4()
        if entity_type is EntityType.IPV6:
            return fake.ipv6()
        raise GenerationError(f"entity type {entity_type.value!r} is not supported by structured replacement")

    def _faker(self, seed: int) -> Faker:
        try:
            fake = Faker(self._settings.locale)
        except Exception as exc:
            raise GenerationError("PII replacement could not initialize Faker for the configured locale") from exc
        fake.seed_instance(seed)
        return fake

    def _generate_email(
        self,
        request: ReplacementGenerationRequest,
        fake: Faker,
        rng: Random,
        dependencies: dict[EntityType, str],
    ) -> str:
        if request.pattern is None:
            return fake.email()

        required = _required_pattern_values(EntityType.EMAIL, request.pattern)
        values = _name_values(dependencies, _faker_name_values(fake, None))
        original_domain = _email_domain(request.original_value)
        if original_domain is None and "domain" in required:
            self._warn_generation_fallback("unresolved_email_domain")
        values["domain"] = original_domain or fake.domain_name()
        organization = dependencies.get(EntityType.ORGANIZATION)
        organization_domain = normalize_organization_domain(organization or "")
        if not organization_domain and "organization" in required:
            self._warn_generation_fallback("unresolved_email_organization")
        values["organization"] = organization_domain or fake.domain_word()
        return render_name_pattern(EntityType.EMAIL, request.pattern, values, rng)

    def _warn_generation_fallback(self, reason: str) -> None:
        if reason in self._warned_generation_fallbacks:
            return
        self._warned_generation_fallbacks.add(reason)
        logger.user.warning(
            "PII replacement used a deterministic generated fallback",
            extra={"reason": reason},
        )


def _faker_name_values(fake: Faker, gender: str | None) -> dict[str, str]:
    return {
        "first": _faker_first_name(fake, gender),
        "middle": _faker_first_name(fake, gender),
        "last": fake.last_name(),
    }


def _select_dependency_label(labels: tuple[str, ...] | None, seed: int) -> str | None:
    if labels is None:
        return None
    return labels[seed % len(labels)]


def _faker_first_name(fake: Faker, gender: str | None) -> str:
    normalized = gender.casefold() if gender else ""
    if normalized in {"female", "f", "woman", "girl"}:
        return fake.first_name_female()
    if normalized in {"male", "m", "man", "boy"}:
        return fake.first_name_male()
    return fake.first_name()
