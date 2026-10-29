# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Faker-backed structured PII replacement generator adapter."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from random import Random
from typing import TYPE_CHECKING, ClassVar
from uuid import UUID

from faker import Faker
from stdnum.br import cpf
from stdnum.ca import sin
from stdnum.cn import ric
from stdnum.fr import nir
from stdnum.it import codicefiscale
from stdnum.kr import rrn
from stdnum.nl import bsn
from stdnum.pl import pesel
from stdnum.se import personnummer
from stdnum.us import ssn

from ....config.replace_pii import EntityType, PiiSamplerBackend, PiiSamplerConfig
from ....errors import GenerationError, InternalError
from ....observability import get_logger
from ...planning.patterns import render_character_mask, render_name_pattern
from ...sampler_catalog import PERSONA_SEX_VALUES
from ..birth_dates import random_birth_date, shift_birth_date
from ._common import (
    NAME_ENTITY_TYPES,
    default_email_pattern,
    dependency_values,
    email_domain,
    name_values,
    normalize_organization_domain,
    render_luhn_pattern,
    render_name,
    required_pattern_values,
    resolve_dependency_values,
    shape_preserving_value,
)

if TYPE_CHECKING:
    from ..generation import ReplacementGenerationRequest

__all__ = ["FakerReplacementGenerator"]

logger = get_logger(__name__)


class FakerReplacementGenerator:
    """Generate structured replacements using Faker.

    Args:
        sampler: Sampler configuration; its backend must be Faker.
        locale: Faker locale for generated values.

    """

    backend: ClassVar[PiiSamplerBackend] = PiiSamplerBackend.FAKER

    def __init__(self, *, sampler: PiiSamplerConfig, locale: str) -> None:
        backend = sampler.backend
        if backend is not self.backend:
            raise InternalError(
                f"{type(self).__name__} requires the {self.backend.value} sampler backend, got {backend.value}"
            )
        self._locale = locale
        self._warned_generation_fallbacks: set[str] = set()

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Generate a Faker-backed replacement for ``request``."""
        rng = Random(request.seed)
        entity_type = request.entity_type
        if entity_type is EntityType.PHONE_NUMBER and request.pattern is not None:
            return render_character_mask(request.pattern, rng)
        if entity_type is EntityType.DATE_OF_BIRTH:
            return self._birth_date(request, rng)
        if entity_type is EntityType.CREDIT_DEBIT_CARD and request.pattern is not None:
            return render_luhn_pattern(request.pattern, rng)
        if entity_type in {EntityType.API_KEY, EntityType.UNIQUE_IDENTIFIER} and (
            shaped := _shaped_identifier(request, rng)
        ):
            return shaped

        fake = self._faker(request.seed)
        dependencies = dependency_values(request.effective_dependency_tuple)
        if entity_type in NAME_ENTITY_TYPES:
            gender_value = dependencies.get(EntityType.GENDER)
            gender_values = (
                resolve_dependency_values(request, EntityType.GENDER, gender_value)
                if gender_value is not None
                else None
            )
            values = name_values(
                dependencies,
                _faker_name_values(fake, _select_dependency_value(gender_values, request.seed)),
                locale=self._locale,
                full_name_pattern=request.full_name_pattern,
            )
            return render_name(entity_type, request.pattern, rng, values, locale=self._locale)
        if entity_type is EntityType.EMAIL:
            return self._generate_email(request, fake, rng, dependencies)
        if entity_type is EntityType.PHONE_NUMBER:
            return fake.phone_number()
        if entity_type is EntityType.STREET_ADDRESS:
            # Some locales, such as en_GB, put a flat or unit on its own line.
            return fake.street_address().replace("\n", ", ")
        if entity_type is EntityType.NATIONAL_ID:
            return _national_id(request.original_value, self._locale, fake, rng)
        if entity_type is EntityType.CREDIT_DEBIT_CARD:
            return fake.credit_card_number()
        if entity_type in {EntityType.API_KEY, EntityType.UNIQUE_IDENTIFIER}:
            if entity_type is EntityType.UNIQUE_IDENTIFIER and _is_uuid(request.original_value):
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
            fake = Faker(self._locale)
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
        pattern = request.pattern or default_email_pattern(dependencies, locale=self._locale)
        if pattern is None:
            return fake.email()

        required = required_pattern_values(EntityType.EMAIL, pattern)
        values = name_values(
            dependencies,
            _faker_name_values(fake, None),
            locale=self._locale,
            full_name_pattern=request.full_name_pattern,
        )
        original_domain = email_domain(request.original_value)
        if original_domain is None and "domain" in required:
            self._warn_generation_fallback("unresolved_email_domain")
        values["domain"] = original_domain or fake.domain_name()
        organization = dependencies.get(EntityType.ORGANIZATION)
        organization_domain = normalize_organization_domain(organization or "")
        if not organization_domain and "organization" in required:
            self._warn_generation_fallback("unresolved_email_organization")
        values["organization"] = organization_domain or fake.domain_word()
        return render_name_pattern(EntityType.EMAIL, pattern, values, rng)

    def _birth_date(self, request: ReplacementGenerationRequest, rng: Random) -> str:
        shifted = shift_birth_date(request.original_value, request.pattern, rng, locale=self._locale)
        if shifted is not None:
            return shifted
        self._warn_generation_fallback("unparseable_birth_date")
        return random_birth_date(request.pattern, rng)

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


def _select_dependency_value(sampler_values: tuple[str, ...] | None, seed: int) -> str | None:
    if sampler_values is None:
        return None
    return sampler_values[seed % len(sampler_values)]


def _faker_first_name(fake: Faker, gender: str | None) -> str:
    # Nemotron-Personas falls back to Faker with its own sex value, such as "femme" in fr_FR.
    normalized = PERSONA_SEX_VALUES.get(gender.casefold(), "") if gender else ""
    if normalized == "female":
        return fake.first_name_female()
    if normalized == "male":
        return fake.first_name_male()
    return fake.first_name()


def _shaped_identifier(request: ReplacementGenerationRequest, rng: Random) -> str | None:
    """Return a masked or shape-preserving identifier, or ``None`` to generate one with Faker."""
    if request.pattern is not None:
        return render_character_mask(request.pattern, rng)
    if request.entity_type is EntityType.UNIQUE_IDENTIFIER and _is_uuid(request.original_value):
        return None
    shaped = shape_preserving_value(request.original_value, rng)
    return shaped if shaped != request.original_value else None


_NATIONAL_ID_CHECKS: Mapping[str, Callable[[str], bool]] = {
    "en_US": ssn.is_valid,
    "en_CA": sin.is_valid,
    "fr_FR": nir.is_valid,
    "it_IT": codicefiscale.is_valid,
    "ko_KR": rrn.is_valid,
    "nl_NL": bsn.is_valid,
    "pl_PL": pesel.is_valid,
    "pt_BR": cpf.is_valid,
    "sv_SE": personnummer.is_valid,
    "zh_CN": ric.is_valid,
}
"""python-stdnum check for each locale whose national ID Faker's ``ssn()`` generates.

Faker returns a US SSN for many other locales, such as ``de_DE`` and ``ja_JP``, so those are left out. Some
``en_CA``, ``zh_CN``, and ``ko_KR`` values Faker generates fail python-stdnum's check.
"""


def _national_id(original: str, locale: str, fake: Faker, rng: Random) -> str:
    """Return a national ID for ``original``: the locale's own ID when ``original`` is one, else the same layout.

    When ``original`` passes python-stdnum's check for the locale's national ID, Faker generates a new one,
    written in the original's layout when both have as many letters and digits. Any other value keeps its
    layout, with every letter and digit replaced, since national IDs include driver's licenses, tax IDs, and
    other countries' IDs.

    Example:
        In ``en_US``, ``"123-45-6789"`` becomes another SSN and ``"123456789"`` another SSN without dashes.
        ``"D1234567"`` becomes a value such as ``"Q8051937"`` in every locale.
    """
    is_valid = _NATIONAL_ID_CHECKS.get(locale)
    if is_valid is not None and _passes(is_valid, original):
        generated = _in_layout(fake.ssn(), original)
        if generated.casefold() != original.casefold():
            return generated
    return shape_preserving_value(original, rng)


def _passes(is_valid: Callable[[str], bool], value: str) -> bool:
    # cn.ric.is_valid raises KeyError for some unknown region codes instead of returning False.
    try:
        return is_valid(value)
    except Exception:
        return False


def _in_layout(generated: str, original: str) -> str:
    """Return ``generated``'s letters and digits in ``original``'s layout, or ``generated`` if their counts differ.

    Example:
        ``_in_layout("258 032 739", "317-280-675")`` is ``"258-032-739"``.
    """
    characters = [character for character in generated if character.isalnum()]
    if len(characters) != sum(character.isalnum() for character in original):
        return generated
    fill = iter(characters)
    return "".join(next(fill) if character.isalnum() else character for character in original)


def _is_uuid(value: str) -> bool:
    """Return whether ``value`` is a UUID written in its canonical hyphenated form, in either letter case."""
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return str(parsed) == value.casefold()
