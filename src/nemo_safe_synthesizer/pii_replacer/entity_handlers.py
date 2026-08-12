# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One handler per entity label: where per-entity behavior attaches.

``entities.py`` says what each label *is* — its channel, its pattern language,
the gates discovery holds it to. This module holds what each label *does*: read
a value as that entity, refuse a column that only looks like one, write a
synthetic stand-in for a standalone value, and (for persona-channel labels)
map a sampled persona onto a cell.

Handlers are deliberately thin. Discovery skips, standalone generation, and
persona writes all go through ``get_handler(label)``. Shared content gates
(``requires_value_match``, ``name_shape_gates``) live in
``skip_reason_named_column`` for ``DefaultHandler``; entity-specific gates
(phone messaging, DOB parseability, API-key shape, street house numbers,
``unique_identifier`` strong/weak tiers) live on the dedicated handler. Faker /
shape-preserving draws live on the handler (``DefaultHandler._faker_draw`` and
overrides).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import TYPE_CHECKING

import pandas as pd

from .detection.column_names import header_matches_patterns
from .detection.value_recognizers import (
    looks_like_sequential_integer_id,
    match_value_entity,
    sample_has_dominant_identifier_template,
    sample_looks_like_api_key,
    sample_looks_like_multi_person,
    sample_looks_like_org_name,
    sample_looks_like_street_address,
)
from .entities import Config, EntityHandler, EntitySpec, spec
from .patterns import (
    generate_from_pattern,
    matching_template,
    pattern_preserving_token,
    split_title,
    synth_card_value,
    try_strftime_formats,
)

if TYPE_CHECKING:
    from .replacement.scope import FakerLike

__all__ = [
    "CreditCardHandler",
    "DateOfBirthHandler",
    "DefaultHandler",
    "EntityHandler",
    "UniqueIdentifierHandler",
    "get_handler",
    "skip_reason_named_column",
]


def skip_reason_named_column(
    entity_spec: EntitySpec,
    series: pd.Series,
    value_entity: str | None,
    apply_path: str,
) -> str | None:
    """Return a skip reason for a name-matched column, or ``None`` to allocate it.

    Shared gates used by ``DefaultHandler``: ``requires_value_match`` and
    ``name_shape_gates``. Entity-specific content checks live on dedicated
    handlers.

    Example:
        Header ``email`` whose values are not emails ->
        ``"values do not match that entity"``.

    Args:
        entity_spec: Registry entry for the name-matched entity.
        series: Column values used for content gates.
        value_entity: Dominant value-derived entity label, or ``None``.
        apply_path: Resolved apply path (``persona`` or ``standalone_map``).

    Returns:
        Human-readable skip reason, or ``None`` when the column may be allocated.
    """
    label = entity_spec.label
    if entity_spec.requires_value_match and value_entity != label:
        return "values do not match that entity"
    if entity_spec.name_shape_gates:
        if sample_looks_like_multi_person(series):
            return (
                "looks like multi-person values (delimiters such as 'and', '/', '&'); "
                "not auto-assigned — pre-split or hand-plan"
            )
        if sample_looks_like_org_name(series):
            return "values look like organizations, not people"
    return None


@dataclass(frozen=True)
class DefaultHandler:
    """Default entity behavior delegated to shared detection and replacement functions.

    A value reads as the entity when the value recognizers say so, a column is
    refused for the reason the discovery gates give, and a replacement is written
    in the column's format when one describes the value, or drawn from Faker when
    none does. Persona-channel writes default to unused (``None``).
    """

    label: str
    """Engine entity label this handler speaks for."""

    def match_value(self, value: object, *, phone_min_digits: int = 10) -> str | None:
        """Return this label when value recognizers read ``value`` as it.

        Args:
            value: Cell value to classify.
            phone_min_digits: Minimum digit count for phone matches.

        Returns:
            ``self.label`` when the value matches, else ``None``.
        """
        return self.label if match_value_entity(value, phone_min_digits=phone_min_digits) == self.label else None

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        """Return the discovery gate's reason for refusing this column.

        Args:
            series: Column values used for content gates.
            value_entity: Dominant value-derived entity label, or ``None``.
            apply_path: Resolved apply path (``persona`` or ``standalone_map``).
            column_name: Optional header; unused by the default shared gates.
            cfg: Optional engine configuration; unused by the default shared gates.

        Returns:
            Human-readable skip reason, or ``None`` to allocate the column.
        """
        entity_spec = spec(self.label)
        if entity_spec is None:
            return None
        return skip_reason_named_column(entity_spec, series, value_entity, apply_path)

    def generate(
        self,
        original: str,
        fake: FakerLike,
        *,
        patterns: Sequence[str] | None = None,
        rng: Random | None = None,
    ) -> str | None:
        """Return a replacement in the column's format, or a Faker draw when none is given.

        Example:
            With ``patterns=["pmc-[68]####"]`` and original ``"pmc-6123"`` -> a fresh
            value in that template; without patterns -> a Faker / shape-preserving draw.

        Args:
            original: Original cell value to replace.
            fake: Faker instance or random source for draws.
            patterns: Optional value templates or strftime formats for the column.
            rng: Optional random source; defaults to ``fake.random``.

        Returns:
            Synthetic replacement string, or ``None`` when none can be generated.
        """
        if patterns:
            return generate_from_pattern(matching_template(original, patterns), self._rng(fake, rng))
        return self._faker_draw(original, fake)

    def persona_value(
        self,
        original: str,
        persona: Mapping[str, object],
        *,
        patterns: Sequence[str] | None = None,
        originals: Mapping[str, object] | None = None,
        fake: FakerLike | None = None,
    ) -> str | None:
        """Return a persona-sourced replacement, or ``None`` when this label has none.

        Persona-channel handlers override this; standalone labels leave cells alone.
        """
        return None

    def _faker_draw(self, original: str, fake: FakerLike) -> str:
        """Return one Faker (or shape-preserving) draw for this handler's label."""
        from .detection import API_PREFIXES, UUID_RE

        rng = fake.random
        match self.label:
            case "credit_debit_card":
                return fake.credit_card_number()
            case "ipv4":
                return fake.ipv4()
            case "ipv6":
                return fake.ipv6()
            case "unique_identifier":
                if UUID_RE.match(original.strip()):
                    return str(fake.uuid4())
                return pattern_preserving_token(original, rng)
            case "api_key":
                for pfx in API_PREFIXES:
                    if original.startswith(pfx):
                        return pfx + pattern_preserving_token(original[len(pfx) :], rng)
                return pattern_preserving_token(original, rng)
            # Persona-sourced entities listed under standalone still get real Faker draws
            # (independent of any persona); pattern_preserving_token is only for opaque tokens.
            case "first_name":
                return fake.first_name()
            case "last_name":
                return fake.last_name()
            case "middle_name":
                return fake.first_name()
            case "full_name":
                return f"{fake.first_name()} {fake.last_name()}"
            case "email":
                return fake.email()
            case "phone_number":
                return fake.phone_number()
            case "ssn":
                return fake.ssn()
            case "street_address":
                return fake.street_address()
            case "city":
                return fake.city()
            case "state":
                return fake.state_abbr()
            case "zipcode":
                return fake.postcode()
            case "national_id":
                return fake.ssn()
            case _:
                return pattern_preserving_token(original, rng)

    def plan_pattern_rejection(self, column_name: str) -> str | None:
        """Return why plan ``patterns`` are illegal for this entity, or ``None`` if allowed."""
        return None

    @staticmethod
    def _rng(fake: FakerLike, rng: Random | None) -> Random:
        return fake.random if rng is None else rng

    @staticmethod
    def _as_str(value: object | None) -> str | None:
        return None if value is None else str(value)


class UniqueIdentifierHandler(DefaultHandler):
    """Unique-identifier handler: sequential skip + weak-name template gate.

    ``EntitySpec.strong_name_patterns`` vs the rest of ``name_patterns`` is the
    product rule; this handler enforces it. Strong headers (``patient_id``,
    ``uuid``, …) only refuse dense sequential integers. Weak leftovers
    (``valid``, ``userid``, …) also need a dominant identifier template.
    """

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        from ..errors import InternalError

        if apply_path != "standalone_map":
            return None
        if looks_like_sequential_integer_id(series):
            return "looks like a sequential integer id (1, 2, 3, …); not treated as a unique identifier"
        entity_spec = spec(self.label)
        if entity_spec is None:
            raise InternalError(
                "Entity registry is missing unique_identifier; cannot apply strong/weak identifier name gates."
            )
        # Empty strong_name_patterns means no weak tier: skip the template gate.
        if not entity_spec.strong_name_patterns or column_name is None:
            return None
        if header_matches_patterns(column_name, entity_spec.strong_name_patterns):
            return None
        if cfg is None or not sample_has_dominant_identifier_template(series, cfg):
            return "values lack a dominant identifier template"
        return None


class CreditCardHandler(DefaultHandler):
    """Credit card handler that validates Luhn checksum on generated values."""

    def generate(
        self,
        original: str,
        fake: FakerLike,
        *,
        patterns: Sequence[str] | None = None,
        rng: Random | None = None,
    ) -> str | None:
        if patterns:
            return synth_card_value(matching_template(original, patterns), self._rng(fake, rng))
        return super().generate(original, fake, rng=rng)


class IpAddressHandler(DefaultHandler):
    """IP address handler: Faker emits a valid address; character templates are rejected."""

    def plan_pattern_rejection(self, column_name: str) -> str | None:
        return (
            f"column {column_name!r} sets patterns, but a {self.label} address is generated by a "
            "generator that guarantees a valid address. A template counts characters and cannot express "
            "the rules an address obeys, such as an octet of at most 255, so drop the patterns."
        )

    def generate(
        self,
        original: str,
        fake: FakerLike,
        *,
        patterns: Sequence[str] | None = None,
        rng: Random | None = None,
    ) -> str | None:
        # Always use Faker's IP generators; never character templates.
        return self._faker_draw(original, fake)


class DateOfBirthHandler(DefaultHandler):
    """Date-of-birth handler: parseable-date gate + date perturbation (not redraw).

    Example:
        ``"1985-03-15"`` with patterns ``["%Y-%m-%d"]`` -> e.g. ``"1986-03-15"``.
    """

    def match_value(self, value: object, *, phone_min_digits: int = 10) -> str | None:
        # No value alone says 'date of birth'; a parseable date under a birth-date
        # header is what discovery accepts, and what the gate below checks for.
        return self.label if match_value_entity(value, phone_min_digits=phone_min_digits) == "date" else None

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        if value_entity != "date":
            return "values are not parseable dates"
        return None

    def generate(
        self,
        original: str,
        fake: FakerLike,
        *,
        patterns: Sequence[str] | None = None,
        rng: Random | None = None,
    ) -> str | None:
        # Deferred: ``replacement.standalone`` imports this module for its generators.
        from .replacement.standalone import synth_dob_programmatic

        # The first format the column writes that parses this date; a date none
        # of them parses is read, and printed, in its own.
        fmt = try_strftime_formats(original.strip(), list(patterns or ()))
        return synth_dob_programmatic(original, self._rng(fake, rng), fmt=fmt)


class PhoneNumberHandler(DefaultHandler):
    """Phone handler: phone-specific value-mismatch messaging + PGM persona writes."""

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        entity_spec = spec(self.label)
        if entity_spec is not None and entity_spec.requires_value_match and value_entity != self.label:
            return "values do not look like phone numbers"
        return None

    def persona_value(
        self,
        original: str,
        persona: Mapping[str, object],
        *,
        patterns: Sequence[str] | None = None,
        originals: Mapping[str, object] | None = None,
        fake: FakerLike | None = None,
    ) -> str | None:
        # Only reached under the PGM backend, whose number is tied to the
        # persona's address; the other backends route phones standalone
        # (see effective_apply_path).
        from .replacement.personas import format_persona_phone

        number = self._as_str(persona.get("phone_number")) or (fake.phone_number() if fake else None)
        return format_persona_phone(number, original, patterns, fake) if number else None


class ApiKeyHandler(DefaultHandler):
    """API-key handler: refuse numeric or non-credential-like columns."""

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        base = super().skip_reason(series, value_entity, apply_path, column_name=column_name, cfg=cfg)
        if base is not None:
            return base
        if pd.api.types.is_numeric_dtype(series) or not sample_looks_like_api_key(series):
            return "content is numeric or not credential-like"
        return None


class StreetAddressHandler(DefaultHandler):
    """Street-address handler: house-number sample gate + persona street-line writes."""

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        base = super().skip_reason(series, value_entity, apply_path, column_name=column_name, cfg=cfg)
        if base is not None:
            return base
        if not sample_looks_like_street_address(series):
            return "values lack house numbers (street name only)"
        return None

    def persona_value(
        self,
        original: str,
        persona: Mapping[str, object],
        *,
        patterns: Sequence[str] | None = None,
        originals: Mapping[str, object] | None = None,
        fake: FakerLike | None = None,
    ) -> str | None:
        parts = [str(x) for x in (persona.get("street_number"), persona.get("street_name")) if x not in (None, "")]
        new_street = " ".join(parts)
        if not new_street:
            return None
        # Preserve city/state/zip context: replace only the street line (before first comma).
        if "," in original:
            return new_street + "," + original.split(",", 1)[1]
        return new_street


class NamePartHandler(DefaultHandler):
    """first_name / last_name / middle_name: pattern write, then persona field fallback."""

    def persona_value(
        self,
        original: str,
        persona: Mapping[str, object],
        *,
        patterns: Sequence[str] | None = None,
        originals: Mapping[str, object] | None = None,
        fake: FakerLike | None = None,
    ) -> str | None:
        from .replacement.personas import persona_written

        written = persona_written(
            self.label,
            original,
            persona,
            patterns,
            originals,
            fake.random if fake else None,
        )
        if written is not None:
            return written
        return self._as_str(persona.get(self.label))


class FullNameHandler(DefaultHandler):
    """full_name: pattern write, then first+last with preserved title."""

    def persona_value(
        self,
        original: str,
        persona: Mapping[str, object],
        *,
        patterns: Sequence[str] | None = None,
        originals: Mapping[str, object] | None = None,
        fake: FakerLike | None = None,
    ) -> str | None:
        from .replacement.personas import persona_written

        written = persona_written(
            self.label,
            original,
            persona,
            patterns,
            originals,
            fake.random if fake else None,
        )
        if written is not None:
            return written
        title, _ = split_title(original)
        full = f"{persona.get('first_name', '')} {persona.get('last_name', '')}".strip()
        if not full:
            return None
        return f"{title} {full}" if title else full


class EmailHandler(DefaultHandler):
    """email: pattern/local-part write, then persona email or Faker fallback."""

    def persona_value(
        self,
        original: str,
        persona: Mapping[str, object],
        *,
        patterns: Sequence[str] | None = None,
        originals: Mapping[str, object] | None = None,
        fake: FakerLike | None = None,
    ) -> str | None:
        from .replacement.personas import persona_written

        written = persona_written(
            self.label,
            original,
            persona,
            patterns,
            originals,
            fake.random if fake else None,
        )
        if written is not None:
            return written
        # Only reached by a value that is no address at all, since one with a
        # domain is written from itself above.
        return self._as_str(persona.get("email_address")) or (fake.email() if fake else None)


# Every label with behavior of its own is listed, including the ones that have
# none yet: the point of the table is that there is a single line to change when
# one of them grows a rule. Anything unlisted takes the shared behavior.
_HANDLERS: dict[str, type[DefaultHandler]] = {
    "ssn": DefaultHandler,
    "national_id": DefaultHandler,
    "date_of_birth": DateOfBirthHandler,
    "credit_debit_card": CreditCardHandler,
    "unique_identifier": UniqueIdentifierHandler,
    "phone_number": PhoneNumberHandler,
    "ipv4": IpAddressHandler,
    "ipv6": IpAddressHandler,
    "api_key": ApiKeyHandler,
    "street_address": StreetAddressHandler,
    "first_name": NamePartHandler,
    "last_name": NamePartHandler,
    "middle_name": NamePartHandler,
    "full_name": FullNameHandler,
    "email": EmailHandler,
}


@lru_cache(maxsize=128)
def get_handler(label: str) -> EntityHandler:
    """Return the handler for ``label``.

    Args:
        label: Engine entity label (e.g. ``ssn``, ``date_of_birth``).

    Returns:
        Entity-specific handler when one is registered, else ``DefaultHandler``.
    """
    return _HANDLERS.get(label, DefaultHandler)(label)
