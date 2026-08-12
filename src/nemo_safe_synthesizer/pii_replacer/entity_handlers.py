# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One handler per entity label: where per-entity behavior attaches.

``entities.py`` says what each label *is* — its channel, its pattern language,
the gates discovery holds it to. This module holds what each label *does*: read
a value as that entity, refuse a column that only looks like one, and write a
synthetic stand-in for a value.

Handlers are deliberately thin. Discovery skips and standalone generation both
go through ``get_handler(label)``. Shared content gates still live in
``detection.persona_grouping.skip_reason_named_column`` (``DefaultHandler.skip_reason``
delegates there); Faker / shape-preserving draws live on the handler
(``DefaultHandler._faker_draw`` and overrides). Call sites should not keep a
parallel ``match entity`` table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from random import Random
from typing import TYPE_CHECKING

import pandas as pd

from .detection.persona_grouping import skip_reason_named_column
from .detection.value_recognizers import match_value_entity
from .entities import EntityHandler, spec
from .patterns import (
    generate_from_pattern,
    matching_template,
    pattern_preserving_token,
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
    "get_handler",
]


@dataclass(frozen=True)
class DefaultHandler:
    """Default entity behavior delegated to shared detection and replacement functions.

    A value reads as the entity when the value recognizers say so, a column is
    refused for the reason the discovery gates give, and a replacement is written
    in the column's format when one describes the value, or drawn from Faker when
    none does.
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

    def skip_reason(self, series: pd.Series, value_entity: str | None, apply_path: str) -> str | None:
        """Return the discovery gate's reason for refusing this column.

        Args:
            series: Column values used for content gates.
            value_entity: Dominant value-derived entity label, or ``None``.
            apply_path: Resolved apply path (``persona`` or ``standalone_map``).

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
    """Date-of-birth handler that perturbs dates rather than redrawing them.

    Example:
        ``"1985-03-15"`` with patterns ``["%Y-%m-%d"]`` -> e.g. ``"1986-03-15"``.
    """

    def match_value(self, value: object, *, phone_min_digits: int = 10) -> str | None:
        # No value alone says 'date of birth'; a parseable date under a birth-date
        # header is what discovery accepts, and what the gate below checks for.
        return self.label if match_value_entity(value, phone_min_digits=phone_min_digits) == "date" else None

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


# Every label with behavior of its own is listed, including the ones that have
# none yet: the point of the table is that there is a single line to change when
# one of them grows a rule. Anything unlisted takes the shared behavior.
_HANDLERS: dict[str, type[DefaultHandler]] = {
    "ssn": DefaultHandler,
    "national_id": DefaultHandler,
    "date_of_birth": DateOfBirthHandler,
    "credit_debit_card": CreditCardHandler,
    "unique_identifier": DefaultHandler,
    "phone_number": DefaultHandler,
    "ipv4": IpAddressHandler,
    "ipv6": IpAddressHandler,
    "api_key": DefaultHandler,
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
