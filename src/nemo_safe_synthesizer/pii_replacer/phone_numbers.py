# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phone number helpers backed by libphonenumber, shared by PII replacement and its evaluation."""

from __future__ import annotations

import re

import phonenumbers

__all__ = ["country_code_prefix", "phone_identity", "phone_in_layout", "phone_region"]

_INTERNATIONAL_PREFIX = re.compile(r"[\s(]*(?:\+|00)\s*(?=[0-9])")
"""The ``+`` or ``00`` that starts an international phone number, before its country code."""


def phone_region(locale: str) -> str | None:
    """Return the libphonenumber region of ``locale``'s country, or ``None`` if it names no supported one.

    The region is used to read phone numbers written without a country code.

    Example:
        ``phone_region("en_US")`` is ``"US"``, ``phone_region("fr_CA")`` is ``"CA"``, and ``phone_region("en")``
        is ``None``.
    """
    _, _, country = locale.replace("-", "_").partition("_")
    region = country.upper()
    return region if region in phonenumbers.SUPPORTED_REGIONS else None


def phone_identity(text: str, region: str | None) -> str | None:
    """Return the E.164 form of the phone number ``text``, with its extension, or ``None`` if it is not one.

    Two ways of writing one number give the same identity. ``region`` reads numbers written without a
    country code; with ``None``, only numbers that start with ``+`` have an identity.

    Example:
        With region ``"US"``, ``"(555) 123-4567"``, ``"555.123.4567"``, and ``"+1 555 123 4567"`` all give
        ``"+15551234567"``, and ``"555-123-4567 x89"`` gives ``"+15551234567;ext=89"``.
    """
    try:
        number = phonenumbers.parse(text, region)
    except phonenumbers.NumberParseException:
        return None
    if not phonenumbers.is_possible_number(number):
        return None
    e164 = phonenumbers.format_number(number, phonenumbers.PhoneNumberFormat.E164)
    return f"{e164};ext={number.extension}" if number.extension else e164


def phone_in_layout(occurrence: str, replacement: str) -> str:
    """Write ``replacement``'s digits in ``occurrence``'s layout, or return ``replacement`` if they do not fit.

    One original number written in several ways shares a replacement, and each occurrence keeps its own
    layout, including its country code or the lack of one. The digits after any country code must be as
    many in both, so a national trunk prefix such as the UK's leading ``0`` makes them not fit.

    Example:
        ``phone_in_layout("555.123.4567", "(812) 904-3321")`` is ``"812.904.3321"``,
        ``phone_in_layout("+1 555 123 4567", "(812) 904-3321")`` is ``"+1 812 904 3321"``, and
        ``phone_in_layout("020 7946 0958", "+44 81 2604 9135")`` is ``"+44 81 2604 9135"``.
    """
    prefix = country_code_prefix(occurrence)
    rest = occurrence[len(prefix) :]
    digits = [character for character in replacement[len(country_code_prefix(replacement)) :] if character.isdigit()]
    if len(digits) != sum(character.isdigit() for character in rest):
        return replacement
    fill = iter(digits)
    return prefix + "".join(next(fill) if character.isdigit() else character for character in rest)


def country_code_prefix(phone: str) -> str:
    """Return the start of ``phone`` through its country code, or ``""`` if it has none.

    The code is read with libphonenumber, so it is found even without a separator after it, as in
    ``"+442079460958"``. Only the code needs to be known; the rest of the number need not be valid.

    Example:
        ``country_code_prefix("+442079460958")`` is ``"+44"``, ``country_code_prefix("001-555")`` is
        ``"001"``, and ``country_code_prefix("+999 123")`` is ``""``, since 999 is not a country code.
    """
    match = _INTERNATIONAL_PREFIX.match(phone)
    if match is None:
        return ""
    try:
        country_code = str(phonenumbers.parse("+" + phone[match.end() :], None).country_code)
    except phonenumbers.NumberParseException:
        return ""
    end = match.end() + len(country_code)
    return phone[:end] if phone[match.end() : end] == country_code else ""
