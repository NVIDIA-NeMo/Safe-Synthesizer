# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phone number helpers backed by libphonenumber, shared by PII replacement and its evaluation."""

from __future__ import annotations

import re

import phonenumbers

__all__ = ["country_code_prefix", "kept_phone_digit_count", "phone_identity", "phone_in_layout", "phone_region"]

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


def kept_phone_digit_count(phone: str, region: str | None) -> int:
    """Return how many leading digits of ``phone`` a replacement keeps: its country code and area code.

    The area code, or a mobile number's network prefix, is libphonenumber's national destination code,
    so where it ends depends on the country. A national trunk prefix written before it, such as the UK's
    leading ``0``, is kept with it. At most half of the national number is kept, so countries without
    area codes, such as Singapore, where libphonenumber reports the first group instead, still get most
    of their digits replaced. ``region`` reads a number written without a country code; when the number
    cannot be read, or it has no area code, only a country code is kept.

    Example:
        ``kept_phone_digit_count("(415) 555-2671", "US")`` is 3, for ``415``;
        ``kept_phone_digit_count("+44 20 7946 0958", None)`` is 4, for ``44`` and ``20``; and
        ``kept_phone_digit_count("12345", "US")`` is 0.
    """
    country_code = country_code_prefix(phone)
    country_code_digits = sum(character.isdigit() for character in country_code)
    try:
        number = phonenumbers.parse(phone, region)
    except phonenumbers.NumberParseException:
        return country_code_digits
    national_number = phonenumbers.national_significant_number(number)
    area_code_length = min(phonenumbers.length_of_national_destination_code(number), len(national_number) // 2)
    # The national number starts after any trunk prefix; letters, as in 1-800-FLOWERS, leave it unfound.
    written_digits = "".join(character for character in phone[len(country_code) :] if character.isdigit())
    start = written_digits.find(national_number)
    if start < 0 or area_code_length == 0:
        return country_code_digits
    return country_code_digits + start + area_code_length
