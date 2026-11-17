# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.pii_replacer.phone_numbers import (
    country_code_prefix,
    kept_phone_digit_count,
    phone_identity,
    phone_in_layout,
    phone_region,
)


@pytest.mark.unit
class TestPhoneNumbers:
    @pytest.mark.parametrize(("locale", "region"), [("en_US", "US"), ("fr_CA", "CA"), ("en-GB", "GB"), ("en", None)])
    def test_phone_region_is_the_locales_country(self, locale: str, region: str | None) -> None:
        assert phone_region(locale) == region

    @pytest.mark.parametrize(
        ("text", "region", "identity"),
        [
            ("(555) 123-4567", "US", "+15551234567"),
            ("555.123.4567", "US", "+15551234567"),
            ("+1 555 123 4567", None, "+15551234567"),
            ("555-123-4567 x89", "US", "+15551234567;ext=89"),
            ("020 7946 0958", "GB", "+442079460958"),
            ("555.123.4567", None, None),
            ("unlisted", "US", None),
            ("12", "US", None),
        ],
    )
    def test_phone_identity_is_the_e164_form(self, text: str, region: str | None, identity: str | None) -> None:
        assert phone_identity(text, region) == identity

    @pytest.mark.parametrize(
        ("occurrence", "replacement", "expected"),
        [
            ("555.123.4567", "(812) 904-3321", "812.904.3321"),
            ("+1 555 123 4567", "(812) 904-3321", "+1 812 904 3321"),
            ("555-123-4567", "+1 (812) 904-3321", "812-904-3321"),
            ("020 7946 0958", "+44 81 2604 9135", "+44 81 2604 9135"),
        ],
    )
    def test_phone_in_layout_moves_digits_after_the_country_code_when_they_fit(
        self, occurrence: str, replacement: str, expected: str
    ) -> None:
        assert phone_in_layout(occurrence, replacement) == expected

    @pytest.mark.parametrize(
        ("phone", "prefix"),
        [
            ("+442079460958", "+44"),
            ("+1-202-555-0101", "+1"),
            ("(+353) 1 234 5678", "(+353"),
            ("001-555-123-4567", "001"),
            ("+999 123", ""),
            ("555-123-4567", ""),
        ],
    )
    def test_country_code_prefix(self, phone: str, prefix: str) -> None:
        assert country_code_prefix(phone) == prefix

    @pytest.mark.parametrize(
        ("phone", "region", "kept"),
        [
            # The area code, or the mobile prefix, is kept.
            ("(415) 555-2671", "US", "415"),
            ("(415) 555-2671 x89", "US", "415"),
            ("+86 138 1234 5678", None, "86138"),
            ("090-1234-5678", "JP", "090"),
            ("03-1234-5678", "JP", "03"),
            ("0171 1234567", "DE", "0171"),
            ("+39 06 1234 5678", None, "3906"),
            # A national trunk prefix before the area code is kept with it.
            ("020 7946 0958", "GB", "020"),
            ("+44 (0)20 7946 0958", None, "44020"),
            ("1-800-356-9377", "US", "1800"),
            # At most half the national number is kept, as in Singapore, which has no area codes.
            ("+91 98765 43210", None, "9198765"),
            ("+65 6123 4567", None, "656123"),
            # Without a known split, only a country code is kept.
            ("12345", "US", ""),
            ("4155552671", None, ""),
            ("+1 415 555 2671", None, "1415"),
            ("1-800-FLOWERS", "US", ""),
        ],
    )
    def test_kept_phone_digit_count_covers_the_country_code_and_area_code(
        self, phone: str, region: str | None, kept: str
    ) -> None:
        count = kept_phone_digit_count(phone, region)

        assert "".join(character for character in phone if character.isdigit())[:count] == kept
