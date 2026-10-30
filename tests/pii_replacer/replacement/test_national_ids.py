# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_safe_synthesizer.pii_replacer.replacement.national_ids import national_id_check


@pytest.mark.parametrize(
    ("locale", "national_id", "other"),
    [
        ("en_US", "123 45 6789", "912-45-6789"),
        ("fr_CA", "653 104 489", "653 104 488"),
        ("ja_JP", "123456789018", "123456789012"),
        ("zh_CN", "11010519491231002X", "99010519491231002X"),
    ],
)
def test_checks_the_locales_country(locale: str, national_id: str, other: str) -> None:
    is_national_id = national_id_check(locale)

    assert is_national_id is not None
    assert is_national_id(national_id)
    assert not is_national_id(other)


def test_loose_checks_apply_only_when_asked() -> None:
    assert national_id_check("nl_NL") is None
    assert (is_bsn := national_id_check("nl_NL", loose=True)) is not None
    assert is_bsn("111222333")


@pytest.mark.parametrize("locale", ["en_GB", "en", "de_AT"])
def test_locales_without_a_check(locale: str) -> None:
    assert national_id_check(locale) is None
