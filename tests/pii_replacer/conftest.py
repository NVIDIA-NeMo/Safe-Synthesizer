# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for pii_replacer tabular tests."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.pii_replacer.helpers import CONTACT_NAMES, FIRSTS, LASTS, PHONE_MINORITY


@pytest.fixture
def fixture_patient_df() -> pd.DataFrame:
    """Grouped patient/provider rows with free-text visit notes."""
    return pd.DataFrame(
        {
            "patient_id": ["A", "A", "B", "B", "B"],
            "first_name": ["Alice", "Alice", "Bob", "Bob", "Bob"],
            "provider_name": ["Dr X", "Dr Y", "Dr Z", "Dr Z", "Dr Z"],
            "notes": [
                "Alice visited Dr X",
                "Alice visited Dr Y",
                "Bob visited Dr Z",
                "Bob again Dr Z",
                "Bob follow-up Dr Z",
            ],
        }
    )


@pytest.fixture
def fixture_dob_df() -> pd.DataFrame:
    """Birth dates in a dominant %m/%d/%Y format plus one ISO minority row."""
    return pd.DataFrame(
        {
            "patient_id": ["A", "B", "C"],
            "first_name": ["Alice", "Bob", "Cleo"],
            "sex": ["Female", "Male", "Female"],
            "date_of_birth": ["01/15/1980", "02/20/1990", "1975-03-25"],
            "notes": ["visit", "follow-up", "discharge"],
        }
    )


@pytest.fixture
def fixture_phone_df() -> pd.DataFrame:
    """Contacts sharing one phone format, plus two rows in a second format."""
    dominant = [f"+1-415-555-{1000 + i:04d}" for i in range(18)]
    return pd.DataFrame(
        {
            "contact_id": [f"C{i:03d}" for i in range(20)],
            "full_name": CONTACT_NAMES * 2,
            "phone": dominant + [PHONE_MINORITY, "(206) 555-0127"],
        }
    )


@pytest.fixture
def fixture_contact_df() -> pd.DataFrame:
    """A directory that writes 'SMITH, Jane' and 'j.smith@acme.com'."""
    rows = []
    for i in range(60):
        first, last = FIRSTS[i % 10], LASTS[(i // 10) % 10]
        rows.append(
            {
                "patient_name": f"{last.upper()}, {first}",
                "patient_email": f"{first[0].lower()}.{last.lower()}@{'acme' if i % 2 else 'globex'}.com",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fixture_numbered_email_df() -> pd.DataFrame:
    """A directory that hyphenates its surnames and numbers every address."""
    rows = []
    for i in range(60):
        first = FIRSTS[i % 10]
        last = f"{LASTS[(i // 10) % 10]}-{LASTS[(i + 3) % 10]}"
        rows.append(
            {
                "contact_name": f"{first} {last}",
                "contact_email": f"{first.lower()}.{last.lower()}.{100 + i}-{2000 + i}@example.invalid",
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def fixture_middle_name_df() -> pd.DataFrame:
    """Sample names for persona pattern inference tests."""
    from nemo_safe_synthesizer.pii_replacer.replacement import seeded_faker

    fake = seeded_faker(3, "en_US")
    rows = []
    for _ in range(60):
        first, middle, last = fake.first_name(), fake.first_name(), fake.last_name()
        rows.append(
            {
                "first_name": first,
                "middle_name": middle,
                "last_name": last,
                "full_name": f"{first} {middle} {last}",
            }
        )
    return pd.DataFrame(rows)
