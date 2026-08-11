# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers and constants for pii_replacer tabular tests."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from nemo_safe_synthesizer.config.pii_replacement import PersonaColumnSet, PiiColumnPlan, PiiReplacementPlan

PHONE_TEMPLATE = "+1-415-555-####"
PHONE_MINORITY = "(206) 555-0114"

CONTACT_NAMES = [
    "Alice Adams",
    "Bob Brown",
    "Cleo Clark",
    "Dan Diaz",
    "Eve Evans",
    "Finn Foley",
    "Gina Gray",
    "Hank Hill",
    "Iris Ito",
    "Jon Jones",
]

FIRSTS = ["Jane", "John", "Maria", "Liam", "Aisha", "Noah", "Priya", "Omar", "Sofia", "Ethan"]
LASTS = ["Smith", "Doe", "Garcia", "Nguyen", "Okafor", "Rossi", "Patel", "Haddad", "Silva", "Brown"]

PgmCheckout = Literal["complete", "without_package", "absent"]


def column_spec(columns: list[PiiColumnPlan], column_name: str) -> PiiColumnPlan | None:
    return next((spec for spec in columns if spec.column_name == column_name), None)


def persona_set(plan: PiiReplacementPlan, persona: str) -> PersonaColumnSet:
    return next(col_set for col_set in plan.persona_backed_columns if col_set.persona == persona)


def pgm_checkout(root: Path, layout: PgmCheckout) -> Path:
    """A stand-in for an sdg-pgms checkout, so no test reads the real one.

    sdg-pgms is an internal repository absent from CI, and its default location
    lives under a home directory the test user may not be able to stat at all.
    """
    src = root / "sdg-pgms" / "src"
    if layout == "absent":
        return src
    src.mkdir(parents=True)
    if layout == "complete":
        (src / "pgms").mkdir()
        (src / "pgms" / "__init__.py").touch()
    return src
