# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers and constants for pii_replacer discovery tests."""

from __future__ import annotations

from nemo_safe_synthesizer.config.replace_pii import EntityType, PiiColumnPlan, PiiReplacementPlan

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


def column_spec(plan: PiiReplacementPlan, column_name: str) -> PiiColumnPlan | None:
    return next((spec for spec in plan.columns_to_replace if spec.column_name == column_name), None)


def depends_on_columns(spec: PiiColumnPlan) -> list[str]:
    return [dep.column_name for dep in spec.depends_on]


def depends_on_types(spec: PiiColumnPlan) -> list[EntityType | None]:
    return [dep.entity_type for dep in spec.depends_on]
