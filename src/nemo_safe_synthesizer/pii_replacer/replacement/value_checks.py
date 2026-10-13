# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks the executor applies to generated replacement values.

This module lives outside ``replacement.generators`` so the executor can check
candidates without importing generator implementations.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING

from ...config.replace_pii import EntityType

if TYPE_CHECKING:
    from .generation import ReplacementGenerationRequest

__all__ = ["generated_value_is_valid", "is_luhn_valid"]


def generated_value_is_valid(request: ReplacementGenerationRequest, value: str) -> bool:
    """Return whether a generated value meets the request's local constraints."""
    if not value or value == request.original_value:
        return False
    if request.entity_type is EntityType.CREDIT_DEBIT_CARD:
        return is_luhn_valid(value)
    if request.entity_type is EntityType.IPV4:
        try:
            return ipaddress.ip_address(value).version == 4
        except ValueError:
            return False
    if request.entity_type is EntityType.IPV6:
        try:
            return ipaddress.ip_address(value).version == 6
        except ValueError:
            return False
    return True


def is_luhn_valid(value: str) -> bool:
    """Return whether the digits in ``value`` pass the Luhn checksum."""
    digits = [int(character) for character in value if character.isdigit()]
    if len(digits) < 2:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0
