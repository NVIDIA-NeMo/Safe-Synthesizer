# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical descriptions of the supported PII replacement pattern grammars."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from ...config.replace_pii import PatternSyntax

__all__ = [
    "CHARACTER_MASK_ESCAPABLE_CHARACTERS",
    "CHARACTER_MASK_TOKENS",
    "NAME_PART_PLACEHOLDERS",
    "CharacterMaskToken",
    "NamePartPlaceholder",
    "pattern_grammar_catalog",
]


@dataclass(frozen=True, slots=True)
class CharacterMaskToken:
    """Regex and user-facing description for one character-mask token."""

    regex: str
    description: str


CHARACTER_MASK_TOKENS: Mapping[str, CharacterMaskToken] = MappingProxyType(
    {
        "#": CharacterMaskToken(regex=r"\d", description="digit 0-9"),
        "^": CharacterMaskToken(regex="[A-Z]", description="uppercase letter A-Z"),
        "@": CharacterMaskToken(regex="[a-z]", description="lowercase letter a-z"),
        "&": CharacterMaskToken(regex="[A-Z0-9]", description="digit or uppercase letter"),
        "%": CharacterMaskToken(regex="[a-z0-9]", description="digit or lowercase letter"),
        "*": CharacterMaskToken(regex="[A-Za-z0-9]", description="digit or letter"),
    }
)
_CHARACTER_MASK_ESCAPE_ORDER = (*CHARACTER_MASK_TOKENS, "[", "]", "\\")
CHARACTER_MASK_ESCAPABLE_CHARACTERS = frozenset(_CHARACTER_MASK_ESCAPE_ORDER)


@dataclass(frozen=True, slots=True)
class NamePartPlaceholder:
    """Meaning and presentation of one accepted name-part placeholder."""

    part: Literal["first", "middle", "last", "domain"]
    capitalization: Literal["lower", "title", "upper"] | None
    initial: bool
    description: str


NAME_PART_PLACEHOLDERS: Mapping[str, NamePartPlaceholder] = MappingProxyType(
    {
        "{first}": NamePartPlaceholder("first", "lower", False, "first name"),
        "{middle}": NamePartPlaceholder("middle", "lower", False, "middle name"),
        "{last}": NamePartPlaceholder("last", "lower", False, "last name"),
        "{f}": NamePartPlaceholder("first", "lower", True, "first-name initial"),
        "{m}": NamePartPlaceholder("middle", "lower", True, "middle-name initial"),
        "{l}": NamePartPlaceholder("last", "lower", True, "last-name initial"),
        "{F}": NamePartPlaceholder("first", "upper", True, "uppercase first-name initial"),
        "{M}": NamePartPlaceholder("middle", "upper", True, "uppercase middle-name initial"),
        "{L}": NamePartPlaceholder("last", "upper", True, "uppercase last-name initial"),
        "{First}": NamePartPlaceholder("first", "title", False, "title-case first name"),
        "{Middle}": NamePartPlaceholder("middle", "title", False, "title-case middle name"),
        "{Last}": NamePartPlaceholder("last", "title", False, "title-case last name"),
        "{FIRST}": NamePartPlaceholder("first", "upper", False, "uppercase first name"),
        "{MIDDLE}": NamePartPlaceholder("middle", "upper", False, "uppercase middle name"),
        "{LAST}": NamePartPlaceholder("last", "upper", False, "uppercase last name"),
        "{domain}": NamePartPlaceholder("domain", None, False, "email domain; email entities only"),
    }
)


def pattern_grammar_catalog() -> dict[str, dict[str, object]]:
    """Return structured grammar documentation keyed by pattern syntax name."""
    return {
        PatternSyntax.NAME_PARTS.name.lower(): {
            "description": "Whole-value templates composed of literals and name-part placeholders.",
            "placeholders": {
                placeholder: definition.description for placeholder, definition in NAME_PART_PLACEHOLDERS.items()
            },
            "rules": [
                "Literal separators and punctuation are preserved.",
                "{domain} may only be used for email.",
                "Email patterns must contain @.",
                "In email patterns, # emits one digit.",
            ],
            "examples": ["{First} {Last}", "{f}.{last}@{domain}"],
        },
        PatternSyntax.CHARACTER_MASK.name.lower(): {
            "description": "One generated character per variable token; all other characters are literal.",
            "tokens": {
                **{token: definition.description for token, definition in CHARACTER_MASK_TOKENS.items()},
                "[abc]": "one literal character from the brackets",
                "\\x": r"literal escaped special character; for example, \# matches #",
            },
            "rules": [
                "Bracket contents are literal choices, not ranges.",
                "Only " + ", ".join(_CHARACTER_MASK_ESCAPE_ORDER) + " may follow an escape character.",
                "The pattern must contain at least one variable token.",
            ],
            "examples": ["pmc-#######-#", "CUST-10[01]###", r"literal-\#-###"],
        },
        PatternSyntax.STRFTIME.name.lower(): {
            "description": "Python strftime/strptime format describing the complete datetime value.",
            "examples": ["%m/%d/%Y", "%Y-%m-%d"],
        },
    }
