# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical descriptions of the supported PII replacement pattern grammars."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ...config.replace_pii import PatternSyntax

__all__ = [
    "CHARACTER_MASK_ESCAPABLE_CHARACTERS",
    "CHARACTER_MASK_TOKENS",
    "CharacterMaskToken",
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


def pattern_grammar_catalog() -> dict[str, dict[str, object]]:
    """Return structured grammar documentation keyed by pattern syntax name."""
    return {
        PatternSyntax.NAME_PARTS.name.lower(): {
            "description": "Whole-value templates composed of literals and name-part placeholders.",
            "placeholders": {
                "{first}": "first name",
                "{middle}": "middle name",
                "{last}": "last name",
                "{f}": "first-name initial",
                "{m}": "middle-name initial",
                "{l}": "last-name initial",
                "{F}": "uppercase first-name initial",
                "{M}": "uppercase middle-name initial",
                "{L}": "uppercase last-name initial",
                "{First}": "title-case first name",
                "{Middle}": "title-case middle name",
                "{Last}": "title-case last name",
                "{FIRST}": "uppercase first name",
                "{MIDDLE}": "uppercase middle name",
                "{LAST}": "uppercase last name",
                "{domain}": "email domain; email entities only",
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
