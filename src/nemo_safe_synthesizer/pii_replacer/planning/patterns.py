# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical descriptions of the supported PII replacement pattern grammars."""

from __future__ import annotations

from ...config.replace_pii import PatternSyntax

__all__ = ["pattern_grammar_catalog"]


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
                "#": "digit 0-9",
                "^": "uppercase letter A-Z",
                "@": "lowercase letter a-z",
                "&": "digit or uppercase letter",
                "%": "digit or lowercase letter",
                "*": "digit or letter",
                "[abc]": "one literal character from the brackets",
                "\\x": "literal x",
            },
            "rules": [
                "Bracket contents are literal choices, not ranges.",
                "The pattern must contain at least one variable token.",
            ],
            "examples": ["pmc-#######-#", "CUST-10[01]###"],
        },
        PatternSyntax.STRFTIME.name.lower(): {
            "description": "Python strftime/strptime format describing the complete datetime value.",
            "examples": ["%m/%d/%Y", "%Y-%m-%d"],
        },
    }
