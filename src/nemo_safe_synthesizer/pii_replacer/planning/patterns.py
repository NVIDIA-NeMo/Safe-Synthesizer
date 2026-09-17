# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical descriptions of the supported PII replacement pattern grammars."""

from __future__ import annotations

import re
import string
from collections.abc import Mapping
from dataclasses import dataclass
from random import Random
from types import MappingProxyType
from typing import Literal

from ...config.replace_pii import EntityType, PatternSyntax

__all__ = [
    "CHARACTER_MASK_ESCAPABLE_CHARACTERS",
    "CHARACTER_MASK_TOKENS",
    "NAME_PART_PLACEHOLDERS",
    "CharacterMaskPart",
    "CharacterMaskToken",
    "NamePartPlaceholder",
    "NamePatternPart",
    "compile_character_mask",
    "compile_name_pattern",
    "parse_character_mask",
    "parse_name_pattern",
    "pattern_grammar_catalog",
    "render_character_mask",
    "render_name_pattern",
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

    part: Literal["first", "middle", "last", "domain", "organization"]
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
        "{organization}": NamePartPlaceholder(
            "organization",
            None,
            False,
            "organization dependency normalized as an email DNS label",
        ),
    }
)


@dataclass(frozen=True, slots=True)
class CharacterMaskPart:
    """One parsed literal or generated character in a character mask."""

    literal: str | None = None
    choices: str | None = None


@dataclass(frozen=True, slots=True)
class NamePatternPart:
    """One parsed literal or placeholder in a name/email pattern."""

    literal: str | None = None
    placeholder: NamePartPlaceholder | None = None


_TOKEN_CHOICES: Mapping[str, str] = MappingProxyType(
    {
        "#": string.digits,
        "^": string.ascii_uppercase,
        "@": string.ascii_lowercase,
        "&": string.digits + string.ascii_uppercase,
        "%": string.digits + string.ascii_lowercase,
        "*": string.digits + string.ascii_letters,
    }
)
_NAME_PART_PATTERN = re.compile(r"\{([^{}]+)\}")


def parse_character_mask(pattern: str) -> tuple[tuple[CharacterMaskPart, ...] | None, str | None]:
    """Parse the character-mask grammar used by validation and generation."""
    parts: list[CharacterMaskPart] = []
    has_variable = False
    index = 0
    while index < len(pattern):
        character = pattern[index]
        if character == "\\":
            if index + 1 >= len(pattern):
                return None, "ends with a trailing '\\'"
            index += 1
            escaped = pattern[index]
            if escaped not in CHARACTER_MASK_ESCAPABLE_CHARACTERS:
                allowed = " ".join(sorted(CHARACTER_MASK_ESCAPABLE_CHARACTERS))
                return None, f"escapes unsupported character {escaped!r}; only {allowed} may be escaped"
            parts.append(CharacterMaskPart(literal=escaped))
        elif choices := _TOKEN_CHOICES.get(character):
            parts.append(CharacterMaskPart(choices=choices))
            has_variable = True
        elif character == "[":
            choices, end, error = _parse_character_class(pattern, index)
            if error is not None:
                return None, error
            parts.append(CharacterMaskPart(choices=choices))
            has_variable = True
            index = end
        else:
            parts.append(CharacterMaskPart(literal=character))
        index += 1
    if not has_variable:
        return None, "has no variable placeholder"
    return tuple(parts), None


def _parse_character_class(pattern: str, start: int) -> tuple[str | None, int, str | None]:
    choices: list[str] = []
    index = start + 1
    while index < len(pattern):
        character = pattern[index]
        if character == "]":
            if not choices:
                return None, index, "has an empty '[]' character class"
            return "".join(choices), index, None
        if character == "\\":
            if index + 1 >= len(pattern):
                return None, index, "ends with a trailing '\\'"
            index += 1
            character = pattern[index]
            if character not in CHARACTER_MASK_ESCAPABLE_CHARACTERS:
                allowed = " ".join(sorted(CHARACTER_MASK_ESCAPABLE_CHARACTERS))
                return None, index, f"escapes unsupported character {character!r}; only {allowed} may be escaped"
        choices.append(character)
        index += 1
    return None, index, "has an unclosed '[' character class"


def compile_character_mask(pattern: str) -> tuple[re.Pattern[str] | None, str | None]:
    """Compile a character mask using the same parse tree used for rendering."""
    parts, error = parse_character_mask(pattern)
    if error is not None or parts is None:
        return None, error
    expression = "".join(
        re.escape(part.literal) if part.literal is not None else f"[{re.escape(part.choices or '')}]" for part in parts
    )
    return re.compile(expression), None


def render_character_mask(pattern: str, rng: Random) -> str:
    """Render one deterministic value from a valid character mask."""
    parts, error = parse_character_mask(pattern)
    if error is not None or parts is None:
        raise ValueError(error or "invalid character mask")
    return "".join(part.literal if part.literal is not None else rng.choice(part.choices or "") for part in parts)


def parse_name_pattern(
    entity_type: EntityType,
    pattern: str,
) -> tuple[tuple[NamePatternPart, ...] | None, str | None]:
    """Parse the shared whole-name and email pattern grammar."""
    matches = list(_NAME_PART_PATTERN.finditer(pattern))
    if not matches:
        return None, "has no name-part placeholder"
    unmatched = _NAME_PART_PATTERN.sub("", pattern)
    if "{" in unmatched or "}" in unmatched:
        return None, "has an unmatched '{' or '}'"

    parts: list[NamePatternPart] = []
    cursor = 0
    for match in matches:
        if literal := pattern[cursor : match.start()]:
            parts.append(NamePatternPart(literal=literal))
        placeholder = NAME_PART_PLACEHOLDERS.get(match.group(0))
        if placeholder is None:
            return None, f"uses unknown placeholder {match.group(0)!r}"
        if placeholder.part in {"domain", "organization"} and entity_type is not EntityType.EMAIL:
            return None, f"uses {match.group(0)} outside an email pattern"
        parts.append(NamePatternPart(placeholder=placeholder))
        cursor = match.end()
    if literal := pattern[cursor:]:
        parts.append(NamePatternPart(literal=literal))
    if entity_type is EntityType.EMAIL and "@" not in pattern:
        return None, "does not contain '@'"
    return tuple(parts), None


def compile_name_pattern(
    entity_type: EntityType,
    pattern: str,
) -> tuple[re.Pattern[str] | None, str | None]:
    """Compile a name/email pattern using the shared parser."""
    parts, error = parse_name_pattern(entity_type, pattern)
    if error is not None or parts is None:
        return None, error
    expression: list[str] = []
    for part in parts:
        if part.literal is not None:
            expression.append(_name_literal_regex(part.literal, entity_type))
            continue
        placeholder = part.placeholder
        if placeholder is None:
            raise RuntimeError("parsed name pattern part has no literal or placeholder")
        if placeholder.part in {"domain", "organization"}:
            expression.append(r"[^@\s]+")
        elif placeholder.initial:
            expression.append(r"[^\W\d_]")
        elif entity_type is EntityType.EMAIL:
            expression.append(r"[^@\s.]+")
        else:
            expression.append(r"[^@\s]+")
    return re.compile("".join(expression), re.UNICODE), None


def _name_literal_regex(literal: str, entity_type: EntityType) -> str:
    if entity_type is not EntityType.EMAIL:
        return re.escape(literal)
    return "".join(r"\d" if character == "#" else re.escape(character) for character in literal)


def render_name_pattern(
    entity_type: EntityType,
    pattern: str,
    values: Mapping[str, str],
    rng: Random,
) -> str:
    """Render a valid name/email pattern from normalized component values."""
    parts, error = parse_name_pattern(entity_type, pattern)
    if error is not None or parts is None:
        raise ValueError(error or "invalid name pattern")
    rendered: list[str] = []
    for part in parts:
        if part.literal is not None:
            literal = part.literal
            if entity_type is EntityType.EMAIL:
                literal = "".join(str(rng.randrange(10)) if character == "#" else character for character in literal)
            rendered.append(literal)
            continue
        placeholder = part.placeholder
        if placeholder is None:
            raise RuntimeError("parsed name pattern part has no literal or placeholder")
        component = values.get(placeholder.part, "")
        if placeholder.initial:
            component = component[:1]
        if placeholder.capitalization == "upper":
            component = component.upper()
        elif placeholder.capitalization == "title":
            component = component.title()
        elif placeholder.capitalization == "lower":
            component = component.lower()
        rendered.append(component)
    return "".join(rendered)


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
                "{organization} may only be used for email.",
                "Email patterns must contain @.",
                "In email patterns, # emits one digit.",
            ],
            "examples": ["{First} {Last}", "{f}.{last}@{domain}", "{first}@mail.{organization}.org"],
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
