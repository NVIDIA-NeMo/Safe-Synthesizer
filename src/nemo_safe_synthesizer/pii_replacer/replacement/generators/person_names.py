# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write full names in a locale's own order and read them back into first, middle, and last names.

A full name without a plan pattern is written the way Faker writes the locale's names: given name first in
``en_US``, family name first in ``ja_JP``, ``hu_HU``, and ``ru_RU``, and family name first with no space in
``zh_CN`` and ``ko_KR``. nameparser reads full names back, told which order to expect: the column's pattern's
order, or else the locale's.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from faker import Faker
from nameparser import FAMILY_FIRST, GIVEN_FIRST, Parser, Policy, locales, parser_for

from ....config.replace_pii import EntityType
from ....errors import GenerationError
from ...planning.patterns import parse_name_pattern

__all__ = ["FullNameLayout", "full_name_layout", "read_full_name", "write_full_name"]

_FAKER_NAME_TOKEN = re.compile(r"\{\{(first|last)_name\w*\}\}")

# nameparser locale packs, by language. The Chinese and Japanese packs split unspaced names, such as 山田太郎.
_NAMEPARSER_PACKS = {"zh": "zh", "ja": "ja", "ru": "ru", "tr": "tr_az", "az": "tr_az"}


@dataclass(frozen=True, slots=True)
class FullNameLayout:
    """How a locale writes a full name."""

    family_first: bool
    """Whether the family name comes before the given name."""

    separator: str
    """Text between the two names: a space, or nothing in Chinese and Korean."""


_GIVEN_FIRST_LAYOUT = FullNameLayout(family_first=False, separator=" ")


@lru_cache(maxsize=None)
def full_name_layout(locale: str) -> FullNameLayout:
    """Return the layout of the first Faker full-name format for ``locale`` that has a given and a family name.

    Raises:
        GenerationError: If Faker cannot load ``locale``.

    Example:
        ``full_name_layout("zh_CN")`` is family first with no separator, from Faker's ``{{last_name}}{{first_name}}``.
    """
    try:
        providers = Faker(locale).providers
    except AttributeError as exc:
        raise GenerationError(f"PII replacement could not load Faker's name formats for locale {locale!r}") from exc
    for provider in providers:
        if getattr(provider, "__provider__", None) != "faker.providers.person":
            continue
        for name_format in getattr(provider, "formats", ()):
            tokens = {match.group(1): match for match in _FAKER_NAME_TOKEN.finditer(name_format)}
            if len(tokens) < 2:
                continue
            first, last = tokens["first"], tokens["last"]
            earlier, later = sorted((first, last), key=lambda match: match.start())
            between = name_format[earlier.end() : later.start()]
            return FullNameLayout(
                family_first=last.start() < first.start(), separator=between if between in {"", " "} else " "
            )
    return _GIVEN_FIRST_LAYOUT


def write_full_name(first: str, last: str, locale: str) -> str:
    """Return ``first`` and ``last`` as a full name in ``locale``'s layout.

    Example:
        ``write_full_name("太郎", "山田", "ja_JP")`` is ``"山田 太郎"``.
    """
    layout = full_name_layout(locale)
    return f"{last}{layout.separator}{first}" if layout.family_first else f"{first}{layout.separator}{last}"


def read_full_name(full_name: str, *, locale: str, pattern: str | None = None) -> dict[str, str]:
    """Return the ``first``, ``middle``, and ``last`` names in ``full_name``, read by nameparser.

    The name is read in the order of ``pattern`` when it places both a first and a last name, and otherwise in
    ``locale``'s order. A comma always marks the last name first, a particle such as ``van`` stays with the last name,
    and titles and suffixes are left out.

    Example:
        ``read_full_name("Hopper, Grace", locale="en_US")`` and ``read_full_name("Nagy Péter", locale="hu_HU")``
        have first names ``Grace`` and ``Péter``.
    """
    family_first = _pattern_family_first(pattern)
    if family_first is None:
        family_first = full_name_layout(locale).family_first
    first, middle, last = _parse(full_name, locale.split("_", 1)[0].casefold(), family_first)
    return {"first": first, "middle": middle, "last": last}


def _pattern_family_first(pattern: str | None) -> bool | None:
    if pattern is None:
        return None
    parts, error = parse_name_pattern(EntityType.FULL_NAME, pattern)
    if error is not None or parts is None:
        return None
    order = [part.placeholder.part for part in parts if part.placeholder is not None]
    if "first" not in order or "last" not in order:
        return None
    return order.index("last") < order.index("first")


@lru_cache(maxsize=1024)
def _parse(full_name: str, language: str, family_first: bool) -> tuple[str, str, str]:
    # Parsing takes about 0.1 ms, and every column of a row that depends on the full name reads it again.
    parsed = _parser(language, family_first).parse(full_name)
    return parsed.given, parsed.middle, parsed.family


@lru_cache(maxsize=None)
def _parser(language: str, family_first: bool) -> Parser:
    # An empty script_orders keeps nameparser from reading Chinese, Japanese, and Korean script family first on its
    # own: the order is always the one the name was written in.
    base = Parser(policy=Policy(name_order=FAMILY_FIRST if family_first else GIVEN_FIRST, script_orders=()))
    pack = _NAMEPARSER_PACKS.get(language)
    if pack is None:
        return base
    if pack == "ja":
        # The Japanese splitter comes from nameparser's ja extra and takes about a second to load.
        return parser_for(locales.get(pack), base=base, segmenter=locales.ja_segmenter())
    return parser_for(locales.get(pack), base=base)
