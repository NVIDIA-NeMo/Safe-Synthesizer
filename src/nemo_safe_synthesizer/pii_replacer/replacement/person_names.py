# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write full names in a locale's own order and read them back into first, middle, and last names.

A full name without a plan pattern is written the way Faker writes the locale's names: given name first in
``en_US``, family name first in ``ja_JP``, ``hu_HU``, and ``ru_RU``, and family name first with no space in
``zh_CN`` and ``ko_KR``. nameparser reads full names back, told which order to expect: the column's pattern's
order, or else the locale's.

This module lives outside ``replacement.generators`` so free-text replacement can read names without loading the
generator implementations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from faker import Faker
from nameparser import FAMILY_FIRST, GIVEN_FIRST, Lexicon, ParsedName, Parser, Policy, locales, parser_for

from ...config.replace_pii import EntityType
from ...errors import GenerationError
from ..planning.patterns import parse_name_pattern

__all__ = [
    "FullNameLayout",
    "full_name_layout",
    "is_kept_title",
    "name_affixes",
    "parse_full_name",
    "read_full_name",
    "titles_before",
    "write_full_name",
]

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
    parsed = parse_full_name(full_name, locale=locale, pattern=pattern)
    return {"first": parsed.given, "middle": parsed.middle, "last": parsed.family}


def parse_full_name(full_name: str, *, locale: str, pattern: str | None = None) -> ParsedName:
    """Return nameparser's reading of ``full_name``, in the order :func:`read_full_name` describes.

    The result also gives each word's role and position, such as a title, given name, or family name.
    """
    family_first = _pattern_family_first(pattern)
    if family_first is None:
        family_first = full_name_layout(locale).family_first
    return _parse(full_name, locale.split("_", 1)[0].casefold(), family_first)


_KEPT_TITLES = frozenset(
    (
        # Academic and medical
        "dr",
        "doctor",
        "dra",
        "prof",
        "professor",
        # Religious
        "rev",
        "reverend",
        "pastor",
        "rabbi",
        "imam",
        "bishop",
        "fr",
        "father",
        # Military and police
        "capt",
        "captain",
        "col",
        "colonel",
        "gen",
        "general",
        "lt",
        "lieutenant",
        "maj",
        "major",
        "sgt",
        "sergeant",
        "adm",
        "admiral",
        "cmdr",
        "cdr",
        "commander",
        "det",
        "detective",
        "officer",
        # Judicial and government
        "hon",
        "honorable",
        "honourable",
        "judge",
        "justice",
        "sen",
        "senator",
        "rep",
        "representative",
        "gov",
        "governor",
        "amb",
        "ambassador",
        "mayor",
    )
)
"""Titles of a profession or office, without any trailing period and case-folded, which replacement keeps.

Other titles, such as ``Mr.``, ``Mrs.``, ``Herr``, and ``Sir``, only say how to address someone and are dropped.
"""

_TITLES = Lexicon.default().titles
"""Every title nameparser knows, case-folded and without a trailing period, such as ``mr``, ``dr``, and ``herr``."""

# One capitalized word, with an optional period, and the spaces after it, at the end of some text.
_TRAILING_WORD = re.compile(r"(?<![\w.])([^\W\d_][\w'\u2019-]*\.?)(\s+)$")
_MAX_TITLES_BEFORE = 3


def is_kept_title(word: str) -> bool:
    """Return whether replacement keeps the title ``word``: one of a profession or office, such as ``Dr.``.

    Structured and free-text replacement both keep a title only when this is true, and drop every other title.

    Example:
        ``is_kept_title("Dr.")`` and ``is_kept_title("Capt")`` are true; ``is_kept_title("Mrs.")`` is false.
    """
    return word.rstrip(".").casefold() in _KEPT_TITLES


def titles_before(text: str, name_start: int, *, floor: int = 0) -> tuple[int, str]:
    """Find titles written just before a name in ``text`` and return where they start and the ones to keep.

    Up to three capitalized words that nameparser knows as titles, each followed by spaces, are read back from
    ``name_start`` but not before ``floor``. The kept text has only the titles :func:`is_kept_title` keeps, each
    with the spaces after it, so writing it in place of ``text[start:name_start]`` drops the others.

    Example:
        ``titles_before("Seen by Mr. Dr. Ada", 16)`` is ``(8, "Dr. ")``, and ``titles_before("Seen by Ada", 8)`` is
        ``(8, "")``.
    """
    start = name_start
    kept: list[str] = []
    for _ in range(_MAX_TITLES_BEFORE):
        match = _TRAILING_WORD.search(text, floor, start)
        if match is None:
            break
        word = match.group(1)
        if not word[0].isupper() or word.rstrip(".").casefold() not in _TITLES:
            break
        start = match.start()
        if is_kept_title(word):
            kept.insert(0, match.group(0))
    return start, "".join(kept)


def name_affixes(full_name: str, *, locale: str, pattern: str | None = None) -> tuple[str, str]:
    """Return ``full_name``'s leading titles and trailing suffixes as written, or ``""`` for each it lacks.

    Titles of a profession or office, such as ``Dr.`` and ``Capt.``, and suffixes such as ``Jr.`` and ``PhD`` are
    kept, each with the spaces and commas that follow or precede it. Other titles, such as ``Mr.`` and ``Mrs.``,
    are left out. Names whose titles or suffixes sit between name words return
    ``("", "")``.

    Example:
        ``name_affixes("Dr. Ada Lovelace, PhD", locale="en_US")`` is ``("Dr. ", ", PhD")``, and
        ``name_affixes("Mrs. Grace Hopper", locale="en_US")`` is ``("", "")``.
    """
    tokens = parse_full_name(full_name, locale=locale, pattern=pattern).tokens
    roles = [token.role.value if token.role is not None else None for token in tokens]
    named = [position for position, role in enumerate(roles) if role not in {"title", "suffix"}]
    spans = [token.span for token in tokens]
    if not named or any(span is None for span in spans):
        return "", ""
    first, last = named[0], named[-1]
    if any(role != "title" for role in roles[:first]) or any(role != "suffix" for role in roles[last + 1 :]):
        return "", ""
    starts = [span.start for span in spans if span is not None]
    ends = [span.end for span in spans if span is not None]
    # Each kept title runs up to the next word, so the space or other text after it stays.
    leading = "".join(
        full_name[starts[position] : starts[position + 1]]
        for position in range(first)
        if is_kept_title(tokens[position].text)
    )
    trailing = full_name[ends[last] :] if last < len(tokens) - 1 else ""
    return leading, trailing


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
def _parse(full_name: str, language: str, family_first: bool) -> ParsedName:
    # Parsing takes about 0.1 ms, and every column of a row that depends on the full name reads it again.
    return _parser(language, family_first).parse(full_name)


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
