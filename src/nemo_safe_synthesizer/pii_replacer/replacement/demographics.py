# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading a row's demographics into the vocabulary persona sampling conditions on."""

from __future__ import annotations

import difflib
import re

import pandas as pd

from ..entities import Config

CATEGORY_FUZZY_THRESHOLD = 0.82

# Tokens that invert a following category word ("Not Hispanic", "non-white").
_NEGATION_TOKENS = frozenset({"not", "non", "no"})

__all__ = [
    "CATEGORY_FUZZY_THRESHOLD",
    "ethnicity_to_pgm",
    "fuzzy_category",
    "norm_sex",
    "race_to_sfv",
]


def _norm_cat(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _alias_preceded_by_negation(value_tokens: list[str], a_tokens: list[str]) -> bool:
    """Return True when a contiguous alias span is under a negation.

    Covers ``not black`` / ``non-white`` (alias directly after a negation token)
    and ``Not Hispanic or Latino`` (right disjunct under a shared ``not … or …``).
    Leaves ``Non-Hispanic White`` free to match ``white``.
    """
    n, m = len(value_tokens), len(a_tokens)
    if m == 0 or m > n:
        return False
    for i in range(n - m + 1):
        if value_tokens[i : i + m] != a_tokens:
            continue
        if i > 0 and value_tokens[i - 1] in _NEGATION_TOKENS:
            return True
        # Shared negation over a disjunction: "not hispanic or latino".
        if i >= 2 and value_tokens[i - 1] == "or":
            j = i - 2
            while j > 0 and value_tokens[j] not in _NEGATION_TOKENS and value_tokens[j] != "or":
                j -= 1
            if value_tokens[j] in _NEGATION_TOKENS:
                return True
    return False


def _alias_score(value_tokens: list[str], value_join: str, alias: str) -> float:
    a = _norm_cat(alias)
    if not a:
        return 0.0
    a_tokens = a.split()
    if len(a_tokens) == 1:
        if a in value_tokens:
            # "Not Hispanic" / "non-white" must not score as hispanic / white.
            if _alias_preceded_by_negation(value_tokens, a_tokens):
                return 0.0
            return 1.0
        return difflib.SequenceMatcher(None, value_join, a).ratio()
    if _alias_preceded_by_negation(value_tokens, a_tokens):
        return 0.0
    if a in value_join:
        return 1.0
    if all(t in value_tokens for t in a_tokens):
        return 0.95
    return difflib.SequenceMatcher(None, value_join, a).ratio()


def fuzzy_category(
    value: object, options: dict[str, list[str]], threshold: float = CATEGORY_FUZZY_THRESHOLD
) -> str | None:
    """Fuzzy-match a categorical value against alias options.

    Args:
        value: Raw cell value.
        options: Canonical key to alias list.
        threshold: Minimum similarity score (default ``CATEGORY_FUZZY_THRESHOLD``).

    Returns:
        Best-matching canonical key, or ``None`` if below ``threshold``.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    vj = _norm_cat(value)
    if not vj:
        return None
    vt = vj.split()
    best_key, best = None, 0.0
    for key, aliases in options.items():
        for alias in (key, *aliases):
            sc = _alias_score(vt, vj, alias)
            if sc > best:
                best_key, best = key, sc
    return best_key if best >= threshold else None


SEX_ALIASES: dict[str, list[str]] = {
    "Male": ["m", "man", "boy", "masculine", "male"],
    "Female": ["f", "woman", "girl", "feminine", "female"],
}


def norm_sex(value: object) -> str | None:
    """Normalize a sex/gender cell to ``"Male"`` / ``"Female"``, or ``None``.

    Args:
        value: Raw sex/gender cell value.

    Returns:
        ``"Male"``, ``"Female"``, or ``None`` when no alias matches.

    Example:
        ``"F"`` / ``"female"`` -> ``"Female"``.
    """
    return fuzzy_category(value, SEX_ALIASES)


# --- Fine-grained map: dataset value -> a SINGLE PGM `ethnic_background` category.
# Keys here are EXACTLY the PGM's `ethnic_background` vocabulary (39 categories, see
# sdg-pgms USPersonGenerator). When the dataset carries a specific value (e.g.
# "Mexican", "Native Hawaiian", "Vietnamese") we map it 1:1 to the matching PGM
# category so the granularity present in the data is preserved for name generation,
# instead of collapsing into a broad bucket. Generic values (e.g. bare "Asian",
# "Hispanic") deliberately do NOT match here and fall through to the broad buckets
# below, which constrain to the whole subgroup.
_PGM_ETHNIC_ALIASES: dict[str, list[str]] = {
    "white": ["white", "caucasian", "european", "anglo", "euro american"],
    "black": ["black", "african american", "african", "afro caribbean", "afro", "afro american"],
    "east asian": ["east asian", "chinese", "japanese", "korean", "taiwanese", "mongolian", "han chinese"],
    "south asian": [
        "south asian",
        "asian indian",
        "indian american",
        "pakistani",
        "bangladeshi",
        "sri lankan",
        "nepali",
        "nepalese",
        "desi",
    ],
    "southeast asian": [
        "southeast asian",
        "south east asian",
        "filipino",
        "filipina",
        "vietnamese",
        "thai",
        "indonesian",
        "malaysian",
        "cambodian",
        "khmer",
        "burmese",
        "laotian",
        "hmong",
    ],
    "central asian": ["central asian", "kazakh", "uzbek", "turkmen", "kyrgyz", "tajik", "afghan"],
    "asian other": ["asian other", "other asian"],
    "mexican": ["mexican", "chicano", "chicana"],
    "puerto rican": ["puerto rican", "puerto rico", "boricua"],
    "cuban": ["cuban"],
    "dominican": ["dominican"],
    "guatemalan": ["guatemalan"],
    "honduran": ["honduran"],
    "salvadoran": ["salvadoran", "salvadorian", "el salvador"],
    "nicaraguan": ["nicaraguan"],
    "panamanian": ["panamanian"],
    "costa rican": ["costa rican", "costa rica"],
    "argentinean": ["argentinean", "argentine", "argentinian"],
    "bolivian": ["bolivian"],
    "chilean": ["chilean"],
    "colombian": ["colombian"],
    "ecuadorian": ["ecuadorian", "ecuadorean"],
    "paraguayan": ["paraguayan"],
    "peruvian": ["peruvian"],
    "uruguayan": ["uruguayan"],
    "venezuelan": ["venezuelan"],
    "spaniard": ["spaniard"],
    "spanish": ["spanish"],
    "spanish american": ["spanish american"],
    "hispanic or latino other": ["hispanic or latino other", "other hispanic", "other latino"],
    "other central american": ["other central american"],
    "other south american": ["other south american"],
    "american indian": ["american indian", "native american", "amerindian", "first nations", "indigenous american"],
    "alaska native": ["alaska native", "alaskan native", "alaska"],
    "latin american indian": ["latin american indian"],
    "pacific islander": [
        "pacific islander",
        "native hawaiian",
        "hawaiian",
        "samoan",
        "tongan",
        "fijian",
        "chamorro",
        "guamanian",
    ],
    "polynesian": ["polynesian", "maori", "tahitian"],
    "melanesian": ["melanesian", "papuan"],
    "micronesian": ["micronesian", "marshallese", "palauan"],
}

# --- Broad buckets: fallback for generic/coarse inputs. Each maps to the FULL set
# of PGM subcategories so a generic value (e.g. "Asian") constrains to the whole
# subgroup rather than dropping any subcategory.
_ETHNICITY_GROUPS: dict[str, set[str]] = {
    "white": {"white"},
    "black": {"black"},
    "asian": {"asian other", "east asian", "south asian", "central asian", "southeast asian"},
    "hispanic": {
        "argentinean",
        "bolivian",
        "chilean",
        "colombian",
        "costa rican",
        "cuban",
        "dominican",
        "ecuadorian",
        "guatemalan",
        "hispanic or latino other",
        "honduran",
        "latin american indian",
        "mexican",
        "nicaraguan",
        "other central american",
        "other south american",
        "panamanian",
        "paraguayan",
        "peruvian",
        "puerto rican",
        "salvadoran",
        "spaniard",
        "spanish",
        "spanish american",
        "uruguayan",
        "venezuelan",
    },
    "american_indian": {"american indian", "latin american indian", "alaska native"},
    "pacific": {"polynesian", "pacific islander", "melanesian", "micronesian"},
}
_ETHNICITY_ALIASES: dict[str, list[str]] = {
    "white": ["white", "caucasian", "european", "anglo"],
    "black": ["black", "african american", "african", "afro caribbean", "afro"],
    "asian": [
        "asian",
        "east asian",
        "south asian",
        "southeast asian",
        "asian indian",
        "chinese",
        "japanese",
        "korean",
        "filipino",
        "vietnamese",
    ],
    "hispanic": ["hispanic", "latino", "latina", "latinx", "latin american", "spanish"],
    "american_indian": [
        "american indian",
        "alaska native",
        "native american",
        "indigenous",
        "first nations",
        "amerindian",
    ],
    "pacific": ["native hawaiian", "hawaiian", "pacific islander", "pacific", "polynesian", "samoan"],
}


def ethnicity_to_pgm(value: object) -> list[str] | None:
    """Map a dataset race/ethnicity value to PGM ``ethnic_background`` categories.

    Prefers a fine-grained 1:1 mapping (preserving the dataset's granularity); falls
    back to a broad bucket (the whole subgroup) only for generic/coarse inputs.

    Args:
        value: Raw race/ethnicity cell value.

    Returns:
        List of PGM ``ethnic_background`` categories, or ``None`` when unmatched.

    Example:
        ``"Vietnamese"`` -> ``["southeast asian"]``;
        ``"Asian"`` -> all east/south/southeast asian PGM categories.
    """
    fine = fuzzy_category(value, _PGM_ETHNIC_ALIASES)
    if fine:
        return [fine]
    grp = fuzzy_category(value, _ETHNICITY_ALIASES)
    return sorted(_ETHNICITY_GROUPS[grp]) if grp else None


def race_to_sfv(race_value: object, cfg: Config) -> dict[str, list[str]] | None:
    """Build persona ``select_field_values`` from race/ethnicity, or ``None`` under Faker.

    Args:
        race_value: Raw race/ethnicity cell value.
        cfg: Replacement configuration (``persona_backend`` selects matching behavior).

    Returns:
        ``{"ethnic_background": [...]}`` for PGM/managed backends, or ``None`` for Faker.

    Example:
        ``"Mexican"`` with backend ``"pgm"`` -> ``{"ethnic_background": ["mexican"]}``;
        backend ``"faker"`` -> ``None`` (Faker only conditions on sex).
    """
    # Faker only conditions given names on sex; ethnicity matching is PGM/managed.
    if cfg.persona_backend == "faker":
        return None
    cats = ethnicity_to_pgm(race_value)
    return {"ethnic_background": cats} if cats else None


def persona_match_map(match_persona_by: list | None) -> dict[str, str]:
    """Map ``persona_attribute`` names to dataframe column names.

    Args:
        match_persona_by: Plan ``match_persona_by`` entries.

    Returns:
        ``{persona_attribute: column_name}`` for entries with both fields set.

    Example:
        ``[{"persona_attribute": "sex", "column_name": "gender"}]``
        -> ``{"sex": "gender"}``.
    """
    return {
        entry["persona_attribute"]: entry["column_name"]
        for entry in match_persona_by or []
        if entry.get("persona_attribute") and entry.get("column_name")
    }
