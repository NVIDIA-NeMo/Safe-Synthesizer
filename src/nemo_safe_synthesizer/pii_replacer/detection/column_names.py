# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column-header matching: normalization, regex labels, and the fuzzy backstop."""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable, Mapping

from ...observability import get_logger
from ..entities import DEMO_FUZZY_KEYWORDS, FUZZY_KEYWORDS, is_identify_only

logger = get_logger(__name__)

_BUG_REPORT_URL = "https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues"


def normalize_column_name_for_match(col: str) -> str:
    """Lowercase a column name with separators at camelCase and digit boundaries.

    ``MailingStreet`` and ``AddressLine1`` become ``mailing street`` and
    ``address line 1``, so token-aware entity regexes can match compound headers.
    Underscores are left in place so patterns like ``applicant(?![a-z_])`` still
    reject ``applicant_id``. Content gates decide whether a column is kept.

    Args:
        col: Raw column header name.

    Returns:
        Normalized lowercase string for regex and fuzzy matching.
    """
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", col)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def header_matches_patterns(col: str, patterns: Iterable[str]) -> bool:
    """Return whether the normalized header matches any of ``patterns``."""
    name = normalize_column_name_for_match(col)
    return any(re.search(pattern, name, re.IGNORECASE) for pattern in patterns)


def match_labels(col: str, patterns: Mapping[str, list[str]]) -> list[str]:
    """Return every label whose regex matches the normalized column name.

    Labels are returned in ``patterns`` iteration order (registry order for
    ``ENTITY_NAME_PATTERNS``).

    Args:
        col: Raw column header name.
        patterns: Mapping of label to list of regex fragments.

    Returns:
        Matching labels (possibly empty).
    """
    name = normalize_column_name_for_match(col)
    return [label for label, regexes in patterns.items() if any(re.search(rg, name) for rg in regexes)]


def match_label(col: str, patterns: Mapping[str, list[str]]) -> str | None:
    """Return the first label whose regex matches the normalized column name.

    Prefer ``match_column_header`` for detection (handles multi-match across entity
    and demographic patterns). This helper remains for simple single-pattern checks.

    Args:
        col: Raw column header name.
        patterns: Mapping of label to list of regex fragments.

    Returns:
        First matched label, or ``None`` when no pattern matches.
    """
    labels = match_labels(col, patterns)
    return labels[0] if labels else None


def name_supports_value_entity(name_label: str | None, value_entity: str) -> bool:
    """Whether a column header supports using this value-derived entity for replacement.

    Replaceable entities are never assigned from values alone: value evidence is kept
    only when the header already names the same entity (or a compatible one, e.g.
    ``date_of_birth`` headers with date-shaped values). Identify-not-replaced
    temporals are exempt and keep value evidence without a name match.

    Example:
        ``("email", "email")`` -> ``True``
        ``(None, "ssn")`` -> ``False`` (values alone never allocate replaceable PII)
        ``(None, "date")`` -> ``True`` (identify-only temporals are exempt)

    Args:
        name_label: Entity label inferred from the column header, or ``None``.
        value_entity: Dominant entity label inferred from cell values.

    Returns:
        ``True`` when value evidence may be used for this column.
    """
    if is_identify_only(value_entity):
        return True
    if name_label is None:
        return False
    if name_label == value_entity:
        return True
    return name_label == "date_of_birth" and value_entity == "date"


def _fuzzy_keys(label: str) -> list[str]:
    keys = list(FUZZY_KEYWORDS.get(label) or ())
    keys.extend(DEMO_FUZZY_KEYWORDS.get(label) or ())
    return keys


def _best_fuzzy_label(col: str, candidates: Iterable[str] | None = None) -> tuple[str | None, float]:
    """Return ``(label, score)`` for the best fuzzy keyword match among candidates.

    When ``candidates`` is ``None``, scores every label in ``FUZZY_KEYWORDS`` (entity
    typo backstop only). Ties keep the first label seen (candidate order).
    """
    joined = re.sub(r"[^a-z0-9]+", "", col.lower())
    if not joined:
        return None, 0.0
    labels: Iterable[str] = FUZZY_KEYWORDS.keys() if candidates is None else candidates
    best_label, best = None, 0.0
    for label in labels:
        for key in _fuzzy_keys(label):
            sc = difflib.SequenceMatcher(None, joined, key).ratio()
            if sc > best:
                best_label, best = label, sc
    return best_label, best


def _warn_multi_header_match(col: str, matches: list[str], chosen: str) -> None:
    alternatives = [label for label in matches if label != chosen]
    logger.user.warning(
        f"[PII Replacement] Column {col!r} matched multiple header labels by name "
        f"({', '.join(matches)}); chose {chosen!r} by fuzzy similarity "
        f"(alternatives: {', '.join(alternatives)}). Review the replacement plan; "
        f"if the chosen type is wrong, please file a bug report at {_BUG_REPORT_URL}"
    )


def match_column_header(
    col: str,
    entity_patterns: Mapping[str, list[str]],
    demo_patterns: Mapping[str, list[str]],
    threshold: float,
) -> tuple[str | None, str | None]:
    """Classify a column header as an entity label, a demographic label, or neither.

    Regex matches are collected across **both** pattern maps. At most one of
    ``name_label`` / ``demo_label`` is returned:

    1. Zero matches → fuzzy backstop over entity ``FUZZY_KEYWORDS`` (must meet
       ``threshold``); demographics are not invented from typos alone.
    2. One match → that label (entity or demographic).
    3. Multiple matches (entity/entity, demo/demo, or entity/demo) → fuzzy score
       among those candidates; pick the highest and warn.

    Args:
        col: Raw column header name.
        entity_patterns: Entity label → regex fragments (e.g. ``ENTITY_NAME_PATTERNS``).
        demo_patterns: Demographic label → regex fragments (e.g. ``DEMO_LABEL_PATTERNS``).
        threshold: Minimum fuzzy similarity for the no-regex entity backstop.

    Returns:
        ``(name_label, demo_label)`` with at most one side set.
    """
    demo_keys = set(demo_patterns)
    matches = match_labels(col, entity_patterns) + [label for label in match_labels(col, demo_patterns)]
    # De-dupe while preserving order (entity hits first, then demos).
    seen: set[str] = set()
    ordered: list[str] = []
    for label in matches:
        if label not in seen:
            seen.add(label)
            ordered.append(label)
    matches = ordered

    def _split(chosen: str | None) -> tuple[str | None, str | None]:
        if chosen is None:
            return None, None
        if chosen in demo_keys:
            return None, chosen
        return chosen, None

    if len(matches) == 1:
        return _split(matches[0])
    if len(matches) > 1:
        best_label, _ = _best_fuzzy_label(col, matches)
        chosen = best_label or matches[0]
        _warn_multi_header_match(col, matches, chosen)
        return _split(chosen)

    best_label, best = _best_fuzzy_label(col)
    if best_label is None or best < threshold:
        return None, None
    return _split(best_label)


# Curated, specific keyword spellings per label for the fuzzy backstop. These are
# deliberately NOT generic single words (no bare "name"/"date"/"id"), so a typo'd
# variant matches but unrelated columns (e.g. "event_name", "event_date") do not.
def fuzzy_match_label(col: str, patterns: Mapping[str, list[str]], threshold: float) -> str | None:
    """Return an entity label via regex/fuzzy match (entity patterns only).

    Detection should prefer ``match_column_header``, which also considers
    demographic patterns. This helper remains for entity-only callers and tests.

    Example:
        ``"emial_address"`` at threshold ``0.86`` -> ``"email"``.
    """
    name_label, _demo_label = match_column_header(col, patterns, {}, threshold)
    return name_label
