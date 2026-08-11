# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column-header matching: normalization, regex labels, and the fuzzy backstop."""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

from ...observability import get_logger
from ..entities import FUZZY_KEYWORDS, is_identify_only

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


def match_labels(col: str, patterns: dict[str, list[str]]) -> list[str]:
    """Return every entity label whose regex matches the normalized column name.

    Labels are returned in ``patterns`` iteration order (registry order for
    ``ENTITY_NAME_PATTERNS``).

    Args:
        col: Raw column header name.
        patterns: Mapping of entity label to list of regex fragments.

    Returns:
        Matching entity labels (possibly empty).
    """
    name = normalize_column_name_for_match(col)
    return [label for label, regexes in patterns.items() if any(re.search(rg, name) for rg in regexes)]


def match_label(col: str, patterns: dict[str, list[str]]) -> str | None:
    """Return the first entity label whose regex matches the normalized column name.

    Prefer ``fuzzy_match_label`` for entity detection (handles multi-match). This
    helper remains for demographics and simple single-label checks.

    Args:
        col: Raw column header name.
        patterns: Mapping of entity label to list of regex fragments.

    Returns:
        First matched entity label, or ``None`` when no pattern matches.
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


def _best_fuzzy_label(col: str, candidates: Iterable[str] | None = None) -> tuple[str | None, float]:
    """Return ``(label, score)`` for the best fuzzy keyword match among candidates.

    When ``candidates`` is ``None``, scores every label in ``FUZZY_KEYWORDS``.
    Ties keep the first label seen (candidate / registry order).
    """
    joined = re.sub(r"[^a-z0-9]+", "", col.lower())
    if not joined:
        return None, 0.0
    labels: Iterable[str] = FUZZY_KEYWORDS.keys() if candidates is None else candidates
    best_label, best = None, 0.0
    for label in labels:
        for key in FUZZY_KEYWORDS.get(label) or ():
            sc = difflib.SequenceMatcher(None, joined, key).ratio()
            if sc > best:
                best_label, best = label, sc
    return best_label, best


# Curated, specific keyword spellings per label for the fuzzy backstop. These are
# deliberately NOT generic single words (no bare "name"/"date"/"id"), so a typo'd
# variant matches but unrelated columns (e.g. "event_name", "event_date") do not.
def fuzzy_match_label(col: str, patterns: dict[str, list[str]], threshold: float) -> str | None:
    """Return an entity label via regex match(es), with fuzzy resolution when needed.

    1. Collect **all** regex matches against ``patterns``.
    2. Zero matches → fuzzy backstop over ``FUZZY_KEYWORDS`` (must meet ``threshold``).
    3. One match → that label.
    4. Multiple matches → fuzzy score among those candidates only; pick the highest
       and warn so the plan can be reviewed (overlapping name patterns are unexpected).

    Example:
        ``"emial_address"`` at threshold ``0.86`` -> ``"email"``.

    Args:
        col: Raw column header name.
        patterns: Mapping of entity label to list of regex fragments.
        threshold: Minimum fuzzy similarity ratio in ``[0.0, 1.0]`` for the
            no-regex-match backstop.

    Returns:
        Matched entity label, or ``None`` when neither regex nor fuzzy match succeeds.
    """
    matches = match_labels(col, patterns)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        best_label, _ = _best_fuzzy_label(col, matches)
        chosen = best_label or matches[0]
        alternatives = [label for label in matches if label != chosen]
        logger.user.warning(
            f"[PII Replacement] Column {col!r} matched multiple entity types by name "
            f"({', '.join(matches)}); chose {chosen!r} by fuzzy similarity "
            f"(alternatives: {', '.join(alternatives)}). Review the replacement plan; "
            f"if the chosen type is wrong, please file a bug report at {_BUG_REPORT_URL}"
        )
        return chosen

    best_label, best = _best_fuzzy_label(col)
    return best_label if best >= threshold else None
