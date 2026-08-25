# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Column-header matching via normalization and regex labels."""

from __future__ import annotations

import re
from collections.abc import Mapping

from ...observability import get_logger
from ..entities import is_identify_only

logger = get_logger(__name__)

_BUG_REPORT_URL = "https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues"


def normalize_column_name_for_match(col: str) -> str:
    """Lowercase a column name with separators at camelCase and digit boundaries.

    ``MailingStreet`` and ``AddressLine1`` become ``mailing street`` and
    ``address line 1``, so token-aware entity regexes can match compound headers.
    Underscores are left in place so patterns like ``applicant(?![a-z_])`` still
    reject ``applicant_id``.

    Args:
        col: Raw column header name.

    Returns:
        Normalized lowercase string for regex matching.
    """
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", col)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"([A-Za-z])(\d)", r"\1 \2", s)
    s = re.sub(r"(\d)([A-Za-z])", r"\1 \2", s)
    return re.sub(r"\s+", " ", s).strip().lower()


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


def _warn_multi_header_match(col: str, matches: list[str], chosen: str) -> None:
    alternatives = [label for label in matches if label != chosen]
    logger.user.warning(
        f"[PII Replacement] Column {col!r} matched multiple header labels by name "
        f"({', '.join(matches)}); chose {chosen!r} (first match; "
        f"alternatives: {', '.join(alternatives)}). Review the replacement plan; "
        f"if the chosen type is wrong, please file a bug report at {_BUG_REPORT_URL}"
    )


def match_column_header(
    col: str,
    entity_patterns: Mapping[str, list[str]],
    demo_patterns: Mapping[str, list[str]],
) -> tuple[str | None, str | None]:
    """Classify a column header as an entity label, a demographic label, or neither.

    Regex matches are collected across **both** pattern maps. At most one of
    ``name_label`` / ``demo_label`` is returned:

    1. Zero matches → neither.
    2. One match → that label (entity or demographic).
    3. Multiple matches → first in registry order; warn.

    Args:
        col: Raw column header name.
        entity_patterns: Entity label → regex fragments (e.g. ``ENTITY_NAME_PATTERNS``).
        demo_patterns: Demographic label → regex fragments (e.g. ``DEMO_LABEL_PATTERNS``).

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

    if not matches:
        return None, None
    if len(matches) > 1:
        _warn_multi_header_match(col, matches, matches[0])
    return _split(matches[0])
