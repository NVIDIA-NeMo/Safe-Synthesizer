# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split structured street-address values into a replaceable street and the locality text after it."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ...config.replace_pii import EntityType, PiiColumnPlan
from ...errors import GenerationError, InternalError, ParameterError
from ...observability import get_logger
from .canonicalization import is_missing_scalar
from .detection import FreeTextDetector, fresh_detection_entity_types
from .types import DetectedSpan, DetectionCell, DetectionCellId, detected_text

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["AddressParts", "detect_address_parts"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AddressParts:
    """One street-address value split at the end of its detected street.

    Example:
        ``"489 Park Blvd, Mesa, IA 76237"`` splits into street ``"489 Park Blvd"`` and locality
        ``", Mesa, IA 76237"``.
    """

    street: str = field(repr=False)
    """Leading street text that is replaced. Excluded from ``repr`` because it is PII."""

    locality: str = field(repr=False)
    """Text after the street, such as city, postcode, and country, kept as written. Excluded from ``repr``."""

    def replacement_street(self, replacement: str) -> str | None:
        """Return the generated street inside ``replacement``, or ``None`` when it does not end with the locality."""
        if not replacement.endswith(self.locality):
            return None
        return replacement[: len(replacement) - len(self.locality)] or None


def detect_address_parts(
    detector: FreeTextDetector | None,
    dataframe: pd.DataFrame,
    specs: Sequence[PiiColumnPlan],
) -> dict[str, dict[str, AddressParts]]:
    """Find the street inside each distinct street-address value, keyed by column name and value text.

    GLiNER2 street spans exclude the city, postcode, and country, so a value whose only detected span is a street
    at its start keeps the text after that street. Values without such a span are left out and replaced whole, as
    is every value when no detector is configured or detection cannot run. Detection reads the original values
    before any column is replaced; value text is the same string form the structured generator receives.
    """
    if detector is None or not specs:
        return {}
    cells = list(_address_cells(dataframe, specs))
    if not cells:
        return {}
    try:
        spans = detector.detect(cells).spans
    except GenerationError as exc:
        logger.user.warning(
            "Street-address values were replaced whole because free-text PII detection could not run",
            extra={"column_count": len(specs), "reason": str(exc)},
        )
        return {}
    requested = {cell.cell_id: cell for cell in cells}
    spans_by_cell: dict[DetectionCellId, list[DetectedSpan]] = defaultdict(list)
    for span in spans:
        cell = requested.get(span.cell_id)
        if cell is None:
            raise InternalError("free-text PII detector returned a span for an unrequested street-address cell")
        detected_text(cell, span)
        spans_by_cell[span.cell_id].append(span)

    parts: dict[str, dict[str, AddressParts]] = {}
    for cell in cells:
        split = _split_at_street(cell.text, spans_by_cell.get(cell.cell_id, ()))
        if split is not None:
            parts.setdefault(cell.cell_id.column_name, {})[cell.text] = split
    return parts


def _address_cells(dataframe: pd.DataFrame, specs: Sequence[PiiColumnPlan]) -> list[DetectionCell]:
    """Build one detection cell per distinct nonempty value text in each street-address column."""
    allowed = fresh_detection_entity_types()
    cells: list[DetectionCell] = []
    for spec in specs:
        cells.extend(_column_address_cells(dataframe, spec.column_name, allowed))
    return cells


def _column_address_cells(
    dataframe: pd.DataFrame, column_name: str, allowed: frozenset[EntityType]
) -> list[DetectionCell]:
    """Build one detection cell per distinct nonempty value text in one street-address column."""
    column_position = dataframe.columns.get_loc(column_name)
    if not isinstance(column_position, int):
        raise ParameterError(f"replacement column {column_name!r} is duplicated in the dataframe")
    column = dataframe.iloc[:, column_position]
    cells: list[DetectionCell] = []
    seen: set[str] = set()
    for row_position, (value, text) in enumerate(zip(column.tolist(), column.astype(str).tolist(), strict=True)):
        if is_missing_scalar(value) or not text or text in seen:
            continue
        seen.add(text)
        cells.append(DetectionCell(DetectionCellId(row_position, column_name), text, allowed))
    return cells


def _split_at_street(text: str, spans: Sequence[DetectedSpan]) -> AddressParts | None:
    """Split ``text`` after its street, or return ``None`` when the whole value should be replaced.

    The value must have exactly one detected span: a street that starts the value and ends at a comma or
    whitespace, followed by text containing a letter or digit. Anything else, such as a leading unit outside the
    span (``"Unit 5, 42 George Street, Sydney"``) or another detected entity, is replaced whole so no part of
    the original street can survive.
    """
    if len(spans) != 1:
        return None
    (span,) = spans
    locality = text[span.end :]
    if span.entity_type is not EntityType.STREET_ADDRESS or span.start != 0 or not locality:
        return None
    if not (locality[0] == "," or locality[0].isspace()) or not any(character.isalnum() for character in locality):
        return None
    return AddressParts(street=text[: span.end], locality=locality)
