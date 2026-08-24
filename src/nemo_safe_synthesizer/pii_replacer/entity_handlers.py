# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-entity discovery gates: when a name-matched column may be allocated.

``entities.EntitySpec`` declares routing and match data. This module attaches
``skip_reason`` behavior used during heuristic discovery. Apply-time generation
(``generate`` / ``persona_value`` / pattern rejection) is deferred to the
execution PR — see ``tmp/split_prs/pii_replacement_plan_spec.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .detection.column_names import header_matches_patterns
from .detection.value_recognizers import (
    looks_like_sequential_integer_id,
    sample_has_dominant_identifier_template,
    sample_looks_like_api_key,
    sample_looks_like_multi_person,
    sample_looks_like_org_name,
    sample_looks_like_street_address,
)
from .entities import Config, EntityHandler, EntitySpec, spec

__all__ = [
    "ApiKeyHandler",
    "DateOfBirthHandler",
    "DefaultHandler",
    "EntityHandler",
    "PhoneNumberHandler",
    "StreetAddressHandler",
    "UniqueIdentifierHandler",
    "get_handler",
    "skip_reason_named_column",
]


def skip_reason_named_column(
    entity_spec: EntitySpec,
    series: pd.Series,
    value_entity: str | None,
) -> str | None:
    """Return a skip reason for a name-matched column, or ``None`` to allocate it.

    Shared gates used by ``DefaultHandler``: ``requires_value_match`` and
    ``name_shape_gates``. Entity-specific content checks live on dedicated
    handlers.

    Example:
        Header ``email`` whose values are not emails ->
        ``"values do not match that entity"``.

    Args:
        entity_spec: Registry entry for the name-matched entity.
        series: Column values used for content gates.
        value_entity: Dominant value-derived entity label, or ``None``.

    Returns:
        Human-readable skip reason, or ``None`` when the column may be allocated.
    """
    label = entity_spec.label.value
    if entity_spec.requires_value_match and value_entity != label:
        return "values do not match that entity"
    if entity_spec.name_shape_gates:
        if sample_looks_like_multi_person(series):
            return (
                "looks like multi-person values (delimiters such as 'and', '/', '&'); "
                "not auto-assigned — pre-split or hand-plan"
            )
        if sample_looks_like_org_name(series):
            return "values look like organizations, not people"
    return None


@dataclass(frozen=True)
class DefaultHandler(EntityHandler):
    """Default discovery skip gates for a name-matched entity label."""

    label: str
    """Engine entity label this handler speaks for."""

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        """Return why discovery must not allocate this column, or ``None`` to keep it.

        Args:
            series: Column values under consideration.
            value_entity: Dominant value-derived entity label, or ``None``.
            apply_path: Resolved apply path for this label.
            column_name: Unused by the default gates (entity-specific handlers may use it).
            cfg: Unused by the default gates (entity-specific handlers may use it).

        Returns:
            A skip reason, or ``None`` when the column may be allocated.
        """
        _ = (apply_path, column_name, cfg)
        entity_spec = spec(self.label)
        assert entity_spec is not None  # name-matched labels always have overlays
        return skip_reason_named_column(entity_spec, series, value_entity)


class UniqueIdentifierHandler(DefaultHandler):
    """Unique-identifier handler: sequential skip + weak-name template gate.

    ``EntitySpec.strong_name_patterns`` vs the rest of ``name_patterns`` is the
    product rule; this handler enforces it. Strong headers (``patient_id``,
    ``uuid``, …) only refuse dense sequential integers. Weak leftovers
    (``valid``, ``userid``, …) also need a dominant identifier template.
    """

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        _ = (value_entity, apply_path)
        if looks_like_sequential_integer_id(series):
            return "looks like a sequential integer id (1, 2, 3, …); not treated as a unique identifier"
        entity_spec = spec(self.label)
        assert entity_spec is not None
        # Empty strong_name_patterns means no weak tier: skip the template gate.
        if not entity_spec.strong_name_patterns or column_name is None:
            return None
        if header_matches_patterns(column_name, entity_spec.strong_name_patterns):
            return None
        if cfg is None or not sample_has_dominant_identifier_template(series, cfg):
            return "values lack a dominant identifier template"
        return None


class DateOfBirthHandler(DefaultHandler):
    """Date-of-birth handler: values must be parseable dates."""

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        _ = (series, apply_path, column_name, cfg)
        if value_entity != "date":
            return "values are not parseable dates"
        return None


class PhoneNumberHandler(DefaultHandler):
    """Phone handler: phone-specific value-mismatch messaging."""

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        _ = (series, apply_path, column_name, cfg)
        entity_spec = spec(self.label)
        assert entity_spec is not None
        if entity_spec.requires_value_match and value_entity != self.label:
            return "values do not look like phone numbers"
        return None


class ApiKeyHandler(DefaultHandler):
    """API-key handler: refuse numeric or non-credential-like columns."""

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        base = super().skip_reason(series, value_entity, apply_path, column_name=column_name, cfg=cfg)
        if base is not None:
            return base
        if pd.api.types.is_numeric_dtype(series) or not sample_looks_like_api_key(series):
            return "content is numeric or not credential-like"
        return None


class StreetAddressHandler(DefaultHandler):
    """Street-address handler: house-number sample gate."""

    def skip_reason(
        self,
        series: pd.Series,
        value_entity: str | None,
        apply_path: str,
        *,
        column_name: str | None = None,
        cfg: Config | None = None,
    ) -> str | None:
        base = super().skip_reason(series, value_entity, apply_path, column_name=column_name, cfg=cfg)
        if base is not None:
            return base
        if not sample_looks_like_street_address(series):
            return "values lack house numbers (street name only)"
        return None


# Labels with entity-specific discovery gates. Unlisted labels use DefaultHandler.
_HANDLERS: dict[str, type[DefaultHandler]] = {
    "ssn": DefaultHandler,
    "national_id": DefaultHandler,
    "date_of_birth": DateOfBirthHandler,
    "credit_debit_card": DefaultHandler,
    "unique_identifier": UniqueIdentifierHandler,
    "phone_number": PhoneNumberHandler,
    "ipv4": DefaultHandler,
    "ipv6": DefaultHandler,
    "api_key": ApiKeyHandler,
    "street_address": StreetAddressHandler,
    "first_name": DefaultHandler,
    "last_name": DefaultHandler,
    "middle_name": DefaultHandler,
    "full_name": DefaultHandler,
    "email": DefaultHandler,
}


@lru_cache(maxsize=128)
def get_handler(label: str) -> EntityHandler:
    """Return the discovery handler for ``label``.

    Args:
        label: Engine entity label (e.g. ``ssn``, ``date_of_birth``).

    Returns:
        Entity-specific handler when one is registered, else ``DefaultHandler``.
    """
    return _HANDLERS.get(label, DefaultHandler)(label)
