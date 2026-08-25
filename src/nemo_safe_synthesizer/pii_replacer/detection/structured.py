# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured-column detection for heuristic PII plan discovery."""

from __future__ import annotations

import pandas as pd

from ...errors import ParameterError
from ...observability import get_logger
from ..entities import (
    DEMO_LABEL_PATTERNS,
    ENTITY_NAME_PATTERNS,
    Config,
    EntitySpec,
    demo_keys_for_backend,
    is_identify_only,
    spec,
)
from ..models import (
    ColumnEvidence,
    DetectedField,
    DetectedStandalone,
    DemographicAttribute,
    DiscoveryResult,
    SamePersonBundle,
)
from ..patterns import date_patterns
from .column_names import match_column_header, name_supports_value_entity
from .value_recognizers import analyze_column_patterns, probe_numeric_column

logger = get_logger(__name__)

_MULTI_PERSON_MSG = (
    "Looks like there is more than one person in the dataset. "
    "Heuristic auto-discovery supports a single subject; use LLM mode for multi-person data."
)


def allocation_skip_reason(entity_spec: EntitySpec, value_entity: str | None) -> str | None:
    """Return why a name-matched column must not be allocated, or ``None`` to keep it.

    Discovery gates only:
    - ``requires_value_match`` entities need a matching dominant value entity
    - ``date_of_birth`` needs date-shaped values (classifier emits ``\"date\"``)

    Apply-time per-entity generate / persona sampling is deferred to the
    execution PR (see ``tmp/split_prs/pii_replacement_plan_spec.md``).
    """
    label = entity_spec.label.value
    if label == "date_of_birth":
        if value_entity != "date":
            return "values are not parseable dates"
        return None
    if entity_spec.requires_value_match and value_entity != label:
        return "values do not match that entity"
    return None


def detect_structured_columns(df_subset: pd.DataFrame, cfg: Config) -> DiscoveryResult:
    """Detect persona-backed and standalone PII columns.

    Heuristics mode assumes a single subject: at most one column per
    persona-backed entity type (``apply_path="persona"``). Duplicates raise
    ``ParameterError`` pointing at LLM mode.

    Args:
        df_subset: Dataframe whose columns are classified.
        cfg: Engine configuration controlling thresholds and sampler backend.

    Returns:
        Typed discovery result. ``free_text_columns`` is always empty here;
        discovery fills it via ``select_free_text_columns``.
    """
    person_fields: dict[str, str] = {}
    demo_cols: dict[str, str] = {}
    standalone: list[DetectedStandalone] = []
    identified_not_replaced: list[str] = []

    def _add_standalone(col: str, entity: str, pattern: str | None = None, *, note: str = "") -> None:
        standalone.append(DetectedStandalone(column=col, entity=entity, pattern=pattern))
        extra = f" — {note}" if note else ""
        logger.runtime.info(
            f"[PII Replacement] Standalone column {col!r} (entity={entity}, pattern={pattern!r}){extra}"
        )

    def _add_person_field(col: str, label: str) -> None:
        if label in person_fields:
            prior = person_fields[label]
            raise ParameterError(
                f"{_MULTI_PERSON_MSG} "
                f"Found multiple {label!r} columns ({prior!r} and {col!r})."
            )
        person_fields[label] = col
        logger.runtime.info(f"[PII Replacement] Same-person column {col!r} (entity={label})")

    def _gather(col: str) -> ColumnEvidence:
        series = df_subset[col]
        name_label, demo_label = match_column_header(col, ENTITY_NAME_PATTERNS, DEMO_LABEL_PATTERNS)
        phone_min = 7 if name_label == "phone_number" else 10
        analysis = analyze_column_patterns(series, cfg, phone_min_digits=phone_min)
        value_entity = analysis["entity"] if analysis["structured"] else None
        # Never assign *replaceable* entities from values alone. Temporals
        # (identify-not-replaced) keep value evidence without a name match.
        if value_entity is not None and not name_supports_value_entity(name_label, value_entity):
            value_entity = None
        return ColumnEvidence(col, series, name_label, value_entity, analysis, demo_label)

    for col in df_subset.columns:
        ev = _gather(col)

        # 1) Numeric probes (dtype-driven; still require a supportive header).
        numeric_probe = probe_numeric_column(ev.series, ev.name_label)
        if numeric_probe == "date_of_birth":
            _add_standalone(
                col,
                "date_of_birth",
                "%Y%m%d",
                note="perturbed per record/group (not same-person-tied)",
            )
            continue
        if numeric_probe in {"unique_identifier", "national_id", "ssn"}:
            _add_standalone(col, numeric_probe, None)
            continue

        # 2) Name-matched entity: resolve EntitySpec → gates → allocate.
        if ev.name_label is not None:
            entity_spec = spec(ev.name_label)
            if is_identify_only(ev.name_label):
                identified_not_replaced.append(col)
                logger.runtime.info(
                    f"[PII Replacement] Identified column {col!r} (entity={ev.name_label}) — "
                    "excluded from replacement plan"
                )
                continue
            apply_path = entity_spec.apply_path if entity_spec is not None else None
            if entity_spec is not None and apply_path is not None:
                skip = allocation_skip_reason(entity_spec, ev.value_entity)
                if skip is not None:
                    logger.user.warning(
                        f"[PII Replacement] Column {col!r} looks like {ev.name_label} by name but {skip}; skipped."
                    )
                    continue

                if apply_path == "standalone_map":
                    pattern: str | None = None
                    note = ""
                    if ev.name_label == "date_of_birth":
                        dob_patterns = date_patterns(ev.series.dropna())
                        pattern = dob_patterns[0] if dob_patterns else None
                        note = "perturbed per record/group (not same-person-tied)"
                    _add_standalone(col, ev.name_label, pattern, note=note)
                    continue

                _add_person_field(col, ev.name_label)
                continue

        # 3) Demographic matchers (read-only conditioners).
        if ev.demo_label:
            if ev.demo_label in demo_cols:
                logger.user.warning(
                    f"[PII Replacement] Ignoring duplicate demographic column {col!r} "
                    f"({ev.demo_label}); keeping {demo_cols[ev.demo_label]!r}."
                )
            else:
                demo_cols[ev.demo_label] = col

        # 4) Identify-only temporals (value evidence, no replaceable name match).
        if (
            ev.value_entity is not None
            and is_identify_only(ev.value_entity)
            and ev.name_label != "date_of_birth"
            and ev.demo_label is None
        ):
            identified_not_replaced.append(col)
            logger.runtime.info(
                f"[PII Replacement] Identified temporal column {col!r} (entity={ev.value_entity}, "
                f"pattern={ev.analysis['pattern']}, coverage={ev.analysis['coverage']}) — "
                "excluded from replacement plan"
            )

    same_person_bundles: list[SamePersonBundle] = []
    if person_fields or demo_cols:
        demographics: dict[DemographicAttribute, str] = {}
        for attr in demo_keys_for_backend(cfg.sampler_backend):
            if not demo_cols.get(attr):
                continue
            if attr in {"gender", "ethnic_background"}:
                demographics[attr] = demo_cols[attr]
        # Demographics-only (no persona fields) is dropped at plan emission.
        # Heuristics mode emits at most one bundle; LLM mode may emit more later.
        same_person_bundles.append(
            SamePersonBundle(
                bundle_id="person",
                fields={label: DetectedField(column=col) for label, col in person_fields.items()},
                demographics=demographics,
            )
        )

    return DiscoveryResult(
        same_person_bundles=same_person_bundles,
        free_text_columns=[],
        standalone_columns=standalone,
        identified_not_replaced=identified_not_replaced,
    )
