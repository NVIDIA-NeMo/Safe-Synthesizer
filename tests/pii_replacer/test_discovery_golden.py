# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden ``discover_plan`` snapshots for stub datasets.

After intentional discovery changes, regenerate each ``*.plan.yaml`` under
``tests/pii_replacer/golden/`` from the matching stub CSV (faker backend) and
re-check ``test_discover_plan_matches_golden``.

Compare structured plan content (not comment formatting) so header wording can change.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nemo_safe_synthesizer.config.replace_pii import (
    PiiReplacementPlan,
    PiiSamplerBackend,
    PiiSamplerConfig,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer.entities import config_from_replace_pii
from nemo_safe_synthesizer.pii_replacer.planning import discover_plan, discover_plan_with_hints

_STUB = Path(__file__).resolve().parents[1] / "stub_datasets"
_GOLDEN = Path(__file__).resolve().parent / "golden"

# (golden stem, csv relative to stub_datasets, group_training_examples_by, optional nrows)
_GOLDEN_CASES = (
    ("telco_churn", "telco_churn_sample.csv", None, None),
    ("clinc_oos", "clinc_oos.csv", None, 50),
)


def _plan_fingerprint(plan: PiiReplacementPlan) -> dict:
    return json.loads(plan.model_dump_json(exclude_none=True, exclude_defaults=True))


def _load_golden_plan(stem: str) -> PiiReplacementPlan:
    raw = yaml.safe_load((_GOLDEN / f"{stem}.plan.yaml").read_text()) or {}
    return PiiReplacementPlan.model_validate(raw)


@pytest.mark.parametrize(("stem", "csv_name", "group_key", "nrows"), _GOLDEN_CASES)
def test_discover_plan_matches_golden(stem: str, csv_name: str, group_key: str | None, nrows: int | None):
    """Auto-discovery on stub CSVs must match checked-in plan goldens."""
    config = ReplacePiiConfig(sampler=PiiSamplerConfig(backend=PiiSamplerBackend.faker))
    cfg = config_from_replace_pii(config)
    df = pd.read_csv(_STUB / csv_name, nrows=nrows)
    discovered = discover_plan(df, group_key=group_key, cfg=cfg, config=config)
    expected = _load_golden_plan(stem)
    assert _plan_fingerprint(discovered) == _plan_fingerprint(expected)


def test_patient_events_multi_person_emits_unlinked_plan():
    """Heuristics mode flattens multi-person stubs (patient + provider names)."""
    config = ReplacePiiConfig(sampler=PiiSamplerConfig(backend=PiiSamplerBackend.faker))
    cfg = config_from_replace_pii(config)
    df = pd.read_csv(_STUB / "sample-patient-events-12groups-200-records.csv")
    plan, hints = discover_plan_with_hints(df, group_key="patient_id", cfg=cfg, config=config)
    assert plan.scope is not None
    assert all(not spec.depends_on for spec in plan.columns_to_replace)
    assert hints
    assert any("depends_on omitted" in h for h in hints)
