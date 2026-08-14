# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope consistency: repeated originals map to one synthetic within each unit."""

from __future__ import annotations

import pandas as pd
import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiReplacementPlan,
    PiiReplacementScope,
)
from nemo_safe_synthesizer.pii_replacer.entities import Config
from nemo_safe_synthesizer.pii_replacer.replacement import run_replacement


def _cfg() -> Config:
    return Config(
        locale="en_US",
        random_seed=11,
        persona_backend="faker",
        sdg_pgms_src="/tmp",
        managed_assets_path=None,
    )


@pytest.mark.parametrize("scope", list(PiiReplacementScope))
def test_repeated_original_consistent_within_scope_unit(scope: PiiReplacementScope):
    """Within each scope unit, the same original ID/name gets the same synthetic.

    Across units, ``record`` and ``group`` may differ; ``dataframe`` must not.
    """
    df = pd.DataFrame(
        {
            "group_id": ["G1", "G1", "G2", "G2"],
            "patient_id": ["ID-1", "ID-1", "ID-1", "ID-1"],
            "first_name": ["Alice", "Alice", "Alice", "Alice"],
        }
    )
    plan = PiiReplacementPlan(
        scope=scope,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="person_1",
                columns_to_replace=[PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name)],
            )
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="patient_id", entity_type=PiiEntity.unique_identifier),
        ],
    )
    out = run_replacement(df, plan, _cfg(), group_key="group_id").replaced_df

    assert (out["patient_id"] != df["patient_id"]).all()
    assert (out["first_name"] != df["first_name"]).all()

    if scope == PiiReplacementScope.dataframe:
        assert out["patient_id"].nunique() == 1
        assert out["first_name"].nunique() == 1
        return

    if scope == PiiReplacementScope.group:
        for _, g in out.groupby(df["group_id"]):
            assert g["patient_id"].nunique() == 1
            assert g["first_name"].nunique() == 1
        # Same original across groups may (and typically does) differ under group scope,
        # so no cross-group assertion is made here.
        return

    # record: each row is its own unit
    assert out["patient_id"].nunique() == len(out)
    assert out["first_name"].nunique() == len(out)
