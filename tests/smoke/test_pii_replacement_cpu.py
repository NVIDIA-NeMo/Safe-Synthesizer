# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke test for tabular PII discovery and replacement."""

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.pii_replacement import (
    PersonaColumnSet,
    PiiColumnPlan,
    PiiEntity,
    PiiPersonBackend,
    PiiPersonConfig,
    PiiReplacementPlan,
    PiiReplacementScope,
    ReplacePiiConfig,
)
from nemo_safe_synthesizer.pii_replacer import TabularPiiReplacer


def test_tabular_pii_replacement_cpu_smoke():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2"],
            "first_name": ["Alice", "Alice", "Bob"],
            "notes": ["Alice visit", "Alice follow-up", "Bob visit"],
        }
    )
    plan = PiiReplacementPlan(
        scope=PiiReplacementScope.group,
        persona_backed_columns=[
            PersonaColumnSet(
                persona="patient",
                columns_to_replace=[
                    PiiColumnPlan(column_name="first_name", entity_type=PiiEntity.first_name),
                ],
            )
        ],
        standalone_columns_to_replace=[
            PiiColumnPlan(column_name="notes", entity_type=PiiEntity.free_text),
        ],
    )
    replacer = TabularPiiReplacer(
        ReplacePiiConfig(replacement_plan=plan, person=PiiPersonConfig(backend=PiiPersonBackend.faker)),
        data_config=DataParameters(group_training_examples_by="patient_id"),
    )
    replacer.transform_df(df)

    assert replacer.result is not None
    out = replacer.result.transformed_df
    assert "Alice" not in out["first_name"].tolist()
    assert out.loc[out["patient_id"] == "p1", "first_name"].nunique() == 1
