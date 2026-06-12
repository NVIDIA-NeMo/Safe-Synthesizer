# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pandas as pd

from nemo_safe_synthesizer.data_processing.actions.data_actions import DatetimeCol, ReplaceDataSource
from nemo_safe_synthesizer.data_processing.actions.utils import ActionCtx, UniqueIdSource


def test_replace_datasource_state_stays_json_and_restores_column_index():
    ctx = ActionCtx()
    action = ReplaceDataSource(col="replacement", data_source=UniqueIdSource()).with_ctx(ctx)
    source = pd.DataFrame(
        {
            "left": [1, 2],
            "replacement": ["old-a", "old-b"],
            "right": [3, 4],
        }
    )

    preprocessed = action.preprocess(source)

    assert list(preprocessed.columns) == ["left", "right"]
    assert ctx.state[action.hash()] == '{"column_index":1}'

    generated = action.generate(preprocessed)

    assert list(generated.columns) == ["left", "replacement", "right"]
    assert generated["replacement"].notna().all()


def test_datetime_col_state_stays_json_and_validates_with_inferred_format():
    ctx = ActionCtx()
    action = DatetimeCol(name="started_at").with_ctx(ctx)
    source = pd.DataFrame({"started_at": ["2024-01-20", "2024-01-21"]})

    action.preprocess(source)

    state = json.loads(ctx.state[action.hash()])
    assert state == {"dt_format": "%Y-%m-%d"}

    batch = pd.DataFrame({"started_at": ["2024-01-22", "not a date"]})
    valid, rejected = action.validate_batch(batch, pd.DataFrame())

    assert valid["started_at"].tolist() == ["2024-01-22"]
    assert rejected["started_at"].tolist() == ["not a date"]
