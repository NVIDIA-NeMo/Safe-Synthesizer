# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd

from nemo_safe_synthesizer.data_processing.actions.dates import fit_and_transform_dates


def test_fit_and_transform_dates_records_detected_date_column():
    df = pd.DataFrame(
        {
            "event_date": ["2024-01-01", "2024-01-03"],
            "label": ["start", "end"],
        }
    )

    date_columns, transformed = fit_and_transform_dates(df)

    assert date_columns["event_date"]["format"] == "%Y-%m-%d"
    assert date_columns["event_date"]["min"].startswith("2024-01-01")
    assert transformed["event_date"].tolist() == [0.0, 172800.0]
    assert df["event_date"].tolist() == ["2024-01-01", "2024-01-03"]
