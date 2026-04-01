# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from nemo_safe_synthesizer.evaluation.components.pii_replay import PIIReplay
from nemo_safe_synthesizer.evaluation.data_model.evaluation_datasets import EvaluationDatasets

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_pii_replay(training_df_5k, synthetic_df_5k, test_df, column_statistics):
    """PII analysis on 5k rows - computationally expensive."""
    evaluation_datasets = EvaluationDatasets.from_dataframes(training_df_5k, synthetic_df_5k, test_df, column_statistics)
    pii_replay = PIIReplay.from_evaluation_datasets(evaluation_datasets)

    assert len(pii_replay.pii_replay_data) == 3

    assert pii_replay.pii_replay_data[0].column_name == "small_cat"
    assert pii_replay.pii_replay_data[0].pii_type == "some_cats"
    assert pii_replay.pii_replay_data[0].total_training_data == 1988
    assert pii_replay.pii_replay_data[0].unique_training_data == 2
    assert pii_replay.pii_replay_data[0].total_synthetic_data == 2044
    assert pii_replay.pii_replay_data[0].unique_synthetic_data == 2
    assert pii_replay.pii_replay_data[0].unique_synthetic_data_percentage == 100.0
