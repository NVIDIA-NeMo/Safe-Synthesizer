# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest
from nemo_safe_synthesizer.evaluation.components.pii_replay import PIIReplay
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import EvaluationDataset

logger = logging.getLogger(__name__)


@pytest.mark.slow
def test_pii_replay(train_df_5k, synth_df_5k, test_df, column_statistics):
    """PII analysis on 5k rows - computationally expensive."""
    evaluation_dataset = EvaluationDataset.from_dataframes(train_df_5k, synth_df_5k, test_df, column_statistics)
    pii_replay = PIIReplay.from_evaluation_dataset(evaluation_dataset)

    assert len(pii_replay.pii_replay_data) == 3

    assert pii_replay.pii_replay_data[0].column_name == "small_cat"
    assert pii_replay.pii_replay_data[0].pii_type == "some_cats"
    assert pii_replay.pii_replay_data[0].total_ref_data == 1988
    assert pii_replay.pii_replay_data[0].unique_ref_data == 2
    assert pii_replay.pii_replay_data[0].total_synth_data == 2044
    assert pii_replay.pii_replay_data[0].unique_synth_data == 2
    assert pii_replay.pii_replay_data[0].unique_synth_data_percentage == 100.0
