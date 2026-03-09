# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_safe_synthesizer.config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    GenerateParameters,
    SafeSynthesizerParameters,
    TrainingHyperparams,
)


@pytest.fixture
def training_hyperparams():
    return TrainingHyperparams(
        num_input_records_to_sample=100,
        batch_size=10,
        gradient_accumulation_steps=8,
        validation_ratio=0.1,
    )


@pytest.fixture
def simple_safe_synthesizer_parameters():
    return SafeSynthesizerParameters(
        data=DataParameters(
            group_training_examples_by="my_col",
            order_training_examples_by="another_col",
        ),
        training=TrainingHyperparams(
            num_input_records_to_sample=100,
            batch_size=10,
        ),
        generation=GenerateParameters(num_records=1000, use_structured_generation=True),
        privacy=DifferentialPrivacyHyperparams(dp_enabled=False),
    )
