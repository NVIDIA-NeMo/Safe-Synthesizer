# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU resume flow smoke test -- train -> save -> load_from_save_path -> generate."""

import sys

import pandas as pd
import pytest
import torch

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

from .conftest import train_with_sdk

pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]


@pytest.mark.usefixtures("_patch_attn_eager")
def test_nss_resume_generate_after_train(local_tinyllama_dir, iris_df, tmp_path):
    """Train, then create a new SafeSynthesizer instance and generate from saved state.

    Uses doubled iris_df (302 rows) with holdout=0.05 so load_from_save_path()
    has a non-empty test.csv to read. The base holdout=0 config produces an empty
    test split which causes EmptyDataError on resume.
    """
    # Double the dataset to exceed the 200-row holdout minimum
    large_df = pd.concat([iris_df, iris_df], ignore_index=True)

    config = SafeSynthesizerParameters.from_params(
        enable_synthesis=True,
        enable_replace_pii=False,
        pretrained_model=str(local_tinyllama_dir),
        use_unsloth=False,
        num_input_records_to_sample=10,
        num_records=5,
        lora_r=8,
        holdout=0.05,
        max_holdout=50,
    )

    # Step 1: Train
    nss1 = train_with_sdk(config, large_df, tmp_path)
    workdir = nss1._workdir

    # Step 2: New instance (simulates a new process / CLI invocation)
    nss2 = SafeSynthesizer(config=None, workdir=workdir)
    nss2.load_from_save_path()

    # Step 3: Generate from the saved state
    try:
        nss2.generate()
    except GenerationError:
        pass  # Expected: random tiny model may produce no valid records

    # Verify the resume pipeline reached the generation stage
    assert nss2.generator is not None
