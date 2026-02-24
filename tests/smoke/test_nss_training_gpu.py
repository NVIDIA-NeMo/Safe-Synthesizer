# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU SDK training smoke tests -- SafeSynthesizer train (standard + DP) with local tiny model."""

import sys

import pytest
import torch

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters

from .conftest import assert_adapter_saved, train_with_sdk

pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]


@pytest.mark.usefixtures("_patch_attn_eager")
def test_nss_train_one_batch(base_smoke_config, iris_df, tmp_path):
    """Train one batch through the SafeSynthesizer SDK with local tiny model."""
    nss = train_with_sdk(base_smoke_config, iris_df, tmp_path)
    assert_adapter_saved(nss._workdir)


@pytest.mark.usefixtures("_patch_attn_eager")
def test_nss_train_dp_one_batch(local_tinyllama_dir, iris_df, tmp_path):
    """Train one batch with DP enabled through the SafeSynthesizer SDK.

    Uses num_input_records_to_sample=100 (vs 10 for non-DP) to keep the epoch
    count low enough that the DP accountant's composition budget isn't exceeded.
    """
    config = SafeSynthesizerParameters.from_params(
        enable_synthesis=True,
        enable_replace_pii=False,
        pretrained_model=str(local_tinyllama_dir),
        use_unsloth=False,
        num_input_records_to_sample=100,
        num_records=5,
        lora_r=8,
        holdout=0,
        max_holdout=0,
        dp_enabled=True,
        epsilon=100.0,
    )
    nss = train_with_sdk(config, iris_df, tmp_path)
    assert_adapter_saved(nss._workdir)
