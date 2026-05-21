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
def test_nss_train_one_batch(fixture_base_smoke_config, fixture_gpu_smoke_df, tmp_path):
    """Train one batch through the SafeSynthesizer SDK with local tiny model."""
    nss = train_with_sdk(fixture_base_smoke_config, fixture_gpu_smoke_df, tmp_path)
    assert nss._workdir is not None
    assert_adapter_saved(nss._workdir)


@pytest.mark.usefixtures("_patch_attn_eager")
def test_nss_train_one_batch_bnb_4bit_quantization_scheme(fixture_local_tinyllama_dir, fixture_gpu_smoke_df, tmp_path):
    """Train one batch with explicit ``quantization_scheme=bnb-4bit`` (v5 API guard)."""
    config = SafeSynthesizerParameters.from_params(
        replace_pii=None,
        pretrained_model=str(fixture_local_tinyllama_dir),
        num_input_records_to_sample=10,
        num_records=5,
        lora_r=8,
        holdout=0,
        max_holdout=0,
        quantize_model=True,
        quantization_scheme="bnb-4bit",
    )
    nss = train_with_sdk(config, fixture_gpu_smoke_df, tmp_path)
    assert nss._workdir is not None
    assert_adapter_saved(nss._workdir)


@pytest.mark.usefixtures("_patch_attn_eager")
def test_nss_train_dp_one_batch(fixture_local_tinyllama_dir, fixture_gpu_smoke_df, tmp_path):
    """Train one batch with DP enabled through the SafeSynthesizer SDK.

    Uses num_input_records_to_sample=100 (vs 10 for non-DP) to keep the epoch
    count low enough that the DP accountant's composition budget isn't exceeded.
    """
    config = SafeSynthesizerParameters.from_params(
        replace_pii=None,
        pretrained_model=str(fixture_local_tinyllama_dir),
        num_input_records_to_sample=100,
        num_records=5,
        lora_r=8,
        holdout=0,
        max_holdout=0,
        dp_enabled=True,
        epsilon=100.0,
    )
    nss = train_with_sdk(config, fixture_gpu_smoke_df, tmp_path)
    assert nss._workdir is not None
    assert_adapter_saved(nss._workdir)
