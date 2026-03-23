# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Unsloth training smoke test -- use_unsloth=True with real HF model.

CRITICAL: This test must run in its own pytest invocation, separate from all
DP tests. Unsloth invasively patches transformers at import time, which breaks
Opacus/DP training if DP tests run in the same process.

The Makefile test-smoke-gpu target ensures process isolation via markers:
  1) Train-only: -m "requires_gpu and not vllm and not smollm2 and not unsloth"
  2) Each vLLM file in its own process (explicit file list)
  3) -m "requires_gpu and smollm2"
  4) -m "requires_gpu and unsloth"  <-- this test

Requires CUDA + internet access (Unsloth loads TinyLlama from HF Hub).
"""

import sys

import pytest
import torch

from .conftest import assert_adapter_saved, train_with_sdk

pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.unsloth,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]


def test_nss_unsloth_train_one_batch(iris_df, tmp_path):
    """Train one batch with Unsloth backend using TinyLlama from Hub."""
    from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters

    config = SafeSynthesizerParameters.from_params(
        replace_pii=None,
        pretrained_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        use_unsloth=True,
        num_input_records_to_sample=10,
        num_records=5,
        holdout=0,
        max_holdout=0,
    )
    nss = train_with_sdk(config, iris_df, tmp_path)
    assert_adapter_saved(nss._workdir)
