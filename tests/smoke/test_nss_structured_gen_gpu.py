# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU structured generation smoke test -- outlines + json_schema."""

import sys

import pytest
import torch
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

pytestmark = [
    pytest.mark.gpu_integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]


@pytest.mark.usefixtures("_patch_attn_eager")
def test_nss_structured_generation(local_tinyllama_dir, iris_df, tmp_path):
    """Train and generate with outlines structured generation backend.

    The tiny random model produces garbage, so GenerationError (no valid records)
    is acceptable -- we exercise the structured gen code path.
    """
    config = SafeSynthesizerParameters.from_params(
        enable_synthesis=True,
        enable_replace_pii=False,
        pretrained_model=str(local_tinyllama_dir),
        use_unsloth=False,
        num_input_records_to_sample=10,
        num_records=5,
        lora_r=8,
        holdout=0,
        max_holdout=0,
        use_structured_generation=True,
        structured_generation_backend="outlines",
        structured_generation_schema_method="json_schema",
    )
    nss = SafeSynthesizer(config=config, save_path=tmp_path)
    nss.with_data_source(iris_df).process_data().train()
    try:
        nss.generate()
    except GenerationError:
        pass  # Expected: random tiny model produces no valid records
