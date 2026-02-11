# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU full pipeline smoke test -- SmolLM2-135M from Hub + vLLM.

Requires CUDA + internet access (downloads SmolLM2-135M ~270MB from HuggingFace Hub).
"""

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


@pytest.mark.usefixtures("_patch_attn_eager", "_register_smollm2")
def test_full_pipeline_smollm2(iris_df, smoke_save_path):
    """Train SmolLM2-135M then generate with vLLM in a single end-to-end pass.

    SmolLM2-135M with only 50 sampled rows packs into ~1 training example,
    so the model barely learns. Generation may produce no valid records
    (GenerationError), which is acceptable -- this test exercises the full
    train-then-generate code path, not output quality.
    """
    config = SafeSynthesizerParameters.from_params(
        enable_synthesis=True,
        enable_replace_pii=False,
        pretrained_model="HuggingFaceTB/SmolLM2-135M",
        use_unsloth=False,
        num_input_records_to_sample=50,
        num_records=10,
        holdout=0,
        max_holdout=0,
    )
    nss = SafeSynthesizer(config=config, save_path=smoke_save_path)
    nss.with_data_source(iris_df).process_data().train()

    try:
        nss.generate()
    except GenerationError:
        print("\n--- SmolLM2-135M: no valid records (expected with minimal training) ---\n")
        return

    result = nss.generator.gen_results
    assert result is not None
    print("\n--- SmolLM2-135M generation results ---")
    print(
        f"valid: {result.num_valid_records}  invalid: {result.num_invalid_records}  "
        f"fraction valid: {result.valid_record_fraction:.2%}"
    )
    print(result.df.to_string(max_rows=20))
    print("--- end results ---\n")
