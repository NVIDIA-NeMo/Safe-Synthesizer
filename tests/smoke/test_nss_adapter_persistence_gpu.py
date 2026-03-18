# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU adapter persistence smoke tests -- file checks + PEFT load + config validation."""

import json
import sys

import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from .conftest import assert_adapter_saved, train_with_sdk

pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]


@pytest.fixture(scope="class")
def adapter_train_artifacts(_patch_attn_eager, base_smoke_config, iris_df, tmp_path_factory, local_tinyllama_dir):
    """Train once per class, return (workdir, local_model_dir) for all adapter tests."""
    save_path = tmp_path_factory.mktemp("adapter-smoke")
    nss = train_with_sdk(base_smoke_config, iris_df, save_path)
    return nss._workdir, local_tinyllama_dir


class TestAdapterPersistence:
    """Train once, verify adapter artifacts in 3 ways."""

    def test_adapter_files_exist_after_train(self, adapter_train_artifacts):
        """Verify all expected adapter files are written to disk."""
        workdir, _ = adapter_train_artifacts
        assert_adapter_saved(workdir)
        assert workdir.train.adapter.metadata.exists()  # metadata_v2.json
        assert workdir.train.adapter.schema.exists()  # dataset_schema.json
        assert workdir.train.config.exists()  # train/safe-synthesizer-config.json

    def test_adapter_loadable_by_peft(self, adapter_train_artifacts):
        """Verify the adapter can be loaded by PEFT and produces valid output."""
        workdir, local_dir = adapter_train_artifacts
        base = AutoModelForCausalLM.from_pretrained(
            str(local_dir),
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base, str(workdir.train.adapter.path))
        model.eval()
        dummy_input = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            output = model(dummy_input)
        assert output.logits is not None
        assert output.logits.shape[0] == 1

    def test_adapter_config_valid(self, adapter_train_artifacts):
        """Verify adapter_config.json has correct LoRA parameters."""
        workdir, _ = adapter_train_artifacts
        config_path = workdir.train.adapter.path / "adapter_config.json"
        with open(config_path) as f:
            adapter_cfg = json.load(f)
        assert adapter_cfg["peft_type"] == "LORA"
        assert adapter_cfg["r"] == 8
        assert "q_proj" in adapter_cfg["target_modules"]
