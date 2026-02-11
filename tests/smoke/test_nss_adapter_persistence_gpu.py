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
    pytest.mark.gpu_integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]


@pytest.mark.usefixtures("_patch_attn_eager")
class TestAdapterPersistence:
    """Train once, verify adapter artifacts in 3 ways."""

    @classmethod
    def setup_class(cls):
        cls.workdir = None
        cls.local_dir = None

    def _ensure_trained(self, base_smoke_config, iris_df, tmp_path_factory, local_tinyllama_dir):
        """Lazily train once on first test that needs it."""
        if self.workdir is not None:
            return
        save_path = tmp_path_factory.mktemp("adapter-smoke")
        nss = train_with_sdk(base_smoke_config, iris_df, save_path)
        self.__class__.workdir = nss._workdir
        self.__class__.local_dir = local_tinyllama_dir

    def test_adapter_files_exist_after_train(self, base_smoke_config, iris_df, tmp_path_factory, local_tinyllama_dir):
        """Verify all expected adapter files are written to disk."""
        self._ensure_trained(base_smoke_config, iris_df, tmp_path_factory, local_tinyllama_dir)
        assert_adapter_saved(self.workdir)
        assert self.workdir.train.adapter.metadata.exists()  # metadata_v2.json
        assert self.workdir.train.adapter.schema.exists()  # dataset_schema.json
        assert self.workdir.train.config.exists()  # train/safe-synthesizer-config.json

    def test_adapter_loadable_by_peft(self, base_smoke_config, iris_df, tmp_path_factory, local_tinyllama_dir):
        """Verify the adapter can be loaded by PEFT and produces valid output."""
        self._ensure_trained(base_smoke_config, iris_df, tmp_path_factory, local_tinyllama_dir)
        base = AutoModelForCausalLM.from_pretrained(
            str(self.local_dir),
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base, str(self.workdir.train.adapter.path))
        model.eval()
        # Quick forward pass
        dummy_input = torch.randint(0, 100, (1, 10))
        with torch.no_grad():
            output = model(dummy_input)
        assert output.logits is not None
        assert output.logits.shape[0] == 1

    def test_adapter_config_valid(self, base_smoke_config, iris_df, tmp_path_factory, local_tinyllama_dir):
        """Verify adapter_config.json has correct LoRA parameters."""
        self._ensure_trained(base_smoke_config, iris_df, tmp_path_factory, local_tinyllama_dir)
        config_path = self.workdir.train.adapter.path / "adapter_config.json"
        with open(config_path) as f:
            adapter_cfg = json.load(f)
        assert adapter_cfg["peft_type"] == "LORA"
        assert adapter_cfg["r"] == 8
        assert "q_proj" in adapter_cfg["target_modules"]
