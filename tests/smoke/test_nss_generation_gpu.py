# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU SDK generation smoke tests -- full chain train+generate + manual VllmBackend."""

import sys

import pytest
import torch
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

pytestmark = [
    pytest.mark.gpu_integration,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]


@pytest.fixture(scope="class")
def trained_nss(base_smoke_config, iris_df, tmp_path_factory):
    """Train once per class; both SDK chain and manual VllmBackend tests consume this.

    Patches attn_implementation inline because _patch_attn_eager is function-scoped
    (via monkeypatch) and cannot be active when a class-scoped fixture runs.
    """
    from nemo_safe_synthesizer.training.huggingface_backend import HuggingFaceBackend

    original_build = HuggingFaceBackend._build_base_framework_params

    def patched_build(self, model_kwargs):
        model_kwargs.setdefault("attn_implementation", "sdpa")
        return original_build(self, model_kwargs)

    HuggingFaceBackend._build_base_framework_params = patched_build
    try:
        save_path = tmp_path_factory.mktemp("gen-smoke")
        nss = SafeSynthesizer(config=base_smoke_config, save_path=save_path)
        nss.with_data_source(iris_df).process_data().train()
        return nss
    finally:
        HuggingFaceBackend._build_base_framework_params = original_build


class TestNSSGenerationGPU:
    """GPU generation tests sharing a single training run via the trained_nss fixture."""

    def test_nss_full_chain_train_and_generate(self, trained_nss):
        """Train and generate through the full SDK chain.

        The tiny random model produces garbage output, so GenerationError
        (no valid records) is acceptable -- we just exercise the code path.
        """
        try:
            trained_nss.generate()
        except GenerationError:
            pass  # Expected: random tiny model produces no valid records

    def test_manual_vllm_backend_with_local_model(self, trained_nss, local_tinyllama_dir):
        """Manually construct VllmBackend and generate with the saved adapter."""
        from nemo_safe_synthesizer.generation.vllm_backend import VllmBackend
        from nemo_safe_synthesizer.llm.metadata import ModelMetadata

        workdir = trained_nss._workdir
        config = trained_nss._nss_config
        metadata = ModelMetadata.from_metadata_json(workdir.metadata_file, workdir=workdir)
        backend = VllmBackend(config=config, model_metadata=metadata, workdir=workdir)
        backend.initialize()
        backend.prepare_params(temperature=0.9, top_p=1.0, max_new_tokens=64)
        try:
            backend.generate(keep_llm_state=False)
        except GenerationError:
            pass  # Expected: random tiny model produces no valid records
