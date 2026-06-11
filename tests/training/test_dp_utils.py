# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nemo_safe_synthesizer.privacy.dp_transformers import dp_utils
from nemo_safe_synthesizer.privacy.dp_transformers.dp_utils import OpacusDPTrainer

pytestmark = pytest.mark.unit


def test_loss_memory_probe_rejects_overlapping_installs():
    """Concurrent enabled probes must fail fast instead of sharing global state."""
    dp_utils._uninstall_causal_lm_loss_memory_probe()
    try:
        assert dp_utils._install_causal_lm_loss_memory_probe(
            debug_loss_memory=True,
            chunked_loss=False,
            chunk_tokens=1024,
        )

        with pytest.raises(RuntimeError, match="already installed"):
            dp_utils._install_causal_lm_loss_memory_probe(
                debug_loss_memory=True,
                chunked_loss=False,
                chunk_tokens=1024,
            )
    finally:
        dp_utils._uninstall_causal_lm_loss_memory_probe()


def test_ghost_clipping_fails_when_model_ignores_logits_to_keep():
    """Ghost clipping should report models that return full-sequence logits."""

    class LogitsModel:
        def __call__(self, **_model_inputs):
            return SimpleNamespace(logits=torch.zeros(1, 3, 5))

    trainer = object.__new__(OpacusDPTrainer)
    trainer.dp_loss = lambda *_args, **_kwargs: torch.tensor(0.0)
    trainer.accelerator = SimpleNamespace(unwrap_model=lambda model, keep_fp32_wrapper=False: model)

    with pytest.raises(RuntimeError, match="logits_to_keep"):
        trainer._compute_ghost_clipping_loss(LogitsModel(), {"labels": torch.tensor([[1, 2, 3]])})
