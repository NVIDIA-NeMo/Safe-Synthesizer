# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nemo_safe_synthesizer.config.training import TrainingMemoryControls
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


def test_chunked_cross_entropy_matches_transformers_loss_with_masked_labels():
    """Chunked loss should preserve masking and normalization semantics."""
    from transformers.loss import loss_utils

    logits = torch.randn(2, 5, 7)
    shift_labels = torch.tensor(
        [
            [1, -100, 3, 4, -100],
            [-100, 2, 5, -100, 6],
        ]
    )

    flat_logits = logits.float().view(-1, logits.shape[-1])
    flat_labels = shift_labels.view(-1).to(flat_logits.device)

    expected_mean = loss_utils.fixed_cross_entropy(flat_logits, flat_labels, None, -100)
    actual_mean = dp_utils._chunked_cross_entropy(
        logits,
        shift_labels,
        vocab_size=logits.shape[-1],
        num_items_in_batch=None,
        ignore_index=-100,
        chunk_size=3,
    )
    assert torch.allclose(actual_mean, expected_mean)

    num_items = torch.tensor(4)
    expected_scaled = loss_utils.fixed_cross_entropy(flat_logits, flat_labels, num_items, -100)
    actual_scaled = dp_utils._chunked_cross_entropy(
        logits,
        shift_labels,
        vocab_size=logits.shape[-1],
        num_items_in_batch=num_items,
        ignore_index=-100,
        chunk_size=3,
    )
    assert torch.allclose(actual_scaled, expected_scaled)


def test_loss_memory_probe_skips_ghost_mode_and_warns(monkeypatch):
    """Ghost clipping bypasses Transformers causal-LM loss, so probe-only controls warn."""
    trainer = object.__new__(OpacusDPTrainer)
    trainer.grad_sample_mode = "ghost"
    trainer.memory_controls = TrainingMemoryControls(
        chunked_causal_lm_loss=True,
        debug_loss_memory=True,
    )

    def fail_install(**_kwargs):
        raise AssertionError("ghost mode should not install the Transformers loss probe")

    warnings: list[str] = []
    monkeypatch.setattr(dp_utils, "_install_causal_lm_loss_memory_probe", fail_install)
    monkeypatch.setattr(dp_utils.logger, "warning", lambda message: warnings.append(message))

    assert trainer._install_loss_memory_probe() is False
    assert any("ignored for grad_sample_mode='ghost'" in message for message in warnings)


def test_training_observability_survives_logger_failure(monkeypatch):
    """Logger failures should not drop the structured event for backend handoff."""
    trainer = object.__new__(OpacusDPTrainer)
    trainer.args = SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=3)
    trainer.grad_sample_mode = "hooks"
    trainer._peak_loss_logits_bytes = 4 * 1024**3
    trainer.last_training_observability = None

    def fail_log(*_args, **_kwargs):
        raise RuntimeError("log handler failed")

    monkeypatch.setattr(dp_utils.logger.runtime, "info", fail_log)
    monkeypatch.setattr(dp_utils.logger, "warning", lambda _message: None)

    trainer._emit_training_observability(peak_vram_gb=5.0)

    event = trainer.last_training_observability
    assert event is not None
    assert event.peak_vram_gb == 5.0
    assert event.peak_loss_logits_gb == 4.0
    assert event.effective_batch_size == 6
