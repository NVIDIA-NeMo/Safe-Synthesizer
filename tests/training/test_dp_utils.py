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


def test_chunked_cross_entropy_averages_per_sequence_means():
    """Chunked DP loss should keep each sequence as one privacy unit."""
    logits = torch.randn(3, 5, 7, requires_grad=True)
    shift_labels = torch.tensor(
        [
            [1, -100, 3, -100, -100],
            [1, 2, 5, -100, 6],
            [-100, -100, -100, -100, -100],
        ]
    )

    expected = torch.stack(
        [
            torch.nn.functional.cross_entropy(logits[0, [0, 2]].float(), shift_labels[0, [0, 2]]),
            torch.nn.functional.cross_entropy(logits[1, [0, 1, 2, 4]].float(), shift_labels[1, [0, 1, 2, 4]]),
            logits[2].reshape(-1)[0].float() * 0,
        ]
    ).mean()
    actual = dp_utils._chunked_cross_entropy(
        logits,
        shift_labels,
        vocab_size=logits.shape[-1],
        ignore_index=-100,
        chunk_size=3,
    )
    torch.testing.assert_close(actual, expected)
    (expected_grad,) = torch.autograd.grad(expected, logits, retain_graph=True)
    (actual_grad,) = torch.autograd.grad(actual, logits)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_chunked_cross_entropy_saves_only_inputs_for_backward():
    """Chunking should not retain a full-vocabulary fp32 tensor per chunk."""
    logits = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.tensor([[1, 2, 3, 4, 5], [1, -100, 3, -100, 5]])
    saved_tensors: list[torch.Tensor] = []

    def record(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        loss = dp_utils._chunked_cross_entropy(
            logits,
            labels,
            vocab_size=7,
            ignore_index=-100,
            chunk_size=2,
        )

    assert [tensor.shape for tensor in saved_tensors] == [logits.shape, labels.shape]
    loss.backward()


def test_hooks_chunked_loss_records_fp32_logits_size():
    """Hooks mode should apply memory controls without patching Transformers globals."""
    logits = torch.randn(2, 5, 7, requires_grad=True)

    class LogitsModel:
        def __call__(self, **_model_inputs):
            return (logits,)

    trainer = object.__new__(OpacusDPTrainer)
    trainer.memory_controls = TrainingMemoryControls(
        chunked_causal_lm_loss=True,
        chunked_causal_lm_loss_tokens=2,
        debug_loss_memory=True,
    )
    trainer._peak_loss_logits_bytes = 0
    labels = torch.tensor(
        [
            [0, 1, -100, 3, -100],
            [0, 1, 2, 3, 4],
        ]
    )

    loss = trainer._compute_hooks_causal_lm_loss(LogitsModel(), {"labels": labels})

    assert loss.requires_grad
    assert trainer._peak_loss_logits_bytes == logits.numel() * 4


@pytest.mark.parametrize(
    ("compute_loss_func", "label_smoother", "message"),
    [
        (lambda *_args, **_kwargs: torch.tensor(0.0), None, "compute_loss_func"),
        (None, object(), "label smoothing"),
    ],
)
def test_hooks_rejects_nonseparable_trainer_loss_customizations(compute_loss_func, label_smoother, message):
    trainer = object.__new__(OpacusDPTrainer)
    trainer.grad_sample_mode = "hooks"
    trainer.compute_loss_func = compute_loss_func
    trainer.label_smoother = label_smoother

    with pytest.raises(ValueError, match=message):
        trainer._validate_hooks_loss_configuration()


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (SimpleNamespace(loss_type="ForSequenceClassification"), "default causal-LM loss"),
        (SimpleNamespace(loss_type="ForCausalLM", _loss_function=lambda: None), "loss_function overrides"),
    ],
)
def test_hooks_rejects_model_loss_customizations(model, message):
    trainer = object.__new__(OpacusDPTrainer)
    trainer.grad_sample_mode = "hooks"

    with pytest.raises(ValueError, match=message):
        trainer._validate_hooks_model_loss_configuration(model)


@pytest.mark.parametrize(
    "outputs",
    [
        SimpleNamespace(logits=torch.randn(1, 3, 5), aux_loss=torch.tensor(1.0)),
        (torch.tensor(1.0), torch.randn(1, 3, 5)),
    ],
)
def test_hooks_rejects_model_auxiliary_losses(outputs):
    class AuxiliaryLossModel:
        def __call__(self, **_model_inputs):
            return outputs

    trainer = object.__new__(OpacusDPTrainer)
    trainer.memory_controls = TrainingMemoryControls()

    with pytest.raises(RuntimeError, match="auxiliary loss"):
        trainer._compute_hooks_causal_lm_loss(AuxiliaryLossModel(), {"labels": torch.tensor([[1, 2, 3]])})


def test_gib_upper_bound_bucket_is_coarse():
    assert dp_utils._gib_upper_bound_bucket(0) is None
    assert dp_utils._gib_upper_bound_bucket(64 * 1024**2) == 0.125
    assert dp_utils._gib_upper_bound_bucket(int(1.25 * 1024**3)) == 2.0


def test_loss_memory_log_omits_exact_logits_shape(monkeypatch):
    logits = SimpleNamespace(
        is_cuda=True,
        device="cuda:0",
        ndim=3,
        dtype=torch.float16,
        numel=lambda: 2048,
        element_size=lambda: 2,
    )
    warnings: list[str] = []
    monkeypatch.setattr(dp_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dp_utils.torch.cuda, "mem_get_info", lambda _device: (2 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(dp_utils.torch.cuda, "memory_allocated", lambda _device: int(1.25 * 1024**3))
    monkeypatch.setattr(dp_utils.torch.cuda, "memory_reserved", lambda _device: 4 * 1024**3)
    monkeypatch.setattr(dp_utils.logger, "warning", lambda message, *_args, **_kwargs: warnings.append(message))

    dp_utils._log_cuda_loss_memory("before_logits_float", logits, enabled=True)  # ty: ignore[invalid-argument-type] -- minimal tensor-like stub

    assert len(warnings) == 1
    assert "logits_shape" not in warnings[0]
    assert "logits_rank=3" in warnings[0]
    assert "logits_gib_bucket_le=0.125" in warnings[0]
    assert "allocated_gib_bucket_le=2.0" in warnings[0]


def test_loss_memory_controls_warn_when_ignored_in_ghost_mode(monkeypatch):
    """Ghost clipping uses its own loss wrapper, so hooks-only controls warn."""
    trainer = object.__new__(OpacusDPTrainer)
    trainer.grad_sample_mode = "ghost"
    trainer.memory_controls = TrainingMemoryControls(
        chunked_causal_lm_loss=True,
        debug_loss_memory=True,
    )

    warnings: list[str] = []
    monkeypatch.setattr(dp_utils.logger, "warning", lambda message, *_args, **_kwargs: warnings.append(message))

    trainer._warn_if_ghost_memory_controls_ignored()
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
    monkeypatch.setattr(dp_utils.logger, "warning", lambda *_args, **_kwargs: None)

    trainer._emit_training_observability(peak_vram_gb=5.0)

    event = trainer.last_training_observability
    assert event is not None
    assert event.peak_vram_gb == 5.0
    assert event.peak_loss_logits_gb_bucket_le == 4.0
    assert event.effective_batch_size == 6
