# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the training-observability event schema and wandb sink."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.training.training_observability import (
    TRAINING_COMPLETE_EVENT,
    TrainingObservability,
)

pytestmark = pytest.mark.unit


def test_event_name_constant():
    assert TRAINING_COMPLETE_EVENT == "training.complete"


def test_all_fields_default_to_none():
    event = TrainingObservability()
    assert event.model_dump() == {
        "peak_vram_gb": None,
        "peak_loss_logits_gb": None,
        "per_device_train_batch_size": None,
        "gradient_accumulation_steps": None,
        "effective_batch_size": None,
        "grad_sample_mode": None,
    }


def test_extra_fields_forbidden():
    with pytest.raises(ValidationError):
        TrainingObservability(unexpected=1)  # ty: ignore[unknown-argument] -- asserting extra=forbid


def test_to_wandb_payload_drops_none_and_prefixes():
    event = TrainingObservability(
        peak_vram_gb=2.5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        effective_batch_size=8,
        grad_sample_mode="ghost",
    )
    payload = event.to_wandb_payload()
    assert payload == {
        "training/peak_vram_gb": 2.5,
        "training/per_device_train_batch_size": 2,
        "training/gradient_accumulation_steps": 4,
        "training/effective_batch_size": 8,
        "training/grad_sample_mode": "ghost",
    }
    # peak_loss_logits_gb was None and must be dropped, not emitted as None.
    assert "training/peak_loss_logits_gb" not in payload


def test_to_wandb_payload_custom_prefix():
    event = TrainingObservability(peak_vram_gb=1.0)
    assert event.to_wandb_payload(prefix="dp_train") == {"dp_train/peak_vram_gb": 1.0}


def test_log_observability_event_is_noop_without_active_run(monkeypatch):
    """The generic wandb sink must no-op (not raise, not log) when no run is active."""
    import nemo_safe_synthesizer.cli.wandb_setup as wandb_setup

    monkeypatch.setattr(wandb_setup.wandb, "run", None)

    logged: list[dict] = []
    monkeypatch.setattr(wandb_setup.wandb, "log", lambda payload: logged.append(payload))

    wandb_setup.log_observability_event(TrainingObservability(peak_vram_gb=1.0), prefix="training")
    assert logged == []


def test_log_observability_event_logs_to_active_run(monkeypatch):
    import nemo_safe_synthesizer.cli.wandb_setup as wandb_setup

    monkeypatch.setattr(wandb_setup.wandb, "run", object())
    logged: list[dict] = []
    monkeypatch.setattr(wandb_setup.wandb, "log", lambda payload: logged.append(payload))

    wandb_setup.log_observability_event(TrainingObservability(peak_vram_gb=1.0), prefix="training")
    assert logged == [{"training/peak_vram_gb": 1.0}]


def test_emit_training_observability_assembles_event_from_trainer_state():
    """``OpacusDPTrainer._emit_training_observability`` builds the event from
    resolved batching + grad-sample mode and stashes it on
    ``last_training_observability`` (the backend, not the trainer, forwards it
    to wandb). Tested via the unbound method on a lightweight stub to avoid a
    full training run.
    """
    from types import SimpleNamespace

    from nemo_safe_synthesizer.privacy.dp_transformers import dp_utils

    stub = SimpleNamespace(
        args=SimpleNamespace(per_device_train_batch_size=2, gradient_accumulation_steps=4),
        grad_sample_mode="ghost",
        _peak_loss_logits_bytes=1024**3,  # 1 GiB recorded by the probe
        last_training_observability=None,
    )
    dp_utils.OpacusDPTrainer._emit_training_observability(stub, peak_vram_gb=3.5)  # ty: ignore[invalid-argument-type] -- duck-typed stub

    event = stub.last_training_observability
    assert event is not None
    assert event.peak_vram_gb == 3.5
    assert event.peak_loss_logits_gb == pytest.approx(1.0)
    assert event.per_device_train_batch_size == 2
    assert event.gradient_accumulation_steps == 4
    assert event.effective_batch_size == 8
    assert event.grad_sample_mode == "ghost"


def test_emit_training_observability_omits_peak_loss_logits_when_unrecorded():
    from types import SimpleNamespace

    from nemo_safe_synthesizer.privacy.dp_transformers import dp_utils

    stub = SimpleNamespace(
        args=SimpleNamespace(per_device_train_batch_size=1, gradient_accumulation_steps=1),
        grad_sample_mode="hooks",
        _peak_loss_logits_bytes=0,  # debug probe was off -> nothing recorded
        last_training_observability=None,
    )
    dp_utils.OpacusDPTrainer._emit_training_observability(stub, peak_vram_gb=None)  # ty: ignore[invalid-argument-type] -- duck-typed stub

    event = stub.last_training_observability
    assert event is not None
    assert event.peak_loss_logits_gb is None
    assert event.peak_vram_gb is None
