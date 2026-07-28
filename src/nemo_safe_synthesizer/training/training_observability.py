# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training observability event schema.

Schema-frozen training-observability event emitted once per fit by
``OpacusDPTrainer.train()`` and consumed by downstream surfaces (structured
logs, wandb). Mirrors the generation-side ``GenerationObservability`` pattern:
every field is optional so producers populate what they can capture and
``extra="forbid"`` forces producers to update the schema when adding fields.

The shared NVML/loadavg sampling primitives this event reports live in
:mod:`nemo_safe_synthesizer.observability` so both the training and generation
backends draw from one implementation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

#: Structured-log key / wandb namespace for the training-complete event.
TRAINING_COMPLETE_EVENT = "training.complete"


class TrainingObservability(BaseModel):
    """One ``training.complete`` event payload.

    Emitted once when a training run finishes (success or failure). Consumed
    by structured-log routing (``logger.runtime.info``) and wandb (when a run
    is active). Every measurement field is optional; ``None`` means "not
    captured on this run" (e.g. NVML unavailable, or the diagnostic was off).
    """

    model_config = ConfigDict(extra="forbid")

    peak_vram_gb: float | None = Field(
        default=None,
        description=(
            "Peak device-wide VRAM usage in GiB, sampled by NVML across the whole training "
            "run. ``None`` when NVML is unavailable. Device-wide reading; on a shared GPU it "
            "includes other processes."
        ),
    )
    peak_loss_logits_gb_bucket_le: float | None = Field(
        default=None,
        description=(
            "Coarse power-of-two GiB upper-bound bucket for the peak fp32 causal-LM logits "
            "tensor size seen by the loss function. This avoids exporting exact padded "
            "sequence-shape-derived memory. Populated only when "
            "training.memory.debug_loss_memory is enabled; ``None`` otherwise."
        ),
    )
    per_device_train_batch_size: int | None = Field(
        default=None,
        description="Resolved physical per-device microbatch size passed to the Trainer.",
    )
    gradient_accumulation_steps: int | None = Field(
        default=None,
        description="Resolved gradient accumulation steps passed to the Trainer.",
    )
    effective_batch_size: int | None = Field(
        default=None,
        description="Resolved effective (logical) batch size = per_device_train_batch_size * gradient_accumulation_steps.",
    )
    grad_sample_mode: str | None = Field(
        default=None,
        description="Opacus per-sample gradient mode used for DP training (``hooks`` or ``ghost``); ``None`` for non-DP runs.",
    )

    def to_wandb_payload(self, prefix: str = "training") -> dict[str, Any]:
        """Flatten this event into a wandb-friendly ``wandb.log(...)`` dict.

        Drops ``None`` values (wandb would drop them anyway) and namespaces
        every key under ``prefix`` so training events don't collide with other
        wandb metrics in the same run.
        """
        payload: dict[str, Any] = {}
        for field_name in self.__class__.model_fields:
            value = getattr(self, field_name)
            if value is not None:
                payload[f"{prefix}/{field_name}"] = value
        return payload
