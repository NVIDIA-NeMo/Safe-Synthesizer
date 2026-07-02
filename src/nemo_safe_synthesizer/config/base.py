# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Pydantic base model and enumerations for NeMo Safe Synthesizer configs."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = ["NSSBaseModel", "LRScheduler", "pydantic_model_config"]

pydantic_model_config = ConfigDict(
    arbitrary_types_allowed=True,
    validation_error_cause=True,
    from_attributes=True,
    validate_default=True,
    protected_namespaces=(),
    # This is needed to ensure we don't generate separate
    # -Input/-Output schemas for the child objects when used in the NMP sdk.
    json_schema_mode_override="validation",
)


class NSSBaseModel(BaseModel):
    """Base model for all Safe Synthesizer configuration and result models.

    Applies ``pydantic_model_config`` (strict schema validation, attribute
    construction, and ``"validation"``-only JSON schema mode) so that
    every subclass inherits consistent Pydantic behavior.
    """

    model_config = pydantic_model_config

    def dict(self, **kwargs: Any) -> dict[str, Any]:  # ty: ignore[invalid-type-form] -- method name shadows builtin dict; backward-compat shim for model_dump
        """Return a dict representation via ``model_dump`` for backward compatibility."""
        return self.model_dump(**kwargs)


class LRScheduler(str, Enum):
    """Learning-rate scheduler names accepted by the training pipeline."""

    COSINE = "cosine"
    LINEAR = "linear"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
