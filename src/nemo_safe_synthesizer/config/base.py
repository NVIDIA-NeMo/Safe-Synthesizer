# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    """Base model for all NeMo Safe Synthesizer configuration and result models that do not use Parameters."""

    model_config = pydantic_model_config

    def dict(self, **kwargs: Any) -> dict[str, Any]:
        return self.model_dump(**kwargs)


class LRScheduler(str, Enum):
    COSINE = "cosine"
    LINEAR = "linear"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
