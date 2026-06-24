# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch.optim import Optimizer
from typing_extensions import override

class DPOptimizer(Optimizer):
    original_optimizer: Optimizer
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: int,
        **kwargs: Any,
    ) -> None: ...
    @override
    def step(self, closure: Any = None) -> Any: ...
    @override
    def zero_grad(self, set_to_none: bool = True) -> None: ...
    @property
    def param_groups(self) -> list[dict[str, Any]]: ...
