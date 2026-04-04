# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

class Accountant:
    def get_epsilon(self, delta: float, **kwargs: Any) -> float: ...

class RDPAccountant(Accountant):
    def step(self, *, noise_multiplier: float, sample_rate: float) -> None: ...
    def get_epsilon(self, delta: float, **kwargs: Any) -> float: ...
