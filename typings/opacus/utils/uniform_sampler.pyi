# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Iterator

import torch
from torch.utils.data import Sampler

class UniformWithReplacementSampler(Sampler[list[int]]):
    num_samples: int
    sample_rate: float
    steps: int
    generator: torch.Generator | None
    def __init__(self, *, num_samples: int, sample_rate: float, generator: object = None) -> None: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[list[int]]: ...
