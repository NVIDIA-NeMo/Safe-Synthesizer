# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch.nn as nn
from opacus import optimizers as optimizers

class GradSampleModule(nn.Module):
    def __init__(self, m: nn.Module, **kwargs: Any) -> None: ...
