# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# This file has been adapted from the `dp-transformers` library.
# Original source: https://github.com/microsoft/dp-transformers/blob/main/research/fine_tune_llm_w_qlora/linear.py
# See THIRD_PARTY.md for the original MIT license terms.

# This module is included for training with Opacus so that per sample gradients
# are correctly calculated. It is imported for effects, never directly used.
# Convert activations and backprops to float for per-sample gradient
# computation. During mixed precision training it is possible that the
# activations and/or backprops are not in full precision.
from typing import Dict, List

import torch
import torch.nn as nn
from opacus.grad_sample.utils import register_grad_sampler
from opt_einsum import contract


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activation = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        gs = contract("n...i,n...j->nij", backprops.float(), activation.float())
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = contract("n...k->nk", backprops.float())
    return ret
