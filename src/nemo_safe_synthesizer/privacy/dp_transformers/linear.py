# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0 AND MIT

# This file has been adapted from the `dp-transformers` library.
# Original source: https://github.com/microsoft/dp-transformers/blob/main/research/fine_tune_llm_w_qlora/linear.py
# See THIRD_PARTY.md for the original MIT license terms.


"""Per-sample gradient hook for ``nn.Linear`` (Opacus).

Registering this module with Opacus ensures per-sample gradients are computed
correctly for Linear layers. Import this module for its side effect; do not
call ``compute_linear_grad_sample`` directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from opacus.grad_sample.utils import register_grad_sampler
from opt_einsum import contract


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample(
    layer: nn.Linear, activations: list[torch.Tensor], backprops: torch.Tensor
) -> dict[nn.Parameter, torch.Tensor]:
    """Compute per-sample gradients for an ``nn.Linear`` layer.

    Used by Opacus for correct per-sample gradient accumulation. Converts
    activations and backprops to float for mixed-precision compatibility.

    Args:
        layer: The Linear layer being sampled.
        activations: List of activation tensors from the forward pass.
        backprops: Backpropagated gradient tensor.

    Returns:
        Dictionary mapping each trainable parameter (weight, bias) to its
        per-sample gradient tensor of shape ``(batch, ...)``.
    """
    activation = activations[0]
    ret = {}
    if layer.weight.requires_grad:
        gs = contract("n...i,n...j->nij", backprops.float(), activation.float())
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = contract("n...k->nk", backprops.float())
    return ret
