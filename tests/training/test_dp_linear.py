# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch
from torch import nn

from nemo_safe_synthesizer.privacy.dp_transformers import linear as linear_mod


def test_compute_linear_grad_sample_returns_weight_and_bias_gradients():
    layer = nn.Linear(3, 2)
    activation = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    backprops = torch.tensor(
        [
            [0.5, 1.5],
            [2.0, 3.0],
        ]
    )

    result = linear_mod.compute_linear_grad_sample(layer, [activation], backprops)

    assert set(result) == {layer.weight, layer.bias}
    torch.testing.assert_close(result[layer.weight], torch.einsum("ni,nj->nij", backprops, activation))
    torch.testing.assert_close(result[layer.bias], backprops)


def test_contract_tensor_rejects_non_tensor_result(monkeypatch: pytest.MonkeyPatch):
    def fake_contract(*_args, **_kwargs) -> int:
        return 1

    monkeypatch.setattr(linear_mod, "contract", fake_contract)

    with pytest.raises(TypeError, match=r"expected opt_einsum\.contract"):
        linear_mod._contract_tensor("n->n", torch.ones(1))


def test_linear_parameter_rejects_plain_tensor():
    with pytest.raises(TypeError, match=r"expected nn\.Linear parameter"):
        linear_mod._linear_parameter(torch.ones(1))
