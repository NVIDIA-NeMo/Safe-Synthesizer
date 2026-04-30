# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for llm.utils helpers."""

from pathlib import Path

import pytest

from nemo_safe_synthesizer.llm.utils import trust_remote_code_for_model


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("nvidia/Nemotron-Mini-4B-Instruct", True),
        ("nvidia/some-model", True),
        ("gretel/tabulargemma-2b", False),
        ("meta-llama/Llama-3.2-1B-Instruct", False),
        ("/models/my-local-model", False),
        ("", False),
        ("nvidia", False),
        (Path("/home/user/models/nvidia/Nemotron-Mini-4B-Instruct"), False),
        (
            Path("/home/user/.cache/huggingface/hub/models--nvidia--Nemotron-Mini-4B-Instruct/snapshots/abc123"),
            True,
        ),
    ],
)
def test_trust_remote_code_for_model(model_name: str | Path, expected: bool) -> None:
    assert trust_remote_code_for_model(model_name) is expected
