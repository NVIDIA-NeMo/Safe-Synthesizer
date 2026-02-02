# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

# Skip all tests in this module if torch is not available
pytest.importorskip("torch", reason="torch is required for these tests (install with: uv sync --extra cpu)")

from nemo_safe_synthesizer.pii_replacer.data_editor.filters import partial_mask


def test_partial_mask():
    assert partial_mask("foobar", 0, "MASK", 0) == "MASK"
    assert partial_mask("foobar", 0, "XXX", 2) == "XXXar"
    assert partial_mask("foobar", 2, "XXX", 0) == "foXXX"
    assert partial_mask("john", 1, "XXX", 1) == "jXXXn"
    assert partial_mask("john", 10, "XXX", 10) == "jXXXn"
    assert partial_mask("sue", 3, "XXX", 3) == "XXXe"
    assert partial_mask("sue", 3, "XXX", 3, tolerance=0) == "sXXXe"
    assert partial_mask("sami", 3, "XXX", 5) == "XXXmi"
    assert partial_mask("sami", 2, "XXX", 2) == "sXXXi"
    assert partial_mask(12345, 1, "NUM", 1) == "1NUM5"
    assert partial_mask("bob", 1, "XXX", 1, tolerance=10) == "XXX"
    assert partial_mask("bob", 1, "XXX", 1, tolerance=0) == "bXXXb"
    assert partial_mask("bob", 1, "XXX", 1, tolerance=3) == "XXX"

    with pytest.raises(ValueError, match="mask value"):
        partial_mask("foobar", 0, "", 0)

    with pytest.raises(ValueError, match="must be greater"):
        partial_mask("foobar", 0, "mask", 0, tolerance=-1)
