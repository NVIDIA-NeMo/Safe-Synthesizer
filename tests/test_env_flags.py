# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared environment-flag parsing helpers."""

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.utils import env_flag_is_true


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
        ("", False),
    ],
)
def test_env_flag_is_true(value: str, expected: bool, monkeypatch):
    monkeypatch.setenv("LOCAL_FILES_ONLY", value)
    assert env_flag_is_true("LOCAL_FILES_ONLY") is expected


def test_env_flag_is_true_unset_uses_default(monkeypatch):
    monkeypatch.delenv("LOCAL_FILES_ONLY", raising=False)
    assert env_flag_is_true("LOCAL_FILES_ONLY") is False
    assert env_flag_is_true("LOCAL_FILES_ONLY", default=True) is True
