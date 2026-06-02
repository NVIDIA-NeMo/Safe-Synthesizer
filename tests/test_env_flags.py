# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for shared environment-flag parsing helpers."""

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.utils import env_flag_is_true, hf_offline_enabled

_PROBE_VAR = "NSS_TEST_FLAG"


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
    monkeypatch.setenv(_PROBE_VAR, value)
    assert env_flag_is_true(_PROBE_VAR) is expected


def test_env_flag_is_true_unset_uses_default(monkeypatch):
    monkeypatch.delenv(_PROBE_VAR, raising=False)
    assert env_flag_is_true(_PROBE_VAR) is False
    assert env_flag_is_true(_PROBE_VAR, default=True) is True


@pytest.mark.parametrize("offline_var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
def test_hf_offline_enabled_true_for_either_var(offline_var: str, monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setenv(offline_var, "1")
    assert hf_offline_enabled() is True


def test_hf_offline_enabled_false_when_unset(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert hf_offline_enabled() is False
