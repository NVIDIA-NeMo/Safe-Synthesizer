# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from nemo_safe_synthesizer.config.training import TrainingHyperparams


def _fake_torch(available: bool = True, capability: tuple[int, int] = (9, 0)):
    class _FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return available

        @staticmethod
        def get_device_capability(_idx: int = 0) -> tuple[int, int]:
            return capability

    class _FakeTorch:
        cuda = _FakeCuda

    return _FakeTorch


class TestAttnFallback:
    def test_non_default_user_value_is_untouched(self):
        with patch("platform.machine", return_value="x86_64"):
            hp = TrainingHyperparams(attn_implementation="flash_attention_2")
        assert hp.attn_implementation == "flash_attention_2"

    def test_aarch64_falls_back_to_sdpa(self):
        with patch("platform.machine", return_value="aarch64"):
            hp = TrainingHyperparams()
        assert hp.attn_implementation == "sdpa"

    def test_sm100_blackwell_falls_back_to_sdpa(self):
        fake = _fake_torch(available=True, capability=(10, 0))
        with patch("platform.machine", return_value="x86_64"), patch.dict(
            "sys.modules", {"torch": fake}
        ):
            hp = TrainingHyperparams()
        assert hp.attn_implementation == "sdpa"

    def test_hopper_sm90_stays_on_flash_attn(self):
        fake = _fake_torch(available=True, capability=(9, 0))
        with patch("platform.machine", return_value="x86_64"), patch.dict(
            "sys.modules", {"torch": fake}
        ):
            hp = TrainingHyperparams()
        assert hp.attn_implementation == "kernels-community/vllm-flash-attn3"

    def test_no_cuda_falls_back_to_sdpa(self):
        fake = _fake_torch(available=False, capability=(0, 0))
        with patch("platform.machine", return_value="x86_64"), patch.dict(
            "sys.modules", {"torch": fake}
        ):
            hp = TrainingHyperparams()
        assert hp.attn_implementation == "sdpa"

    def test_torch_import_failure_does_not_block(self):
        with patch("platform.machine", return_value="x86_64"), patch.dict(
            "sys.modules", {"torch": None}
        ):
            # A None entry in sys.modules forces ImportError on import.
            hp = TrainingHyperparams()
        assert hp.attn_implementation == "kernels-community/vllm-flash-attn3"

    def test_explicit_sdpa_on_blackwell_stays_sdpa(self):
        fake = _fake_torch(available=True, capability=(10, 0))
        with patch("platform.machine", return_value="x86_64"), patch.dict(
            "sys.modules", {"torch": fake}
        ):
            hp = TrainingHyperparams(attn_implementation="sdpa")
        assert hp.attn_implementation == "sdpa"


def test_fallback_message_logged(caplog):
    import logging

    with patch("platform.machine", return_value="aarch64"), caplog.at_level(
        logging.INFO, logger="nemo_safe_synthesizer.config.training"
    ):
        TrainingHyperparams()
    assert any("Auto-overriding" in rec.message for rec in caplog.records)
