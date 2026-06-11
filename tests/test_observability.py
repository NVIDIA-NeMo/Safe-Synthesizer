# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_safe_synthesizer.observability import NvmlPeakSampler, _default_nvml_device_index, read_loadavg


@pytest.mark.skipif(not Path("/proc/loadavg").exists(), reason="/proc/loadavg is Linux-specific")
def test_read_loadavg_returns_float_triple_on_linux() -> None:
    loadavg = read_loadavg()

    assert loadavg is not None
    assert len(loadavg) == 3
    assert all(isinstance(value, float) for value in loadavg)


@pytest.mark.parametrize(
    ("visible_devices", "expected"),
    [
        (None, 0),
        ("", 0),
        ("2,3", 2),
        (" GPU-abc123 ", 0),
    ],
)
def test_default_nvml_device_index_reads_first_cuda_visible_device(
    monkeypatch: pytest.MonkeyPatch,
    visible_devices: str | None,
    expected: int,
) -> None:
    if visible_devices is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible_devices)

    assert _default_nvml_device_index() == expected


def test_nvml_peak_sampler_degrades_when_pynvml_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)

    with NvmlPeakSampler(interval_seconds=0.001) as sampler:
        assert sampler.peak_gb is None


def test_nvml_peak_sampler_shuts_down_when_handle_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeNvmlError(Exception):
        pass

    fake_pynvml = SimpleNamespace(shutdown_calls=0)

    def nvml_init() -> None:
        return None

    def get_handle_by_index(_device_index: int) -> object:
        raise FakeNvmlError("bad device")

    def nvml_shutdown() -> None:
        fake_pynvml.shutdown_calls += 1

    fake_pynvml.NVMLError = FakeNvmlError
    fake_pynvml.nvmlInit = nvml_init
    fake_pynvml.nvmlDeviceGetHandleByIndex = get_handle_by_index
    fake_pynvml.nvmlShutdown = nvml_shutdown
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    with NvmlPeakSampler(interval_seconds=0.001) as sampler:
        assert sampler.peak_gb is None

    assert fake_pynvml.shutdown_calls == 1
