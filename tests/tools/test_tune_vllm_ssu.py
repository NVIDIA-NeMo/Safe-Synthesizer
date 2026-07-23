# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the standalone Nemotron vLLM SSU tuning tool."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool() -> ModuleType:
    tool_path = Path(__file__).parents[2] / "tools" / "nemotron" / "tune_vllm_ssu.py"
    spec = importlib.util.spec_from_file_location("tune_vllm_ssu", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {tool_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load_tool()
CONFIG_FILENAME = tool.CONFIG_FILENAME


def _write_config(folder: Path, payload: object) -> Path:
    path = folder / CONFIG_FILENAME
    path.write_text(json.dumps(payload))
    return path


def _valid_payload() -> dict[str, object]:
    return {
        "triton_version": "3.6.0",
        **{
            str(effective_batch): {"BLOCK_SIZE_M": 4, "num_warps": 1}
            for effective_batch in tool.EXPECTED_EFFECTIVE_BATCHES
        },
    }


def test_validate_exact_a100_nemotron_config(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["128"] = {"BLOCK_SIZE_M": 8, "num_warps": 4}
    path = _write_config(tmp_path, payload)

    assert tool.validate_config(path)["128"] == {"BLOCK_SIZE_M": 8, "num_warps": 4}
    assert tool.expected_config_path(tmp_path) == path


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"triton_version": "3.6.0"},
        {"triton_version": "3.6.0", "bad": {"BLOCK_SIZE_M": 4, "num_warps": 4}},
        {"triton_version": "3.6.0", "128": {"BLOCK_SIZE_M": 256, "num_warps": 4}},
        {"triton_version": "3.6.0", "128": {"BLOCK_SIZE_M": 4, "num_warps": 16}},
    ],
)
def test_validate_rejects_incomplete_or_invalid_config(tmp_path: Path, payload: object) -> None:
    with pytest.raises(ValueError):
        tool.validate_config(_write_config(tmp_path, payload))


@pytest.mark.parametrize(
    "entry",
    [
        {"BLOCK_SIZE_M": 256, "num_warps": 4},
        {"BLOCK_SIZE_M": 4, "num_warps": 16},
    ],
)
def test_validate_rejects_invalid_entry_in_complete_grid(tmp_path: Path, entry: object) -> None:
    payload = _valid_payload()
    payload["128"] = entry

    with pytest.raises(ValueError, match="invalid"):
        tool.validate_config(_write_config(tmp_path, payload))


def test_validate_rejects_wrong_target_filename(tmp_path: Path) -> None:
    path = tmp_path / "headdim=64,dstate=128,device_name=NVIDIA_A100-SXM4-80GB,cache_dtype=float32.json"
    path.write_text(json.dumps({"triton_version": "3.6.0", "128": {"BLOCK_SIZE_M": 4, "num_warps": 4}}))

    with pytest.raises(ValueError, match="expected filename"):
        tool.validate_config(path)


def test_shell_export_quotes_user_selected_folder(tmp_path: Path) -> None:
    folder = tmp_path / "tuned configs"

    assert tool.shell_export(folder) == f"export VLLM_TUNED_CONFIG_FOLDER='{folder}'"


def test_benchmark_command_covers_effective_batches_through_65536(tmp_path: Path) -> None:
    command = tool._benchmark_command(tmp_path / "benchmark.py", tmp_path, num_iters=100)

    batch_index = command.index("--batch-sizes")
    head_index = command.index("--nheads")
    assert command[batch_index + 1 : head_index] == ["1", "2", "8", "16", "32", "64", "128", "256", "512"]
    assert command[head_index + 1 : command.index("--num-iters")] == ["128"]
