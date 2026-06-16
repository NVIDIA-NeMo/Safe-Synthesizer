# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI regression tests for the vLLM benchmark tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from click.testing import CliRunner

from nemo_safe_synthesizer.generation.vllm_benchmark import BenchmarkCandidate, BenchmarkOutput, CandidateMetrics


def _load_tool():
    tool_path = Path(__file__).resolve().parents[2] / "tools" / "vllm_benchmark.py"
    spec = importlib.util.spec_from_file_location("vllm_benchmark_tool", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()
cli = tool.cli


def _metric(name: str, condition: str, *, eff: float, wall: float, bracket: int) -> CandidateMetrics:
    return CandidateMetrics(
        name=name,
        raw_tok_s=eff / 0.99,
        acceptance_rate=0.99,
        effective_tok_s=eff,
        ttft_p50_ms=0.0,
        ttft_p99_ms=0.0,
        prompts_attempted=4,
        prompts_accepted=4,
        total_output_tokens=1000,
        total_wall_seconds=wall,
        condition_label=condition,
        bracket_position=bracket,
    )


def _write_output(path: Path) -> None:
    candidate_runs = [
        *(_metric(f"baseline_{i}", "baseline", eff=1500.0 + i, wall=130.0 + i, bracket=2 * i) for i in range(6)),
        *(
            _metric(f"spec_ngram_{i}", "spec_ngram", eff=1700.0 + i, wall=115.0 + i, bracket=2 * i + 1)
            for i in range(6)
        ),
    ]
    path.write_text(
        BenchmarkOutput(corpus_run_id="cli-test", corpus_size=4, candidates=candidate_runs).model_dump_json(),
        encoding="utf-8",
    )


def test_analyze_accepts_min_runs_option(tmp_path: Path) -> None:
    """The public min-run option should bind to the analyzer callback."""
    _write_output(tmp_path / "out.json")

    result = CliRunner().invoke(
        cli,
        ["analyze", str(tmp_path), "--cluster-signal", "wall_seconds", "--min-runs-per-condition", "6"],
        color=False,
    )

    assert result.exit_code == 0, result.output
    assert "Cluster-conditioned analysis" in result.output


def test_hidden_run_candidate_command_is_available() -> None:
    """The subprocess wrapper should target the repo-local tool command."""
    result = CliRunner().invoke(cli, ["_run-candidate", "--help"], color=False)

    assert result.exit_code == 0, result.output
    assert "Run one benchmark candidate" in result.output


def test_truncate_stderr_keeps_tail() -> None:
    long = "x" * 600
    truncated = tool._truncate_stderr(long, limit=100)

    assert len(truncated) <= 100
    assert truncated.endswith("xxxxx")


def test_parse_error_class_finds_exception_name() -> None:
    assert tool._parse_error_class("Traceback (most recent call last):\n...\nValueError: bad") == "ValueError"
    assert tool._parse_error_class("") == "Error"


def test_subprocess_result_success_shape() -> None:
    metrics = CandidateMetrics(
        name="t",
        raw_tok_s=1.0,
        acceptance_rate=0.9,
        effective_tok_s=0.9,
        ttft_p50_ms=0.0,
        ttft_p99_ms=0.0,
        prompts_attempted=10,
        prompts_accepted=9,
        total_output_tokens=100,
        total_wall_seconds=1.0,
    )

    result = tool.SubprocessRunResult(metrics=metrics)

    assert result.metrics is not None and result.error is None


def test_subprocess_result_failure_shape() -> None:
    result = tool.SubprocessRunResult(error="exit 1", error_class="RuntimeError")

    assert result.metrics is None and result.error_class == "RuntimeError"


def test_subprocess_runner_targets_tool_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: Any) -> Any:
        calls.append(args)
        result_path = Path(args[args.index("--result-out") + 1])
        metrics = CandidateMetrics(
            name="baseline",
            raw_tok_s=1.0,
            acceptance_rate=1.0,
            effective_tok_s=1.0,
            ttft_p50_ms=0.0,
            ttft_p99_ms=0.0,
            prompts_attempted=1,
            prompts_accepted=1,
            total_output_tokens=1,
            total_wall_seconds=1.0,
        )
        result_path.write_text(metrics.model_dump_json(), encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(tool.subprocess, "run", fake_run)

    result = tool.run_benchmark_in_subprocess(BenchmarkCandidate(name="baseline"), tmp_path / "trace.jsonl")

    assert result.metrics is not None
    assert calls[0][0] == sys.executable
    assert calls[0][1].endswith("tools/vllm_benchmark.py")
    assert calls[0][2] == "_run-candidate"
