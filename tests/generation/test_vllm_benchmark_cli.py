# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI regression tests for the vLLM benchmark tool."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from click.testing import CliRunner

from nemo_safe_synthesizer.generation.vllm_benchmark import BenchmarkOutput, CandidateMetrics


def _load_cli():
    tool_path = Path(__file__).resolve().parents[2] / "tools" / "vllm_benchmark.py"
    spec = importlib.util.spec_from_file_location("vllm_benchmark_tool", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.cli


cli = _load_cli()


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
