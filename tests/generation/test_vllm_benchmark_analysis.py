# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cluster-conditioned analyzer.

Scope: contracts consumers depend on — cluster partitioning produces
the expected shape, refusals fire when sample size is too small,
effect-size CI brackets behave correctly, JSON output round-trips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from nemo_safe_synthesizer.generation.vllm_benchmark import (
    BenchmarkOutput,
    CandidateMetrics,
)
from nemo_safe_synthesizer.generation.vllm_benchmark_analysis import (
    MIN_CELLS_PER_CONDITION,
    AnalysisReport,
    _effect_size,
    _welch_ttest_ci,
    analyze,
)


def _metric(name: str, condition: str, *, eff: float, accept: float, wall: float, bracket: int = 0) -> CandidateMetrics:
    """Build a CandidateMetrics with minimal scaffolding — used to seed the analyzer."""
    return CandidateMetrics(
        name=name,
        raw_tok_s=eff / max(accept, 0.01),
        acceptance_rate=accept,
        effective_tok_s=eff,
        ttft_p50_ms=0.0,
        ttft_p99_ms=0.0,
        prompts_attempted=143,
        prompts_accepted=int(143 * accept),
        total_output_tokens=100000,
        total_wall_seconds=wall,
        condition_label=condition,
        bracket_position=bracket,
    )


def _write_output_dir(tmp_path: Path, cells: list[CandidateMetrics]) -> Path:
    """Write a single BenchmarkOutput JSON to ``tmp_path / out.json``."""
    out = BenchmarkOutput(corpus_run_id="r1", corpus_size=143, candidates=cells)
    (tmp_path / "out.json").write_text(out.model_dump_json(), encoding="utf-8")
    return tmp_path


@pytest.fixture
def synthetic_sweep_dir(tmp_path: Path) -> Path:
    """6 baselines + 6 spec_ngram cells, both with realistic-noise spread."""
    cells = [
        # Baselines: ~1500 eff_tok_s, ~0.99 acceptance.
        *(_metric(f"baseline_{i}", "baseline", eff=1500 + i * 5, accept=0.99, wall=130.0, bracket=2 * i) for i in range(6)),
        # spec_ngram: ~1700 eff_tok_s (+~13%), similar acceptance.
        *(_metric(f"spec_{i}", "spec_ngram", eff=1700 + i * 5, accept=0.99, wall=115.0, bracket=2 * i + 1) for i in range(6)),
    ]
    return _write_output_dir(tmp_path, cells)


# ---------------------------------------------------------------------------
# Welch CI math
# ---------------------------------------------------------------------------


class TestWelchTtestCi:
    def test_clear_difference_excludes_zero(self) -> None:
        """Genuine difference in means → CI doesn't bracket 0."""
        res = _welch_ttest_ci([1700, 1720, 1690, 1710, 1705, 1715], [1500, 1510, 1495, 1505, 1502, 1498])
        assert res is not None
        mean_diff, ci_low, ci_high, _df = res
        assert mean_diff > 0
        assert ci_low > 0  # CI excludes 0

    def test_identical_means_brackets_zero(self) -> None:
        """Same means + nonzero variance → CI brackets 0."""
        res = _welch_ttest_ci([1500, 1510, 1495, 1505, 1502, 1498], [1500, 1510, 1495, 1505, 1502, 1498])
        assert res is not None
        mean_diff, ci_low, ci_high, _df = res
        assert mean_diff == 0
        assert ci_low < 0 < ci_high

    @pytest.mark.parametrize(
        ("cand", "base"),
        [
            ([1500], [1502]),  # too few candidate observations
            ([1500, 1500], [1502]),  # too few baseline observations
            ([1500, 1500], [1500, 1500]),  # both stddevs zero
        ],
    )
    def test_underdetermined_returns_none(self, cand: list[float], base: list[float]) -> None:
        """Degraded mode: returns None instead of nan/inf/raising."""
        assert _welch_ttest_ci(cand, base) is None


# ---------------------------------------------------------------------------
# Full analyze pipeline
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_partitions_and_aggregates(self, synthetic_sweep_dir: Path) -> None:
        report = analyze(synthetic_sweep_dir, cluster_signal="wall_seconds")
        assert report.n_cells == 12
        # Two conditions present.
        labels = {agg.condition_label for agg in report.condition_aggregates}
        assert labels == {"baseline", "spec_ngram"}
        # Each condition has the expected pooled aggregate.
        spec_agg = next(agg for agg in report.condition_aggregates if agg.condition_label == "spec_ngram")
        assert spec_agg.n_cells == 6
        assert spec_agg.pooled_mean_effective_tok_s == pytest.approx(1712.5, abs=0.1)

    def test_emits_effect_size_for_non_baseline_conditions(self, synthetic_sweep_dir: Path) -> None:
        """spec_ngram vs baseline gets an effect size; baseline itself doesn't (no self-comparison)."""
        report = analyze(synthetic_sweep_dir)
        spec_agg = next(agg for agg in report.condition_aggregates if agg.condition_label == "spec_ngram")
        baseline_agg = next(agg for agg in report.condition_aggregates if agg.condition_label == "baseline")
        assert spec_agg.pooled_effect_size_vs_baseline is not None
        assert baseline_agg.pooled_effect_size_vs_baseline is None
        # The Δ should be roughly +200 tok/s, CI excludes 0.
        es = spec_agg.pooled_effect_size_vs_baseline
        assert es.delta_absolute > 0
        assert es.ci95_low > 0

    def test_refuses_aggregates_below_min_cells(self, tmp_path: Path) -> None:
        """Conditions with N<6 land in refusals, not aggregates."""
        # 6 baselines + only 3 candidate cells → spec_ngram should be refused.
        cells = [
            *(_metric(f"baseline_{i}", "baseline", eff=1500.0, accept=0.99, wall=130.0) for i in range(6)),
            *(_metric(f"spec_{i}", "spec_ngram", eff=1700.0, accept=0.99, wall=115.0) for i in range(3)),
        ]
        out_dir = _write_output_dir(tmp_path, cells)
        report = analyze(out_dir)
        labels = {agg.condition_label for agg in report.condition_aggregates}
        assert "baseline" in labels
        assert "spec_ngram" not in labels
        assert any("spec_ngram" in r for r in report.refusals)

    def test_report_round_trips_through_json(self, synthetic_sweep_dir: Path) -> None:
        report = analyze(synthetic_sweep_dir)
        rt = AnalysisReport.model_validate_json(report.model_dump_json())
        assert rt == report

    def test_load_failure_raises(self, tmp_path: Path) -> None:
        """Empty output dir raises with a clear message."""
        with pytest.raises(ValueError, match="No BenchmarkOutput JSONs"):
            analyze(tmp_path)
