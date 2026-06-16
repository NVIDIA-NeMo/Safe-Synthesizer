# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cluster-conditioned analysis for benchmark output.

This stack does not produce trustworthy single-run measurements.
Pooled cross-run CoV runs ~5-10%; in-cluster CoV is ~3%. This module
partitions candidate runs by a per-dataset cluster signal, then reports
per-condition aggregates both pooled and within-cluster, so the
operator can see which clusters contained the candidate's samples
and trust the in-cluster delta rather than the noisier pooled delta.

Cluster signal per workload shape:

- ``wall_seconds`` for short-context workloads (bike_sales-shape) -
  bimodality is load-driven; partitioning on wall_seconds separates
  the fast vs normal-load clusters.
- ``acceptance_rate`` for long-output workloads (call_transcripts-
  shape) - bimodality is RNG/scheduler driven; partitioning on
  acceptance_rate separates the high vs low cluster the seed-pin
  validation found persists even with seed=42.
- ``auto`` - picks whichever signal has higher pooled CoV.

Cluster count is selected via silhouette score (post-hoc, k in [2, 4]).
Refuses to compute delta-style aggregates when a condition has fewer than
:data:`MIN_CANDIDATE_RUNS_PER_CONDITION` candidate runs - single-run measurements
should never drive promote/reject decisions on this stack.

Effect-size + 95% CI reporting lives in this module too - see
:class:`EffectSize`, computed via Welch's t-test on the difference of
means + Welch-Satterthwaite degrees of freedom.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .vllm_benchmark import BenchmarkOutput, CandidateMetrics
from .vllm_benchmark_presets import DEFAULT_BRACKETED_AB_N

ClusterSignal = Literal["wall_seconds", "acceptance_rate", "auto"]

MIN_CANDIDATE_RUNS_PER_CONDITION: int = DEFAULT_BRACKETED_AB_N
"""Minimum candidate runs per condition before delta-style aggregates are computed.

Matches :data:`DEFAULT_BRACKETED_AB_N`. Below this threshold the
analyzer records a refusal in the report rather than emitting
under-powered aggregates that could mislead the operator.
"""

_DEFAULT_K_MAX: int = 4
"""Upper bound on the silhouette-score sweep for ``n_clusters`` selection."""


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------


class ClusterAssignment(BaseModel):
    """One candidate run's assignment to a cluster + its raw signal value."""

    model_config = ConfigDict(extra="forbid")

    candidate_name: str
    condition_label: str
    bracket_position: int
    cluster_id: int
    signal_value: float


class ClusterStats(BaseModel):
    """Summary statistics for one cluster across all conditions."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: int
    n_candidate_runs: int
    signal_mean: float
    signal_stddev: float
    signal_cov: float


class EffectSize(BaseModel):
    """Welch's-t delta and 95% CI for one condition vs a baseline reference.

    ``cluster_id=None`` is a pooled effect across all clusters;
    integer values mean within-cluster (the brief-mandated headline
    comparison).
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    condition_label: str
    baseline_condition_label: str
    cluster_id: int | None
    n_candidate: int
    n_baseline: int
    candidate_mean: float
    baseline_mean: float
    delta_absolute: float
    delta_pct: float
    ci95_low: float
    ci95_high: float
    welch_df: float


class ConditionClusterAggregate(BaseModel):
    """Per-(condition x cluster) aggregate: in-cluster condition stats."""

    model_config = ConfigDict(extra="forbid")

    condition_label: str
    cluster_id: int
    n_candidate_runs: int
    mean_effective_tok_s: float
    stddev_effective_tok_s: float
    cov_effective_tok_s: float
    mean_acceptance_rate: float
    mean_raw_tok_s: float
    effect_size_vs_baseline: EffectSize | None = None


class ConditionAggregate(BaseModel):
    """Per-condition aggregate: pooled + in-cluster breakdowns."""

    model_config = ConfigDict(extra="forbid")

    condition_label: str
    n_candidate_runs: int
    pooled_mean_effective_tok_s: float
    pooled_stddev_effective_tok_s: float
    pooled_cov_effective_tok_s: float
    pooled_mean_acceptance_rate: float
    pooled_stddev_acceptance_rate: float
    pooled_cov_acceptance_rate: float
    in_cluster: list[ConditionClusterAggregate]
    pooled_effect_size_vs_baseline: EffectSize | None = None


class AnalysisReport(BaseModel):
    """Top-level analysis report ready for serialization or rendering."""

    model_config = ConfigDict(extra="forbid")

    cluster_signal: str
    n_clusters: int
    n_candidate_runs: int
    cluster_assignments: list[ClusterAssignment]
    cluster_stats: list[ClusterStats]
    condition_aggregates: list[ConditionAggregate]
    refusals: list[str] = Field(default_factory=list)

    def to_markdown_summary(self) -> str:
        """Render a human-readable summary."""
        lines: list[str] = [
            f"# Cluster-conditioned analysis ({self.n_candidate_runs} candidate runs, k={self.n_clusters})",
            "",
            f"Cluster signal: `{self.cluster_signal}`",
            "",
            "## Clusters",
            "",
            "| cluster | candidate_runs | signal_mean | signal_stddev | signal_cov |",
            "|--------:|--------:|------------:|--------------:|-----------:|",
        ]
        for cs in self.cluster_stats:
            lines.append(
                f"| {cs.cluster_id} | {cs.n_candidate_runs} | {cs.signal_mean:.4f} | "
                f"{cs.signal_stddev:.4f} | {cs.signal_cov * 100:.2f}% |"
            )
        lines.extend(("", "## Per-condition aggregates", ""))
        for agg in self.condition_aggregates:
            lines.append(f"### `{agg.condition_label}` (n={agg.n_candidate_runs})")
            lines.append("")
            lines.append(
                f"- Pooled: eff_tok_s={agg.pooled_mean_effective_tok_s:.1f} "
                f"+/- {agg.pooled_stddev_effective_tok_s:.1f} "
                f"(CoV {agg.pooled_cov_effective_tok_s * 100:.2f}%); "
                f"accept={agg.pooled_mean_acceptance_rate:.4f} "
                f"(CoV {agg.pooled_cov_acceptance_rate * 100:.2f}%)"
            )
            if agg.pooled_effect_size_vs_baseline is not None:
                es = agg.pooled_effect_size_vs_baseline
                lines.append(
                    f"  - Delta vs baseline (pooled): {es.delta_absolute:+.1f} tok/s "
                    f"({es.delta_pct:+.2f}%) [95% CI: {es.ci95_low:+.1f}, {es.ci95_high:+.1f}; "
                    f"n={es.n_candidate}+{es.n_baseline}, Welch df={es.welch_df:.1f}]"
                )
            if agg.in_cluster:
                lines.append("- In-cluster:")
                for ic in agg.in_cluster:
                    lines.append(
                        f"  - cluster {ic.cluster_id}: n={ic.n_candidate_runs}, "
                        f"eff_tok_s={ic.mean_effective_tok_s:.1f} "
                        f"+/- {ic.stddev_effective_tok_s:.1f} "
                        f"(CoV {ic.cov_effective_tok_s * 100:.2f}%); "
                        f"accept={ic.mean_acceptance_rate:.4f}"
                    )
                    if ic.effect_size_vs_baseline is not None:
                        es = ic.effect_size_vs_baseline
                        lines.append(
                            f"    - Delta vs baseline cluster {es.cluster_id}: "
                            f"{es.delta_absolute:+.1f} tok/s ({es.delta_pct:+.2f}%) "
                            f"[95% CI: {es.ci95_low:+.1f}, {es.ci95_high:+.1f}; "
                            f"n={es.n_candidate}+{es.n_baseline}, Welch df={es.welch_df:.1f}]"
                        )
            lines.append("")
        if self.refusals:
            lines.extend(("## Refusals (insufficient sample size)", ""))
            for r in self.refusals:
                lines.append(f"- {r}")
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _signal_value(candidate_run: CandidateMetrics, signal: str) -> float:
    """Extract the signal value for clustering; raises on unknown signal."""
    if signal == "wall_seconds":
        return candidate_run.total_wall_seconds
    if signal == "acceptance_rate":
        return candidate_run.acceptance_rate
    raise ValueError(f"unknown cluster signal: {signal!r}")


def _pooled_cov(values: list[float]) -> tuple[float, float, float]:
    """Return ``(mean, stddev, cov)`` for a list; CoV is 0 when mean=0."""
    if not values:
        return 0.0, 0.0, 0.0
    mean = statistics.fmean(values)
    stddev = statistics.stdev(values) if len(values) > 1 else 0.0
    cov = (stddev / mean) if mean > 0 else 0.0
    return mean, stddev, cov


def _auto_select_signal(candidate_runs: list[CandidateMetrics]) -> str:
    """Pick whichever signal has higher pooled CoV across the candidate runs.

    Defaults to ``wall_seconds`` on ties (short-context bimodality is
    the more common workload shape).
    """
    if not candidate_runs:
        return "wall_seconds"
    _, _, cov_wall = _pooled_cov([c.total_wall_seconds for c in candidate_runs])
    _, _, cov_acc = _pooled_cov([c.acceptance_rate for c in candidate_runs])
    return "acceptance_rate" if cov_acc > cov_wall else "wall_seconds"


def _select_n_clusters(values: np.ndarray, k_max: int = _DEFAULT_K_MAX) -> int:
    """Silhouette-score-best ``k`` in range ``[2, min(k_max, n-1)]``.

    Returns 1 when there are too few candidate runs (<4) to cluster meaningfully.
    """
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    n = len(values)
    if n < 4:
        return 1
    X = values.reshape(-1, 1)
    best_k = 2
    best_score = -2.0
    for k in range(2, min(k_max, n - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels.tolist())) < 2:
            continue
        score = float(silhouette_score(X, labels))
        if score > best_score:
            best_score = score
            best_k = k
    return best_k


def _assign_clusters(values: np.ndarray, n_clusters: int) -> np.ndarray:
    """KMeans-assign labels, then remap so cluster 0 has the lowest mean."""
    if n_clusters <= 1:
        return np.zeros(len(values), dtype=int)
    from sklearn.cluster import KMeans  # noqa: PLC0415

    X = values.reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    raw_labels = km.fit_predict(X)
    centers = [(float(km.cluster_centers_[i][0]), i) for i in range(n_clusters)]
    centers.sort()
    remap = {orig: new for new, (_, orig) in enumerate(centers)}
    return np.array([remap[int(lbl)] for lbl in raw_labels], dtype=int)


def _welch_ttest_ci(
    candidate_values: list[float],
    baseline_values: list[float],
    alpha: float = 0.05,
) -> tuple[float, float, float, float] | None:
    """Return ``(mean_diff, ci_low, ci_high, welch_df)`` or ``None`` when underdetermined.

    Uses Welch's unequal-variance t-test for the CI on the difference
    of means. ``None`` when either input has fewer than 2 observations
    or both stddevs are zero.
    """
    if len(candidate_values) < 2 or len(baseline_values) < 2:
        return None
    from scipy import stats  # noqa: PLC0415

    cand = np.array(candidate_values, dtype=float)
    base = np.array(baseline_values, dtype=float)
    cand_var = float(np.var(cand, ddof=1))
    base_var = float(np.var(base, ddof=1))
    cand_n = len(cand)
    base_n = len(base)
    mean_diff = float(np.mean(cand) - np.mean(base))
    se_diff_sq = cand_var / cand_n + base_var / base_n
    if se_diff_sq <= 0:
        return None
    se_diff = float(np.sqrt(se_diff_sq))
    df_num = (cand_var / cand_n + base_var / base_n) ** 2
    df_den = (cand_var / cand_n) ** 2 / max(cand_n - 1, 1) + (base_var / base_n) ** 2 / max(base_n - 1, 1)
    if df_den <= 0:
        return None
    welch_df = float(df_num / df_den)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, welch_df))
    half_width = t_crit * se_diff
    return mean_diff, mean_diff - half_width, mean_diff + half_width, welch_df


def _effect_size(
    candidate_values: list[float],
    baseline_values: list[float],
    *,
    metric: str,
    condition_label: str,
    baseline_condition_label: str,
    cluster_id: int | None,
) -> EffectSize | None:
    """Compute an :class:`EffectSize` or return ``None`` when underdetermined."""
    res = _welch_ttest_ci(candidate_values, baseline_values)
    if res is None:
        return None
    mean_diff, ci_low, ci_high, welch_df = res
    baseline_mean = float(np.mean(baseline_values))
    candidate_mean = float(np.mean(candidate_values))
    delta_pct = (mean_diff / baseline_mean * 100.0) if baseline_mean != 0 else 0.0
    return EffectSize(
        metric=metric,
        condition_label=condition_label,
        baseline_condition_label=baseline_condition_label,
        cluster_id=cluster_id,
        n_candidate=len(candidate_values),
        n_baseline=len(baseline_values),
        candidate_mean=candidate_mean,
        baseline_mean=baseline_mean,
        delta_absolute=mean_diff,
        delta_pct=delta_pct,
        ci95_low=ci_low,
        ci95_high=ci_high,
        welch_df=welch_df,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_candidate_runs(output_dir: Path) -> list[CandidateMetrics]:
    """Read every ``BenchmarkOutput`` JSON in ``output_dir``, flatten candidate runs.

    Subdirectories are NOT recursed. Callers wanting cross-dataset
    analysis should invoke once per dataset dir.
    """
    candidate_runs: list[CandidateMetrics] = []
    for path in sorted(output_dir.glob("*.json")):
        try:
            doc = BenchmarkOutput.model_validate_json(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, ValueError):
            continue
        candidate_runs.extend(doc.candidates)
    return candidate_runs


def analyze(
    output_dir: Path,
    cluster_signal: ClusterSignal = "auto",
    min_candidate_runs_per_condition: int = MIN_CANDIDATE_RUNS_PER_CONDITION,
) -> AnalysisReport:
    """Full pipeline: load -> cluster -> per-condition aggregate -> effect-size -> report."""
    candidate_runs = load_candidate_runs(output_dir)
    if not candidate_runs:
        raise ValueError(f"No BenchmarkOutput JSONs found under {output_dir}")

    resolved_signal = _auto_select_signal(candidate_runs) if cluster_signal == "auto" else cluster_signal
    values = np.array([_signal_value(candidate_run, resolved_signal) for candidate_run in candidate_runs], dtype=float)
    n_clusters = _select_n_clusters(values)
    labels = _assign_clusters(values, n_clusters)

    assignments = [
        ClusterAssignment(
            candidate_name=candidate_run.name,
            condition_label=candidate_run.condition_label,
            bracket_position=candidate_run.bracket_position,
            cluster_id=int(lbl),
            signal_value=float(val),
        )
        for candidate_run, lbl, val in zip(candidate_runs, labels, values, strict=True)
    ]

    cluster_stats: list[ClusterStats] = []
    for cid in range(n_clusters):
        cluster_values = [float(values[i]) for i, lbl in enumerate(labels) if int(lbl) == cid]
        mean, stddev, cov = _pooled_cov(cluster_values)
        cluster_stats.append(
            ClusterStats(
                cluster_id=cid,
                n_candidate_runs=len(cluster_values),
                signal_mean=mean,
                signal_stddev=stddev,
                signal_cov=cov,
            ),
        )

    by_condition: dict[str, list[tuple[int, CandidateMetrics]]] = {}
    for candidate_run, lbl in zip(candidate_runs, labels, strict=True):
        by_condition.setdefault(candidate_run.condition_label, []).append((int(lbl), candidate_run))

    baseline_pooled_eff: list[float] = []
    baseline_per_cluster_eff: dict[int, list[float]] = {}
    for lbl, candidate_run in by_condition.get("baseline", []):
        baseline_pooled_eff.append(candidate_run.effective_tok_s)
        baseline_per_cluster_eff.setdefault(lbl, []).append(candidate_run.effective_tok_s)

    aggregates: list[ConditionAggregate] = []
    refusals: list[str] = []
    for condition in sorted(by_condition):
        labeled = by_condition[condition]
        if len(labeled) < min_candidate_runs_per_condition:
            refusals.append(
                f"condition {condition!r} has only {len(labeled)} candidate runs; "
                f"need >={min_candidate_runs_per_condition} - refusing aggregate"
            )
            continue
        pooled_eff = [c.effective_tok_s for _, c in labeled]
        pooled_acc = [c.acceptance_rate for _, c in labeled]
        eff_mean, eff_stddev, eff_cov = _pooled_cov(pooled_eff)
        acc_mean, acc_stddev, acc_cov = _pooled_cov(pooled_acc)

        in_cluster: list[ConditionClusterAggregate] = []
        for cid in range(n_clusters):
            cluster_candidate_runs = [c for lbl, c in labeled if lbl == cid]
            if not cluster_candidate_runs:
                continue
            ic_eff = [c.effective_tok_s for c in cluster_candidate_runs]
            ic_acc = [c.acceptance_rate for c in cluster_candidate_runs]
            ic_raw = [c.raw_tok_s for c in cluster_candidate_runs]
            mean_eff, stddev_eff, cov_eff = _pooled_cov(ic_eff)
            ic_effect: EffectSize | None = None
            if condition != "baseline" and cid in baseline_per_cluster_eff:
                ic_effect = _effect_size(
                    ic_eff,
                    baseline_per_cluster_eff[cid],
                    metric="effective_tok_s",
                    condition_label=condition,
                    baseline_condition_label="baseline",
                    cluster_id=cid,
                )
            in_cluster.append(
                ConditionClusterAggregate(
                    condition_label=condition,
                    cluster_id=cid,
                    n_candidate_runs=len(cluster_candidate_runs),
                    mean_effective_tok_s=mean_eff,
                    stddev_effective_tok_s=stddev_eff,
                    cov_effective_tok_s=cov_eff,
                    mean_acceptance_rate=statistics.fmean(ic_acc),
                    mean_raw_tok_s=statistics.fmean(ic_raw),
                    effect_size_vs_baseline=ic_effect,
                ),
            )
        pooled_effect: EffectSize | None = None
        if condition != "baseline" and baseline_pooled_eff:
            pooled_effect = _effect_size(
                pooled_eff,
                baseline_pooled_eff,
                metric="effective_tok_s",
                condition_label=condition,
                baseline_condition_label="baseline",
                cluster_id=None,
            )
        aggregates.append(
            ConditionAggregate(
                condition_label=condition,
                n_candidate_runs=len(labeled),
                pooled_mean_effective_tok_s=eff_mean,
                pooled_stddev_effective_tok_s=eff_stddev,
                pooled_cov_effective_tok_s=eff_cov,
                pooled_mean_acceptance_rate=acc_mean,
                pooled_stddev_acceptance_rate=acc_stddev,
                pooled_cov_acceptance_rate=acc_cov,
                in_cluster=in_cluster,
                pooled_effect_size_vs_baseline=pooled_effect,
            ),
        )

    return AnalysisReport(
        cluster_signal=resolved_signal,
        n_clusters=n_clusters,
        n_candidate_runs=len(candidate_runs),
        cluster_assignments=assignments,
        cluster_stats=cluster_stats,
        condition_aggregates=aggregates,
        refusals=refusals,
    )
