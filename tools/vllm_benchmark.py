#!/usr/bin/env -S uv run
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""vllm-benchmark: drive the vLLM benchmark harness from the command line.

Full usage notes live in docs/developer-guide/vllm-benchmark.md.

Subcommands:
    list                           Print the available preset matrices.
    run CORPUS --output PATH ...   Replay CORPUS against the chosen candidates
                                   and persist a BenchmarkOutput JSON.
    compare PATH                   Render a previously-saved BenchmarkOutput
                                   as a candidate-by-metric table.

Invocation::

    uv run --frozen --extra cu129 --extra engine --group dev \
        python tools/vllm_benchmark.py list

    uv run --frozen --extra cu129 --extra engine --group dev \
        python tools/vllm_benchmark.py run \
        /path/to/trace.jsonl \
        --output /path/to/benchmark.json \
        --candidates default_matrix
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import click
from pydantic import BaseModel, ConfigDict, Field
from rich.console import Console
from rich.table import Table

from nemo_safe_synthesizer.generation.vllm_benchmark import (
    BenchmarkCandidate,
    BenchmarkCandidateDocument,
    BenchmarkCorpus,
    BenchmarkEngineConfig,
    BenchmarkOutput,
    CandidateMetrics,
    SkipRecord,
)
from nemo_safe_synthesizer.generation.vllm_benchmark_presets import PRESETS
from nemo_safe_synthesizer.generation.vllm_benchmark_wandb import (
    init_candidate_run,
    log_and_finish,
    resolve_sweep_id,
)

console = Console()

SUBPROCESS_STDERR_LIMIT: int = 500
"""Maximum bytes of captured stderr to record on a subprocess failure."""


class SubprocessRunResult(BaseModel):
    """Outcome of one subprocess-isolated candidate run."""

    model_config = ConfigDict(extra="forbid")

    metrics: CandidateMetrics | None = Field(default=None, description="Populated when the child exited successfully.")
    error: str | None = Field(default=None, description="Captured stderr summary; populated on non-zero exit.")
    error_class: str | None = Field(default=None, description="Best-effort exception class name parsed from stderr.")


def _truncate_stderr(stderr: str, limit: int = SUBPROCESS_STDERR_LIMIT) -> str:
    """Trim ``stderr`` to ``limit`` bytes, keeping the tail."""
    stderr = stderr.strip()
    if len(stderr) <= limit:
        return stderr
    head = "...[truncated]..."
    return head + stderr[-(limit - len(head)) :]


def _parse_error_class(stderr: str) -> str:
    """Best-effort parse of the exception class name from a Python traceback."""
    for line in reversed(stderr.strip().splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        head = stripped.split(":", 1)[0]
        if head.isidentifier() or "." in head:
            return head
        return "Error"
    return "Error"


def _run_candidate_command_args(
    candidate: BenchmarkCandidate,
    corpus_path: str | Path,
    result_path: Path,
    simulate_training_overlap_seconds: float,
) -> list[str]:
    """Build argv for this tool's isolated candidate runner."""
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "_run-candidate",
        "--candidate",
        candidate.model_dump_json(),
        "--corpus",
        str(corpus_path),
        "--result-out",
        str(result_path),
        "--simulate-training-overlap-seconds",
        str(simulate_training_overlap_seconds),
    ]


@click.group()
def cli() -> None:
    """VLLM benchmark harness CLI."""


@cli.command("list")
def list_cmd() -> None:
    """Print the available preset names."""
    for name in sorted(PRESETS):
        console.print(name)


def _resolve_candidates(
    base: BenchmarkEngineConfig,
    preset_name: str | None,
    candidates_file: str | None,
) -> list[BenchmarkCandidate]:
    """Resolve the candidate list from a preset name or a JSON file.

    Exactly one of ``preset_name`` / ``candidates_file`` must be set.
    The candidates-file shape is ``{"candidates": [BenchmarkCandidate, ...]}``.
    """
    if preset_name and candidates_file:
        raise click.UsageError("--candidates and --candidates-file are mutually exclusive.")
    if preset_name:
        if preset_name not in PRESETS:
            raise click.UsageError(f"Unknown preset {preset_name!r}; available: {sorted(PRESETS)}")
        return PRESETS[preset_name](base)
    if candidates_file:
        try:
            return BenchmarkCandidateDocument.from_json_file(candidates_file).candidates
        except ValueError as exc:
            raise click.UsageError(f"Invalid candidates file {candidates_file}: {exc}") from exc
    raise click.UsageError("One of --candidates or --candidates-file is required.")


def run_benchmark_in_subprocess(
    candidate: BenchmarkCandidate,
    corpus_path: str | Path,
    simulate_training_overlap_seconds: float = 0.0,
) -> SubprocessRunResult:
    """Run one candidate in a child process the OS reclaims on exit."""
    with tempfile.TemporaryDirectory(prefix="nss-vllm-benchmark-") as result_dir:
        result_path = Path(result_dir) / "result.json"
        completed = subprocess.run(
            _run_candidate_command_args(
                candidate,
                corpus_path,
                result_path,
                simulate_training_overlap_seconds,
            ),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = _truncate_stderr(completed.stderr or completed.stdout)
            return SubprocessRunResult(
                error=stderr or f"subprocess exit {completed.returncode}",
                error_class=_parse_error_class(completed.stderr or completed.stdout),
            )
        if not result_path.exists() or result_path.stat().st_size == 0:
            return SubprocessRunResult(
                error="subprocess exited 0 but produced no result file",
                error_class="RuntimeError",
            )
        metrics = CandidateMetrics.model_validate_json(result_path.read_text(encoding="utf-8"))
        return SubprocessRunResult(metrics=metrics)


@cli.command("run")
@click.argument("corpus_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Where to write the BenchmarkOutput JSON.",
)
@click.option(
    "--candidates",
    "preset_name",
    default=None,
    help=f"Preset name. One of: {sorted(PRESETS)}.",
)
@click.option(
    "--candidates-file",
    "candidates_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Custom JSON file with a 'candidates' list of BenchmarkCandidate objects.",
)
@click.option(
    "--simulate-training-overlap-seconds",
    "simulate_training_overlap_seconds",
    default=0.0,
    type=float,
    show_default=True,
    help="Per-candidate seconds to sleep after engine-init kickoff (simulates concurrent training).",
)
def run_cmd(
    corpus_path: Path,
    output_path: Path,
    preset_name: str | None,
    candidates_file: str | None,
    simulate_training_overlap_seconds: float,
) -> None:
    """Replay CORPUS_PATH against the chosen candidates and persist results."""
    corpus = BenchmarkCorpus.from_trace_jsonl(corpus_path)
    base = BenchmarkEngineConfig.model_validate(corpus.header.engine_parameters or {})
    candidates = _resolve_candidates(base, preset_name, candidates_file)
    if not candidates:
        raise click.UsageError("Resolved candidate list is empty.")

    results: list[CandidateMetrics] = []
    skipped: list[SkipRecord] = []
    sweep_id = resolve_sweep_id()
    for idx, candidate in enumerate(candidates, start=1):
        console.print(f"[{idx}/{len(candidates)}] running candidate {candidate.name!r}")
        wandb_run = init_candidate_run(
            candidate_name=candidate.name,
            candidate_idx=idx,
            total=len(candidates),
            corpus_run_id=corpus.header.run_id,
            corpus_size=len(corpus.prompts),
            sweep_id=sweep_id,
            candidate_condition_label=candidate.condition_label,
            candidate_bracket_position=candidate.bracket_position,
        )
        result = run_benchmark_in_subprocess(
            candidate,
            corpus_path,
            simulate_training_overlap_seconds=simulate_training_overlap_seconds,
        )
        if result.metrics is None:
            console.print(f"  [yellow]skipped:[/yellow] {result.error_class or 'Error'}: {result.error}")
            skipped.append(
                SkipRecord(
                    name=candidate.name,
                    error=result.error or "subprocess failed",
                    error_class=result.error_class or "Error",
                    attempted_at=datetime.now(timezone.utc),
                ),
            )
            log_and_finish(wandb_run, metrics=None, exit_code=1)
            continue
        results.append(result.metrics)
        log_and_finish(wandb_run, metrics=result.metrics, exit_code=0)
        console.print(
            f"  raw={result.metrics.raw_tok_s:.1f} tok/s  "
            f"accept={result.metrics.acceptance_rate:.3f}  "
            f"effective={result.metrics.effective_tok_s:.1f} tok/s",
        )

    output = BenchmarkOutput(
        corpus_run_id=corpus.header.run_id,
        corpus_size=len(corpus.prompts),
        candidates=results,
        skipped_candidates=skipped,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"[green]wrote[/green] {output_path}  ({len(results)}/{len(candidates)} ok, {len(skipped)} skipped)")


@cli.command("_run-candidate", hidden=True)
@click.option("--candidate", required=True, help="JSON-serialised BenchmarkCandidate.")
@click.option(
    "--corpus",
    "corpus_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Path to the corpus JSONL.",
)
@click.option(
    "--result-out",
    "result_out",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Path to write the CandidateMetrics JSON.",
)
@click.option(
    "--simulate-training-overlap-seconds",
    type=float,
    default=0.0,
    show_default=True,
    help="Seconds to sleep after kicking off engine init (simulates concurrent training).",
)
def run_candidate_cmd(
    candidate: str,
    corpus_path: Path,
    result_out: Path,
    simulate_training_overlap_seconds: float,
) -> None:
    """Run one benchmark candidate in an isolated child process."""
    from nemo_safe_synthesizer.generation.vllm_benchmark import run_benchmark

    metrics = run_benchmark(
        candidate=BenchmarkCandidate.model_validate_json(candidate),
        corpus=BenchmarkCorpus.from_trace_jsonl(corpus_path),
        simulate_training_overlap_seconds=simulate_training_overlap_seconds,
    )
    result_out.write_text(metrics.model_dump_json(indent=2), encoding="utf-8")


@cli.command("compare")
@click.argument("output_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def compare_cmd(output_path: Path) -> None:
    """Render OUTPUT_PATH as a candidate-by-metric table, sorted by effective_tok_s."""
    output = BenchmarkOutput.model_validate_json(output_path.read_text(encoding="utf-8"))
    sorted_candidates = sorted(output.candidates, key=lambda c: c.effective_tok_s, reverse=True)
    table = Table(title=f"BenchmarkOutput ({output.corpus_run_id}, n={output.corpus_size})")
    for col in (
        "candidate",
        "eff tok/s",
        "raw tok/s",
        "accept",
        "ttft p50 ms",
        "ttft p99 ms",
        "peak vram GiB",
        "startup s",
        "ok/tried",
    ):
        table.add_column(col, justify="right" if col != "candidate" else "left", overflow="fold")
    for m in sorted_candidates:
        table.add_row(
            m.name,
            f"{m.effective_tok_s:.1f}",
            f"{m.raw_tok_s:.1f}",
            f"{m.acceptance_rate:.3f}",
            f"{m.ttft_p50_ms:.1f}",
            f"{m.ttft_p99_ms:.1f}",
            f"{m.observability.peak_vram_gb:.2f}" if m.observability.peak_vram_gb is not None else "--",
            f"{m.startup_seconds:.1f}",
            f"{m.prompts_accepted}/{m.prompts_attempted}",
        )
    console.print(table)
    if output.skipped_candidates:
        skip_table = Table(title="Skipped")
        skip_table.add_column("candidate", overflow="fold")
        skip_table.add_column("error class")
        skip_table.add_column("error", overflow="fold")
        for skip in output.skipped_candidates:
            skip_table.add_row(skip.name, skip.error_class, skip.error)
        console.print(skip_table)


@cli.command("analyze")
@click.argument("output_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option(
    "--cluster-signal",
    type=click.Choice(["wall_seconds", "acceptance_rate", "auto"]),
    default="auto",
    show_default=True,
    help=(
        "Which per-candidate-run metric to partition on. Use 'wall_seconds' for "
        "short-context workloads, 'acceptance_rate' for long-output workloads, "
        "or 'auto' to pick whichever has higher pooled CoV."
    ),
)
@click.option(
    "--min-runs-per-condition",
    "min_candidate_runs_per_condition",
    type=int,
    default=None,
    show_default="MIN_CANDIDATE_RUNS_PER_CONDITION (6)",
    help="Refuse aggregates for conditions below this N. Brief mandates N>=6 candidate runs.",
)
@click.option(
    "--json-out",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to write the full AnalysisReport as JSON.",
)
def analyze_cmd(
    output_dir: Path,
    cluster_signal: str,
    min_candidate_runs_per_condition: int | None,
    json_out: Path | None,
) -> None:
    """Cluster-conditioned analysis across every BenchmarkOutput JSON in OUTPUT_DIR."""
    from nemo_safe_synthesizer.generation.vllm_benchmark_analysis import (
        MIN_CANDIDATE_RUNS_PER_CONDITION,
        ClusterSignal,
        analyze,
    )

    report = analyze(
        output_dir,
        cluster_signal=cast(ClusterSignal, cluster_signal),
        min_candidate_runs_per_condition=(
            MIN_CANDIDATE_RUNS_PER_CONDITION
            if min_candidate_runs_per_condition is None
            else min_candidate_runs_per_condition
        ),
    )
    console.print(report.to_markdown_summary())
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[green]wrote[/green] {json_out}")


if __name__ == "__main__":
    cli()
