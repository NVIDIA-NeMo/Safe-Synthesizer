# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Subprocess entry point for one isolated benchmark candidate run.

Invoked by :func:`vllm_benchmark.run_benchmark_in_subprocess` via
``python -m nemo_safe_synthesizer.generation.vllm_benchmark_single_run``.
Loads the candidate from the ``--candidate`` JSON argv, loads the
corpus from ``--corpus``, runs :func:`vllm_benchmark.run_benchmark`,
and writes the resulting ``CandidateMetrics`` JSON to ``--result-out``.

Subprocess isolation is what makes a multi-candidate matrix reliable
on this stack — vLLM holds significant CUDA + DRAM state in
module-level globals; running each candidate in a fresh interpreter
lets the OS reclaim everything on exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .vllm_benchmark import BenchmarkCandidate, BenchmarkCorpus, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one benchmark candidate in isolation.")
    parser.add_argument("--candidate", required=True, help="JSON-serialised BenchmarkCandidate.")
    parser.add_argument("--corpus", required=True, help="Path to the corpus JSONL.")
    parser.add_argument("--result-out", required=True, help="Path to write the CandidateMetrics JSON.")
    parser.add_argument(
        "--simulate-training-overlap-seconds",
        type=float,
        default=0.0,
        help="Seconds to sleep after kicking off engine init (simulates concurrent training).",
    )
    args = parser.parse_args()

    candidate = BenchmarkCandidate.model_validate_json(args.candidate)
    corpus = BenchmarkCorpus.from_trace_jsonl(args.corpus)
    metrics = run_benchmark(
        candidate=candidate,
        corpus=corpus,
        simulate_training_overlap_seconds=args.simulate_training_overlap_seconds,
    )
    Path(args.result_out).write_text(metrics.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
