# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark preset matrices for the vLLM harness.

A "preset" is a function that takes the corpus's default
:class:`BenchmarkEngineConfig` (extracted from the trace header) and
returns one or more :class:`BenchmarkCandidate` instances. The CLI
driver exposes these by name via ``--candidates <preset_name>``.

Every preset-built candidate pins ``SamplingParams.seed`` to
:data:`DEFAULT_BENCHMARK_SEED` (via :func:`_seeded_overrides`) so the
per-request RNG portion of acceptance-rate variance is eliminated.
Residual variance is structural non-RNG (numerical / scheduler
interleaving) and is handled post-hoc by the cluster-conditioned
analyzer (separate commit).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .vllm_benchmark import BenchmarkCandidate, BenchmarkEngineConfig

PresetFn = Callable[[BenchmarkEngineConfig], list[BenchmarkCandidate]]
"""A preset resolves the corpus-default engine config into candidate cells."""

DEFAULT_BENCHMARK_SEED: int = 42
"""Default ``SamplingParams.seed`` value for preset-built candidates.

Prior triages observed acceptance-rate CoV up to ~14% on long-output
workloads from per-request JSON-schema sampling RNG drift; pinning the
seed collapses that variance to in-cluster noise (~3% CoV; ~5% pooled
on long-output workloads per the 2026-05-26 seed-pin verification).
42 is conventional; any fixed int works.
"""

DEFAULT_MAX_MODEL_LEN: int = 4096
"""Fallback ``max_model_len`` for presets when the corpus header doesn't have a resolver hint.

vLLM 0.20 defaults this to whatever the model's tokenizer reports, which
for Mistral-7B is 32768 — over-provisions the KV cache budget for the
tabular workloads we benchmark.
"""

ATTENTION_BACKENDS: tuple[str, ...] = (
    "FLASHINFER",
    "FLASH_ATTN",
    "TRITON_ATTN",
)
"""CUDA attention backends the sweep covers; excludes ROCm/XPU/MLA variants."""

STRUCTURED_BACKENDS: tuple[str, ...] = ("xgrammar", "outlines", "guidance")
"""Structured-output backends the sweep covers."""

BATCHING_STEPS: tuple[tuple[int, int], ...] = (
    (128, 4096),
    (256, 8192),
    (512, 16384),
)
"""(max_num_seqs, max_num_batched_tokens) steps for the batching sweep."""

MAX_MODEL_LEN_STEPS: tuple[int, ...] = (2048, 4096, 8192)
"""``max_model_len`` steps for the max-model-len sweep."""


def _seeded_overrides(
    seed: int | None = DEFAULT_BENCHMARK_SEED, *, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a ``sampling_overrides`` dict that pins seed + optional extras.

    Returns a fresh dict so callers can mutate without affecting other
    candidates. ``seed=None`` opts out of seed pinning.
    """
    overrides: dict[str, Any] = {}
    if seed is not None:
        overrides["seed"] = seed
    if extra:
        overrides.update(extra)
    return overrides


def _with_default_max_model_len(base: BenchmarkEngineConfig) -> BenchmarkEngineConfig:
    """Set ``max_model_len`` to :data:`DEFAULT_MAX_MODEL_LEN` when unset.

    Saves the KV-cache budget from over-provisioning to the model's
    tokenizer-reported maximum.
    """
    if base.max_model_len is not None:
        return base
    return base.model_copy(update={"max_model_len": DEFAULT_MAX_MODEL_LEN})


def _named_copy(base: BenchmarkEngineConfig, name: str, **updates: Any) -> BenchmarkCandidate:
    """Build a candidate that overrides exactly ``updates`` on top of ``base``.

    Seed-pinned via :data:`DEFAULT_BENCHMARK_SEED`. Callers needing
    un-pinned RNG should construct ``BenchmarkCandidate`` directly.
    """
    return BenchmarkCandidate(
        name=name,
        engine_config=_with_default_max_model_len(base).model_copy(update=updates),
        sampling_overrides=_seeded_overrides(),
    )


def baseline(base: BenchmarkEngineConfig) -> BenchmarkCandidate:
    """Pass-through candidate — runs the corpus's default config unchanged.

    Still seed-pinned for reproducibility. See :func:`_named_copy`.
    """
    return BenchmarkCandidate(
        name="baseline",
        engine_config=_with_default_max_model_len(base),
        sampling_overrides=_seeded_overrides(),
    )


def attention_backend_sweep(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """One candidate per attention backend in :data:`ATTENTION_BACKENDS`."""
    return [
        _named_copy(base, f"attention_backend={backend}", attention_backend=backend) for backend in ATTENTION_BACKENDS
    ]


def prefix_caching_sweep(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """Probe ``enable_prefix_caching=True`` against the implicit-off baseline.

    The single emitted candidate enables prefix caching; the off case
    is the implicit baseline because vLLM defaults
    ``enable_prefix_caching=False`` and the harness leaves the field
    unset in :func:`baseline`.
    """
    return [_named_copy(base, "prefix_caching=on", enable_prefix_caching=True)]


def batching_sweep(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """Vary ``max_num_seqs`` and ``max_num_batched_tokens`` across :data:`BATCHING_STEPS`."""
    return [
        _named_copy(
            base,
            f"batch_seqs={seqs}_tokens={tokens}",
            max_num_seqs=seqs,
            max_num_batched_tokens=tokens,
        )
        for seqs, tokens in BATCHING_STEPS
    ]


def structured_backend_sweep(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """One candidate per structured-output backend in :data:`STRUCTURED_BACKENDS`."""
    return [
        _named_copy(base, f"structured_backend={backend}", structured_generation_backend=backend)
        for backend in STRUCTURED_BACKENDS
    ]


def max_model_len_sweep(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """One candidate per ``max_model_len`` step in :data:`MAX_MODEL_LEN_STEPS`."""
    return [_named_copy(base, f"max_model_len={length}", max_model_len=length) for length in MAX_MODEL_LEN_STEPS]


def default_matrix(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """Concatenation of every sweep, deduplicated by (engine config, overrides).

    Dedup is conservative — two candidates with identical engine config
    but different ``name`` are still considered duplicates because the
    resulting engine + sampling combination is what the runner measures.
    """
    seen: set[tuple[Any, ...]] = set()
    out: list[BenchmarkCandidate] = []
    for builder in (
        attention_backend_sweep,
        prefix_caching_sweep,
        batching_sweep,
        structured_backend_sweep,
        max_model_len_sweep,
    ):
        for cand in builder(base):
            key = (
                cand.engine_config.model_dump_json(),
                json.dumps(cand.sampling_overrides, sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(cand)
    return out


DEFAULT_BRACKETED_AB_N: int = 6
"""Cells per condition in :func:`bracketed_ab`.

N=6 is the methodology-critic-recommended compromise between N=4 (~20%
statistical power for detecting +5% effects under the stack's observed
~3% in-cluster CoV) and N=8 (busts the 24h × $350 envelope on a
4-dataset × 4-condition matrix).
"""


def bracketed_ab(
    base: BenchmarkEngineConfig,
    *,
    candidate_engine_overrides: dict[str, Any] | None = None,
    candidate_sampling_overrides: dict[str, Any] | None = None,
    candidate_batch_dispatch_mode: str | None = None,
    condition_label: str,
    n_samples_per_condition: int = DEFAULT_BRACKETED_AB_N,
) -> list[BenchmarkCandidate]:
    """Emit an interleaved baseline-candidate cell sequence for bracketed A/B.

    Returns ``2 * n_samples_per_condition`` cells: ``n`` baselines
    interleaved with ``n`` candidate cells. ``bracket_position`` is set
    on each cell so the cluster-conditioned analyzer can align
    candidate cells with their bracketing baselines for drift detection.

    Both baselines and candidates pin
    ``SamplingParams.seed=DEFAULT_BENCHMARK_SEED``. The seed-pin
    verification (2026-05-26) showed this does NOT collapse acceptance
    variance to <0.5% CoV on long-output workloads but DOES eliminate
    the per-request RNG portion — residual ~5% pooled CoV is structural
    non-RNG variance that the cluster-conditioned analyzer partitions
    out post-hoc.
    """
    engine_overrides = candidate_engine_overrides or {}
    sampling_extra = candidate_sampling_overrides or {}
    cells: list[BenchmarkCandidate] = []
    for i in range(n_samples_per_condition):
        cells.append(
            BenchmarkCandidate(
                name=f"bracket_baseline_{i}",
                engine_config=_with_default_max_model_len(base),
                sampling_overrides=_seeded_overrides(),
                condition_label="baseline",
                bracket_position=2 * i,
            ),
        )
        cand_kwargs: dict[str, Any] = {
            "name": f"bracket_{condition_label}_{i}",
            "engine_config": _with_default_max_model_len(base).model_copy(update=engine_overrides),
            "sampling_overrides": _seeded_overrides(extra=sampling_extra),
            "condition_label": condition_label,
            "bracket_position": 2 * i + 1,
        }
        if candidate_batch_dispatch_mode is not None:
            cand_kwargs["batch_dispatch_mode"] = candidate_batch_dispatch_mode
        cells.append(BenchmarkCandidate(**cand_kwargs))
    return cells


# Phase B matrix-condition wrappers. Each yields a 2N-cell sequence
# (N baselines + N condition-specific candidates, interleaved).


def bracketed_ab_baseline_pool(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """N baseline cells with bracket_position labels — the shared baseline pool."""
    return [
        BenchmarkCandidate(
            name=f"bracket_baseline_pool_{i}",
            engine_config=_with_default_max_model_len(base),
            sampling_overrides=_seeded_overrides(),
            condition_label="baseline",
            bracket_position=i,
        )
        for i in range(DEFAULT_BRACKETED_AB_N)
    ]


def bracketed_ab_n_fanout(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """N baselines + N n_fanout candidates, interleaved.

    n_fanout uses vLLM's ``SamplingParams.n=N`` so a single prompt
    forks the decode state across N samples sharing the prefill KV
    cache. Speculative win only when num_prompts > max_num_seqs.
    """
    return bracketed_ab(
        base,
        candidate_batch_dispatch_mode="n_fanout",
        condition_label="n_fanout",
    )


def bracketed_ab_spec_ngram(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """N baselines + N spec_ngram candidates, interleaved.

    Speculative-decoding overlay: ``speculative_config={'method': 'ngram',
    'num_speculative_tokens': 4, 'prompt_lookup_max': 4}``. Prior triage
    measured +8.1% effective throughput on Mistral-7B + tabular.
    """
    return bracketed_ab(
        base,
        candidate_engine_overrides={
            "speculative_config": {
                "method": "ngram",
                "num_speculative_tokens": 4,
                "prompt_lookup_max": 4,
            },
        },
        condition_label="spec_ngram",
    )


def bracketed_ab_fp8(base: BenchmarkEngineConfig) -> list[BenchmarkCandidate]:
    """N baselines + N fp8 KV-cache candidates, interleaved.

    Forces ``kv_cache_dtype='fp8'`` to halve KV cache memory footprint
    at a per-token quality cost vLLM characterises as small.
    """
    return bracketed_ab(
        base,
        candidate_engine_overrides={"kv_cache_dtype": "fp8"},
        condition_label="fp8",
    )


# Map of preset name → callable used by the CLI to resolve ``--candidates``.
PRESETS: dict[str, PresetFn] = {
    "baseline": lambda base: [baseline(base)],
    "attention_backend_sweep": attention_backend_sweep,
    "prefix_caching_sweep": prefix_caching_sweep,
    "batching_sweep": batching_sweep,
    "structured_backend_sweep": structured_backend_sweep,
    "max_model_len_sweep": max_model_len_sweep,
    "default_matrix": default_matrix,
    "bracketed_ab_baseline_pool": bracketed_ab_baseline_pool,
    "bracketed_ab_n_fanout": bracketed_ab_n_fanout,
    "bracketed_ab_spec_ngram": bracketed_ab_spec_ngram,
    "bracketed_ab_fp8": bracketed_ab_fp8,
}
