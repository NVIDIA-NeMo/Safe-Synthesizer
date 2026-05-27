# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark-harness data models for the vLLM backend.

The harness replays a captured workload corpus (one ``GenerationTrace``
JSONL) under varying engine + sampling configurations and reports
calibrated metrics per candidate. The models in this module stay
CPU-importable; the runner + subprocess wrapper live in sibling
modules and import vLLM lazily.

Architecture:

- :class:`BenchmarkCorpus` — the replayable input (one corpus per
  dataset, captured once via the production ``VllmBackend`` trace
  surface). Header carries the model reference + LoRA path + the
  engine kwargs at capture time. Prompt records carry the original
  sampling params so the harness can replay them faithfully.
- :class:`BenchmarkCandidate` — one configuration to benchmark
  (engine kwargs overlay + sparse sampling overrides + per-cell
  identity for sweep grouping).
- :class:`CandidateMetrics` — per-cell measured outputs: throughput,
  acceptance, TTFT, etc. Composes :class:`CellObservability` from PR-A's
  ``vllm_observability`` module so the benchmark schema doesn't
  re-define observability primitives — it consumes them.
- :class:`BenchmarkOutput` — JSON-serialised result of one matrix
  invocation, with skip records for candidates that failed.

The runner (next commit) is in ``vllm_benchmark.py`` alongside these
models; the subprocess wrapper + single-run entry point are split into
``vllm_benchmark_single_run.py``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..generation.vllm_observability import CellObservability

PromptAssemblyMode = Literal["multi_record", "per_record"]
"""Prompt-assembly regime — controls how max_tokens partitions the budget."""

BatchDispatchMode = Literal["replicate", "n_fanout"]
"""How corpus prompts get submitted to vLLM.

- ``'replicate'``: every corpus prompt becomes an independent request
  (``n=1``). The default; matches production ``VllmBackend.generate``.
- ``'n_fanout'``: dispatch a single prompt with
  ``SamplingParams.n = num_prompts`` so vLLM amortises the shared
  schema-prefix prefill across N samples. Only valid when every
  corpus prompt is identical; the harness uses the first prompt as
  the fanout payload.
"""


class TraceHeader(BaseModel):
    """Header line at the top of a captured trace JSONL.

    Carries the engine + LoRA + dataset metadata needed to rebuild an
    equivalent inference setup at replay time.
    """

    model_config = ConfigDict(extra="allow")

    run_id: str = Field(description="Capture identifier; opaque to the harness, surfaced in JSON.")
    pretrained_model: str = Field(description="Model reference (HF name, local path, or ``ModelRef`` string).")
    lora_path: Path | None = Field(default=None, description="LoRA adapter path; ``None`` for the bare-model case.")
    dataset_schema: dict[str, Any] = Field(description="Tabular schema the processor validates against.")
    engine_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="vLLM constructor kwargs at capture time (e.g. attention backend, structured-output backend).",
    )
    max_tokens_per_example: int | None = Field(
        default=None,
        description="Resolver hint: per-example max-tokens budget at capture time. Drives ``max_model_len`` resolution.",
    )


class BenchmarkPrompt(BaseModel):
    """One captured prompt + its original sampling params + reference output.

    The harness replays ``prompt`` through ``LLM.generate(...)`` using
    a SamplingParams built from ``original_sampling_params`` + the
    candidate's overrides. ``original_output_text`` is preserved for
    qualitative diffing but isn't used by the runner directly.
    """

    model_config = ConfigDict(extra="forbid")

    row_index: int = Field(description="Index in the original capture, surfaced for forensics.")
    prompt: str = Field(description="The exact prompt text submitted to vLLM at capture time.")
    original_sampling_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Sampling params recorded at capture (temperature, top_p, etc.).",
    )
    expected_finish_reason: str | None = Field(
        default=None,
        description="The finish reason the engine returned at capture time; informational only.",
    )
    original_output_text: str = Field(
        default="",
        description="The text the engine generated at capture time. Preserved for diffing; not consumed by the runner.",
    )


class BenchmarkCorpus(BaseModel):
    """One captured workload corpus, loaded from a trace JSONL.

    Use :meth:`from_trace_jsonl` to load. The header line establishes
    the model + LoRA + dataset_schema; every subsequent record line is
    a :class:`BenchmarkPrompt`.
    """

    model_config = ConfigDict(extra="forbid")

    header: TraceHeader
    prompts: list[BenchmarkPrompt]

    @classmethod
    def from_trace_jsonl(cls, path: str | Path) -> BenchmarkCorpus:
        """Load a corpus from a JSONL file containing one ``header`` + many ``record`` lines."""
        path = Path(path)
        header: TraceHeader | None = None
        prompts: list[BenchmarkPrompt] = []
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                payload = json.loads(line)
                kind = payload.pop("kind", None)
                if kind == "header":
                    if header is not None:
                        raise ValueError(f"{path}: duplicate header on line {line_no}")
                    header = TraceHeader.model_validate(payload)
                elif kind == "record":
                    if header is None:
                        raise ValueError(f"{path}: record on line {line_no} before any header")
                    prompts.append(
                        BenchmarkPrompt(
                            row_index=int(payload["row_index"]),
                            prompt=str(payload["prompt"]),
                            original_sampling_params=dict(payload.get("sampling_params") or {}),
                            expected_finish_reason=payload.get("finish_reason"),
                            original_output_text=str(payload.get("output_text", "")),
                        ),
                    )
                else:
                    raise ValueError(f"{path}: unknown kind={kind!r} on line {line_no}")
        if header is None:
            raise ValueError(f"{path}: missing header line")
        return cls(header=header, prompts=prompts)


class BenchmarkEngineConfig(BaseModel):
    """Engine-construction kwargs the harness forwards to ``vllm.LLM(...)``.

    Sparse: every field is optional. Unset fields fall through to vLLM's
    own defaults (or to the corpus header's engine_parameters when the
    runner builds the engine for a candidate). The runner explicitly
    drops ``None``-valued fields when assembling kwargs so vLLM treats
    them as "not configured" rather than "override to None".

    This is a benchmark-side schema, not a production config. Production
    construction lives in ``VllmBackend.initialize``. Eventually if PR-1's
    ``vllm_engine_factory.VllmEngineParameters`` lands, this can compose
    against that; for now it stands alone.
    """

    model_config = ConfigDict(extra="forbid")

    attention_backend: str | None = Field(
        default=None,
        description="vLLM attention backend (``FLASHINFER``, ``FLASH_ATTN``, ``TRITON_ATTN``, etc.). ``None`` or ``'auto'`` leaves it unset.",
    )
    structured_generation_backend: str = Field(
        default="xgrammar",
        description="Structured-outputs backend. ``'xgrammar'`` is vLLM's current default; ``'outlines'`` and ``'guidance'`` are the alternatives.",
    )
    max_model_len: int | None = Field(
        default=None,
        description="Context-window cap forwarded to ``vllm.LLM(max_model_len=...)``. ``None`` lets vLLM auto-resolve from the model.",
    )
    enable_prefix_caching: bool | None = Field(
        default=None,
        description=(
            "Forwarded to ``vllm.LLM(enable_prefix_caching=...)``. "
            "On for shared-schema tabular workloads (the prefix amortises across the batch); "
            "off when measuring per-cell cold-start behaviour. ``None`` keeps vLLM's default."
        ),
    )
    max_num_seqs: int | None = Field(default=None, description="vLLM scheduler ``max_num_seqs`` cap.")
    max_num_batched_tokens: int | None = Field(default=None, description="vLLM scheduler ``max_num_batched_tokens`` cap.")
    enable_chunked_prefill: bool | None = Field(default=None, description="Chunked-prefill engagement.")
    kv_cache_dtype: str | None = Field(
        default=None,
        description="KV-cache dtype (``'auto'``, ``'fp8'``, etc.). Halves memory footprint at a per-token quality cost when set to ``'fp8'``.",
    )
    seed: int | None = Field(
        default=None,
        description="Engine-level RNG seed (``vllm.LLM(seed=...)``). Does NOT pin per-request sampling RNG; that's ``sampling_overrides['seed']`` on :class:`BenchmarkCandidate`.",
    )
    gpu_memory_utilization: float | None = Field(
        default=None,
        description="Fraction of GPU memory vLLM may allocate. ``None`` lets the harness pick a sensible default from the host.",
    )
    speculative_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Speculative-decoding config dict forwarded verbatim to "
            "``vllm.LLM(speculative_config=...)``. ``{'method': 'ngram', "
            "'num_speculative_tokens': 4, 'prompt_lookup_max': 4}`` is the "
            "n-gram preset; ``{'method': 'eagle', 'model': '...'}`` is the "
            "draft-model preset."
        ),
    )
    compilation_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Forwarded to ``vllm.LLM(compilation_config=...)``. Controls "
            "vLLM's torch.compile / cudagraph capture settings. Useful "
            "for perf experiments that probe startup-vs-runtime tradeoffs "
            "without an ``enforce_eager`` rebuild."
        ),
    )
    kv_cache_metrics: bool | None = Field(
        default=None,
        description=(
            "Forwarded to ``vllm.LLM(kv_cache_metrics=...)`` when supported. "
            "Enables vLLM's KV-cache residency / hit-rate metrics surface; "
            "useful for additional observability beyond what "
            "``LLM.get_metrics()`` already provides."
        ),
    )


class BenchmarkCandidate(BaseModel):
    """One configuration to benchmark — engine kwargs + sampling overrides + identity.

    ``sampling_overrides`` is sparse — only fields that differ from the
    corpus default. The runner merges it on top of each prompt's
    ``original_sampling_params``. Pass ``{"seed": int}`` to pin
    ``SamplingParams.seed`` for reproducible acceptance-rate
    measurements.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Human-readable identifier, e.g. ``'baseline'`` or ``'prefix_caching=on'``.")
    engine_config: BenchmarkEngineConfig = Field(
        default_factory=BenchmarkEngineConfig,
        description="Engine-construction snapshot for this candidate.",
    )
    sampling_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description="Sparse sampling-params overlay applied on top of the corpus default.",
    )
    prompt_overrides: dict[str, Any] | None = Field(
        default=None,
        description="Reserved for future prompting-strategy variations; runner may not consume yet.",
    )
    prompt_mode: PromptAssemblyMode = Field(
        default="multi_record",
        description=(
            "Prompt-assembly regime. ``'multi_record'`` keeps the captured "
            "sampling-params unchanged. ``'per_record'`` divides ``max_tokens`` "
            "by the per-record hint so each prompt decodes a single record."
        ),
    )
    batch_dispatch_mode: BatchDispatchMode = Field(
        default="replicate",
        description="See :data:`BatchDispatchMode` module-level doc.",
    )


class CandidateMetrics(BaseModel):
    """Measured outputs for one benchmark cell.

    Carries cell-specific bench measurements (throughput, acceptance,
    TTFT, etc.) directly; observability primitives (peak VRAM, KV
    cache usage, loadavg, engine_runtime_config) are composed from
    PR-A's :class:`CellObservability` schema in the ``observability``
    field. This composition keeps the schema DRY — adding a new
    observability primitive in PR-A automatically flows through to
    benchmark output via ``model_dump()``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Matches :attr:`BenchmarkCandidate.name`.")

    # Cell-level throughput + acceptance.
    raw_tok_s: float = Field(description="Output tokens / wall seconds, ignoring validity.")
    acceptance_rate: float = Field(description="Fraction of generated records that passed validation (0..1).")
    effective_tok_s: float = Field(description="``raw_tok_s * acceptance_rate``; the operator-relevant headline.")

    # Per-request latency stats. TTFT is queue-inclusive under batched
    # submission (vLLM's ``first_token_latency`` = ``first_token_ts -
    # arrival_time`` — so prompts later in the batch contribute their
    # queue wait). Useful for spotting tail-of-batch wait time but not
    # for per-candidate comparisons under varying batch shapes.
    ttft_p50_ms: float = Field(description="Median time-to-first-token in milliseconds (queue-inclusive).")
    ttft_p99_ms: float = Field(description="99th-percentile time-to-first-token in milliseconds (queue-inclusive).")

    # Counts.
    prompts_attempted: int = Field(description="Number of corpus prompts replayed against this candidate.")
    prompts_accepted: int = Field(description="Prompts that produced at least one valid record.")
    total_output_tokens: int = Field(description="Sum of generated tokens across all replays.")
    total_wall_seconds: float = Field(description="Total wall-clock seconds spent generating for this candidate.")
    records_per_second: float = Field(
        default=0.0,
        description="Valid synthetic records produced per wall second of generation.",
    )

    # Startup / overlap accounting.
    startup_seconds: float = Field(
        default=0.0,
        description="Wall seconds the runner blocked waiting on the async engine build.",
    )
    simulate_training_overlap_seconds: float = Field(
        default=0.0,
        description="Seconds the runner slept after kicking off engine build to simulate concurrent training.",
    )
    startup_overlap_savings_seconds: float = Field(
        default=0.0,
        description="Best-effort wall-time savings from overlapping engine init with simulated training.",
    )

    # Finish-reason distribution.
    finish_reason_distribution: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of vLLM finish reasons (``stop``, ``length``, etc.) across replays.",
    )

    # Composed observability (PR-A's schema). The runner builds this from
    # ``NvmlPeakSampler.peak_gb`` + ``read_loadavg`` pre/post +
    # ``probe_engine_runtime_config`` + ``read_vllm_runtime_metrics``,
    # and sets ``flag_did_not_engage`` based on the candidate's intended
    # engine config vs the probed runtime config.
    observability: CellObservability = Field(
        default_factory=CellObservability,
        description=(
            "Cell-level observability snapshot. Composed from PR-A's schema "
            "so benchmark consumers can read e.g. ``metrics.observability."
            "peak_vram_gb`` and ``metrics.observability.kv_cache_usage_perc`` "
            "without the benchmark schema re-defining those fields."
        ),
    )


class SkipRecord(BaseModel):
    """Per-candidate failure record persisted alongside successful metrics.

    Lets a JSON consumer post-mortem which candidates failed without
    parsing the run log.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Matches the candidate's :attr:`BenchmarkCandidate.name`.")
    error: str = Field(description="Truncated stderr captured from the child subprocess.")
    error_class: str = Field(description="Best-effort exception class name parsed from stderr.")
    attempted_at: datetime = Field(description="UTC timestamp when the subprocess was launched.")


class BenchmarkOutput(BaseModel):
    """Full result of one matrix invocation — JSON-serialised for diffing."""

    model_config = ConfigDict(extra="forbid")

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp at which the run was finalised.",
    )
    corpus_run_id: str = Field(description="``run_id`` from the captured trace header.")
    corpus_size: int = Field(description="Number of prompts in the corpus (informational).")
    candidates: list[CandidateMetrics] = Field(description="Per-candidate metrics, in submission order.")
    skipped_candidates: list[SkipRecord] = Field(
        default_factory=list,
        description="Candidates that exited non-zero or produced no result file.",
    )
