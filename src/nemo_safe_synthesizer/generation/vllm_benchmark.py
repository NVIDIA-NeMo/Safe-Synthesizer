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
  acceptance, TTFT, etc. Composes :class:`GenerationObservability` from PR-A's
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
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..errors import InternalError
from ..generation.vllm_observability import (
    GenerationObservability,
    NvmlPeakSampler,
    flag_engagement_mismatches,
    probe_engine_runtime_config,
    read_loadavg,
    read_vllm_runtime_metrics,
)
from ..observability import get_logger

if TYPE_CHECKING:
    from vllm import LLM

logger = get_logger(__name__)

# Fields the corpus carries on each prompt's ``original_sampling_params``
# that aren't valid kwargs to ``vllm.SamplingParams``. Stripped during
# the merge so the harness's SamplingParams constructor doesn't reject
# capture-time-only metadata.
_NON_SAMPLING_FIELDS: tuple[str, ...] = ("structured_outputs",)

SUBPROCESS_STDERR_LIMIT: int = 500
"""Maximum bytes of captured stderr to record on a subprocess failure."""

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

    ``extra='ignore'`` so the CLI can validate a corpus header's raw
    ``engine_parameters`` dict into this model — header dicts may carry
    capture-time kwargs we don't expose as typed fields (those flow
    through ``_build_vllm_kwargs`` as the base layer regardless).
    """

    model_config = ConfigDict(extra="ignore")

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
    max_num_batched_tokens: int | None = Field(
        default=None, description="vLLM scheduler ``max_num_batched_tokens`` cap."
    )
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
    condition_label: str = Field(
        default="",
        description=(
            "Sweep-level condition this candidate measures, e.g. "
            "``'baseline'``, ``'n_fanout'``, ``'spec_ngram'``, ``'fp8'``. "
            "Set by the ``bracketed_ab`` preset family. Used by the "
            "cluster-conditioned analyzer to group cells by condition "
            "regardless of per-cell name suffixes."
        ),
    )
    bracket_position: int = Field(
        default=0,
        ge=0,
        description=(
            "Sequence index within a ``bracketed_ab`` cell stream "
            "(baseline_0=0, candidate_0=1, baseline_1=2, candidate_1=3, "
            "etc.). Used by the analyzer to align candidate cells with "
            "their bracketing baselines for drift detection."
        ),
    )


class CandidateMetrics(BaseModel):
    """Measured outputs for one benchmark cell.

    Carries cell-specific bench measurements (throughput, acceptance,
    TTFT, etc.) directly; observability primitives (peak VRAM, KV
    cache usage, loadavg, engine_runtime_config) are composed from
    PR-A's :class:`GenerationObservability` schema in the ``observability``
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

    # Sweep grouping — copied from BenchmarkCandidate so the analyzer
    # can read them off CandidateMetrics without re-joining to the
    # BenchmarkCandidate by name.
    condition_label: str = Field(
        default="",
        description="Copied from :attr:`BenchmarkCandidate.condition_label`.",
    )
    bracket_position: int = Field(
        default=0,
        ge=0,
        description="Copied from :attr:`BenchmarkCandidate.bracket_position`.",
    )

    # Composed observability (PR-A's schema). The runner builds this from
    # ``NvmlPeakSampler.peak_gb`` + ``read_loadavg`` pre/post +
    # ``probe_engine_runtime_config`` + ``read_vllm_runtime_metrics``,
    # and sets ``flag_did_not_engage`` based on the candidate's intended
    # engine config vs the probed runtime config.
    observability: GenerationObservability = Field(
        default_factory=GenerationObservability,
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


# ---------------------------------------------------------------------------
# Sampling-params + percentile helpers
# ---------------------------------------------------------------------------


def _merge_sampling_kwargs(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Compose ``SamplingParams`` kwargs from corpus default + candidate overrides.

    The corpus's captured ``original_sampling_params`` may carry fields
    that ``vllm.SamplingParams`` doesn't accept (e.g. structured-output
    presence summaries). Strip those unless the override explicitly
    provides them. Overrides win on conflict.
    """
    merged: dict[str, Any] = {**base, **overrides}
    for field in _NON_SAMPLING_FIELDS:
        if field not in overrides:
            merged.pop(field, None)
    return merged


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (``pct`` in [0, 100]); ``0.0`` on empty input."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (rank - lo) * (ordered[hi] - ordered[lo])


def _extract_ttft_ms(output: Any) -> float | None:
    """Read TTFT (ms) off a ``RequestOutput``; ``None`` when the engine omits metrics.

    vLLM 0.18+ exposes per-request first-token latency through
    ``RequestOutput.metrics.first_token_latency`` (seconds, populated
    only when the engine is built with ``disable_log_stats=False``).
    Older paths populated separate ``first_token_time`` / ``arrival_time``
    timestamps; fall through to those when the modern field is absent
    so corpus captures from older vLLM versions still produce TTFT.
    """
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    latency_s = getattr(metrics, "first_token_latency", None)
    if latency_s is not None:
        return max(0.0, float(latency_s) * 1000.0)
    first = getattr(metrics, "first_token_time", None)
    arrival = getattr(metrics, "arrival_time", None)
    if first is None or arrival is None:
        return None
    return max(0.0, (float(first) - float(arrival)) * 1000.0)


# ---------------------------------------------------------------------------
# Engine construction
# ---------------------------------------------------------------------------


def _build_vllm_kwargs(header: TraceHeader, engine_config: BenchmarkEngineConfig) -> dict[str, Any]:
    """Compose ``vllm.LLM(...)`` kwargs from corpus header + candidate overlay.

    The corpus header pins the model + LoRA + dataset_schema. The
    candidate's ``engine_config`` overlays sparse engine-side overrides
    (attention backend, prefix caching, scheduler caps, etc.). Unset
    candidate fields fall through to the header's ``engine_parameters``
    when those are populated, or to vLLM's own defaults otherwise.

    Drops ``None``-valued candidate fields explicitly so vLLM treats
    them as "not configured" rather than "override to None".
    """
    overlay = engine_config.model_dump(exclude_none=True)
    base = dict(header.engine_parameters)
    base.update(overlay)
    # Required-positional kwargs that aren't in BenchmarkEngineConfig:
    base["model"] = header.pretrained_model
    base.setdefault("enable_lora", header.lora_path is not None)
    # ``attention_backend`` → ``attention_config`` translation. vLLM's
    # public API takes a config dict rather than a bare string.
    attention_backend = base.pop("attention_backend", None)
    if attention_backend not in (None, "auto"):
        base["attention_config"] = {"backend": attention_backend}
    # ``structured_generation_backend`` → ``structured_outputs_config``.
    from vllm.config import (
        StructuredOutputsConfig,  # noqa: PLC0415 — lazy, vLLM is heavy
    )

    sg_backend = base.pop("structured_generation_backend", None)
    if sg_backend is not None:
        base["structured_outputs_config"] = StructuredOutputsConfig(backend=sg_backend)
    return base


@dataclass
class _EngineInitResult:
    """Typed channel for the async engine-init thread's outcome.

    Exactly one field is populated once the ``ready`` event fires:
    ``llm`` on success, ``exception`` on failure. Mutated by the worker
    thread; read by the runner after the join.
    """

    llm: LLM | None = None
    exception: BaseException | None = None


def _build_engine_async(
    header: TraceHeader,
    engine_config: BenchmarkEngineConfig,
) -> tuple[threading.Thread, threading.Event, _EngineInitResult]:
    """Start ``vllm.LLM(...)`` in a daemon thread.

    Returns ``(thread, ready_event, result)``. The :class:`_EngineInitResult`
    carries ``llm`` (set on success) or ``exception`` (set on failure).
    The runner can sleep for the simulated-training-overlap window
    before joining via ``ready_event.wait()``, measuring how much of
    the engine-init cost can be hidden behind concurrent training.
    """
    from vllm import LLM as vLLM  # noqa: PLC0415 — lazy

    kwargs = _build_vllm_kwargs(header, engine_config)
    ready = threading.Event()
    result = _EngineInitResult()

    def worker() -> None:
        try:
            result.llm = vLLM(**kwargs)
        except BaseException as exc:  # noqa: BLE001 — surface via the result object
            result.exception = exc
        finally:
            ready.set()

    thread = threading.Thread(target=worker, name="vllm-benchmark-engine-init", daemon=True)
    thread.start()
    return thread, ready, result


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


def run_benchmark(
    candidate: BenchmarkCandidate,
    corpus: BenchmarkCorpus,
    simulate_training_overlap_seconds: float = 0.0,
) -> CandidateMetrics:
    """Replay ``corpus`` against one ``candidate`` and report measured metrics.

    Builds a fresh vLLM engine per candidate so engine-construction
    knobs (attention backend, prefix caching, scheduler limits, ...)
    actually take effect. The engine build runs in a background thread
    so the runner can sleep for ``simulate_training_overlap_seconds``
    before joining; this measures how much of the cold-start cost can
    be hidden inside a training phase.

    All corpus prompts are submitted in one concurrent ``LLM.generate``
    call (matching ``VllmBackend.generate()``'s production dispatch
    shape) so the shared schema-prefix KV cache amortises across the
    batch and prefix caching can actually fire.

    Wraps the entire body in PR-A's :class:`NvmlPeakSampler` context and
    emits a composed :class:`GenerationObservability` on the returned
    :class:`CandidateMetrics`. The ``flag_did_not_engage`` bit is set
    when the engine's effective runtime config disagrees with the
    candidate's intended ``engine_config`` on any checked field.
    """
    # Lazy imports — keep this module CPU-importable.
    from vllm.lora.request import LoRARequest  # noqa: PLC0415
    from vllm.sampling_params import SamplingParams  # noqa: PLC0415

    from ..config.generate import ValidationParameters  # noqa: PLC0415
    from .processors import TabularDataProcessor  # noqa: PLC0415

    loadavg_pre = read_loadavg()
    vram_sampler = NvmlPeakSampler()
    with vram_sampler:
        overlap = max(0.0, simulate_training_overlap_seconds)
        _init_thread, engine_ready, init_result = _build_engine_async(
            corpus.header,
            candidate.engine_config,
        )
        if overlap > 0.0:
            time.sleep(overlap)

        wait_start = time.monotonic()
        engine_ready.wait()
        startup_seconds = max(0.0, time.monotonic() - wait_start)
        if init_result.exception is not None:
            raise init_result.exception
        if init_result.llm is None:
            raise InternalError("engine init thread finished without an LLM or an exception")
        llm = init_result.llm

        startup_overlap_savings_seconds = min(overlap, startup_seconds + overlap) if overlap > 0.0 else 0.0

        # Probe the engine's effective runtime config + check for
        # candidate-intent / engine-actual disagreements.
        engine_runtime_config = probe_engine_runtime_config(llm)
        intended = candidate.engine_config.model_dump(exclude_none=True)
        mismatches = flag_engagement_mismatches(intended, engine_runtime_config)
        if mismatches:
            logger.runtime.warning(
                "vllm_benchmark.flag_did_not_engage",
                extra={"ctx": {"candidate": candidate.name, "mismatches": mismatches}},
            )

        processor = TabularDataProcessor(
            corpus.header.dataset_schema,
            config=ValidationParameters(),
            tokenizer=None,
        )
        lora_request = (
            LoRARequest("lora", 1, str(corpus.header.lora_path)) if corpus.header.lora_path is not None else None
        )

        # Build SamplingParams from corpus default + candidate overrides.
        if corpus.prompts:
            base_sampling = corpus.prompts[0].original_sampling_params
        else:
            base_sampling = {}
        sampling_kwargs = _merge_sampling_kwargs(base_sampling, candidate.sampling_overrides)
        # n=1 unless the caller's override sets it (n_fanout sets it explicitly).
        sampling_kwargs.setdefault("n", 1)
        sampling_params = SamplingParams(**sampling_kwargs)

        # Dispatch.
        prompts: list[str] = [p.prompt for p in corpus.prompts]
        gen_start = time.perf_counter()
        outputs: list[Any] = (
            list(
                llm.generate(
                    prompts=prompts,
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )
            )
            if prompts
            else []
        )
        total_wall = max(time.perf_counter() - gen_start, 0.0)

        # Process outputs.
        ttft_ms_samples: list[float] = []
        finish_reasons: dict[str, int] = {}
        total_output_tokens = 0
        total_valid = 0
        total_invalid = 0
        prompts_accepted = 0
        for prompt_idx, output in enumerate(outputs):
            ttft = _extract_ttft_ms(output)
            if ttft is not None:
                ttft_ms_samples.append(ttft)
            best = output.outputs[0] if getattr(output, "outputs", None) else None
            if best is None:
                continue
            total_output_tokens += len(getattr(best, "token_ids", []) or [])
            finish_reason = str(getattr(best, "finish_reason", None) or "unknown")
            finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
            text = getattr(best, "text", "") or ""
            # ``Processor.__call__(prompt_number, text) -> ParsedResponse`` is
            # the actual interface — see ``processors.py``. ``valid_records``
            # and ``invalid_records`` are properties on ``ParsedResponse``.
            parsed = processor(prompt_idx, text)
            valid_records = parsed.valid_records
            invalid_records = parsed.invalid_records
            total_valid += len(valid_records)
            total_invalid += len(invalid_records)
            if valid_records:
                prompts_accepted += 1

        total_records = total_valid + total_invalid
        acceptance = (total_valid / total_records) if total_records > 0 else 0.0
        raw_tok_s = (total_output_tokens / total_wall) if total_wall > 0 else 0.0
        records_per_second = (total_valid / total_wall) if total_wall > 0 else 0.0

    # The ``with`` block above shut the sampler down, so ``peak_gb`` is now
    # final. ``read_vllm_runtime_metrics`` is hardened to never raise and
    # always returns a stable ``VllmRuntimeMetrics`` (None-valued fields on
    # degrade), so no guard is needed here.
    vllm_metrics = read_vllm_runtime_metrics(llm)

    observability = GenerationObservability(
        peak_vram_gb=vram_sampler.peak_gb,
        kv_cache_usage_perc=vllm_metrics["kv_cache_usage_perc"],
        prefix_cache_hit_rate=vllm_metrics["prefix_cache_hit_rate"],
        spec_accept_rate=vllm_metrics["spec_accept_rate"],
        loadavg_pre=loadavg_pre,
        loadavg_post=read_loadavg(),
        engine_runtime_config=engine_runtime_config,
        flag_did_not_engage=bool(mismatches),
    )

    return CandidateMetrics(
        name=candidate.name,
        raw_tok_s=raw_tok_s,
        acceptance_rate=acceptance,
        effective_tok_s=raw_tok_s * acceptance,
        ttft_p50_ms=_percentile(ttft_ms_samples, 50),
        ttft_p99_ms=_percentile(ttft_ms_samples, 99),
        prompts_attempted=len(corpus.prompts),
        prompts_accepted=prompts_accepted,
        total_output_tokens=total_output_tokens,
        total_wall_seconds=total_wall,
        records_per_second=records_per_second,
        startup_seconds=startup_seconds,
        simulate_training_overlap_seconds=overlap,
        startup_overlap_savings_seconds=startup_overlap_savings_seconds,
        finish_reason_distribution=finish_reasons,
        condition_label=candidate.condition_label,
        bracket_position=candidate.bracket_position,
        observability=observability,
    )


# ---------------------------------------------------------------------------
# Subprocess isolation
# ---------------------------------------------------------------------------


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


def run_benchmark_in_subprocess(
    candidate: BenchmarkCandidate,
    corpus_path: str | Path,
    simulate_training_overlap_seconds: float = 0.0,
) -> SubprocessRunResult:
    """Run one candidate in a child process the OS reclaims on exit.

    Spawns ``python -m nemo_safe_synthesizer.generation.vllm_benchmark_single_run``
    with the candidate JSON-encoded as argv. Child writes
    ``CandidateMetrics`` JSON to a temp file; parent reads it back.

    Subprocess isolation is what makes a multi-candidate matrix
    reliable on this stack: vLLM holds significant CUDA + DRAM state
    in module-level globals that the in-process Python runtime can't
    clean up between candidates. Each candidate runs in a fresh
    interpreter; the OS reclaims everything on child exit.
    """
    candidate_json = candidate.model_dump_json()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as result_fh:
        result_path = Path(result_fh.name)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "nemo_safe_synthesizer.generation.vllm_benchmark_single_run",
                "--candidate",
                candidate_json,
                "--corpus",
                str(corpus_path),
                "--result-out",
                str(result_path),
                "--simulate-training-overlap-seconds",
                str(simulate_training_overlap_seconds),
            ],
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
    finally:
        result_path.unlink(missing_ok=True)
