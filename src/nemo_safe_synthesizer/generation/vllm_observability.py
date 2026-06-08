# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM observability primitives for production + benchmark.

Schema-frozen generation-observability events emitted by ``VllmBackend.generate()``
and consumed by downstream observability surfaces (structured logs, wandb,
the benchmark harness's per-cell aggregator).

Four primitives, all degraded-mode by design:

- :class:`NvmlPeakSampler` — daemon-thread peak device-VRAM tracker via
  ``pynvml``. Reads at the driver layer so it sees vLLM worker-subprocess
  allocations regardless of which process holds the torch handle —
  sidesteps the ``VLLM_ENABLE_V1_MULTIPROCESSING=1`` blind spot where
  ``torch.cuda.max_memory_allocated()`` in the harness reads 0.
- :func:`read_loadavg` — ``/proc/loadavg`` snapshot as a (1m, 5m, 15m)
  triple. ``None`` on non-Linux or read failure.
- :func:`probe_engine_runtime_config` — best-effort introspection of the
  engine's effective scheduler/cache/speculative settings via
  ``llm.llm_engine.vllm_config``. Empty dict on any failure.
- :func:`read_vllm_runtime_metrics` — one-shot snapshot of
  ``llm.get_metrics()`` for KV-cache usage, prefix-cache hit rate, and
  speculative-decoding acceptance rate. Returns a dict with stable keys
  regardless of which metrics the engine actually exposed.

Plus the :class:`GenerationObservability` pydantic model — the schema for the
generation-complete structured event emitted at the end of each
generation invocation. Forward-compatible: new optional fields can be
added without breaking existing consumers because the model uses
``extra="forbid"`` (so producers are forced to update when they add new
fields) but every existing field has a default of ``None`` or empty.

References (design):
- HuggingFace ``train_memory`` blog ("Visualize and understand GPU memory
  in PyTorch") covers in-process torch profiling, which is the gap NVML
  fills here (out-of-process VRAM visibility):
  https://huggingface.co/blog/train_memory
- spark-dashboard (Rust) demonstrates the NVML + ``/metrics`` pattern at
  ~1s polling cadence — the precedent for combining NVML's driver-level
  reading with vLLM's Prometheus surface:
  https://github.com/niklasfrick/spark-dashboard
- vLLM's metrics design doc enumerates the gauges/counters
  :func:`read_vllm_runtime_metrics` reads:
  https://github.com/vllm-project/vllm/blob/main/docs/design/metrics.md
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from ..observability import (
    NvmlPeakSampler,
    _default_nvml_device_index,
    get_logger,
    read_loadavg,
)

if TYPE_CHECKING:
    from vllm import LLM

__all__ = [
    "ENGINE_CONFIG_CHECKED_FIELDS",
    "METRIC_KV_CACHE_USAGE_PERC",
    "METRIC_PREFIX_CACHE_HITS",
    "METRIC_PREFIX_CACHE_QUERIES",
    "METRIC_SPEC_NUM_ACCEPTED_TOKENS",
    "METRIC_SPEC_NUM_DRAFT_TOKENS",
    "GenerationObservability",
    "NvmlPeakSampler",
    "VllmRuntimeMetrics",
    "_default_nvml_device_index",
    "flag_engagement_mismatches",
    "probe_engine_runtime_config",
    "read_loadavg",
    "read_vllm_runtime_metrics",
]

logger = get_logger(__name__)


_LOADAVG_HORIZON_LABELS: tuple[str, str, str] = ("1m", "5m", "15m")
"""Labels for unpacked ``/proc/loadavg`` triples on the wandb side.

Used by :meth:`GenerationObservability.to_wandb_payload`.
"""


# ---------------------------------------------------------------------------
# Schema for the generation-observability event
# ---------------------------------------------------------------------------


class GenerationObservability(BaseModel):
    """One generation-complete event payload.

    Emitted by ``VllmBackend.generate()`` at end of each generation
    invocation. Consumed by:

    - Structured log routing (default — flows through
      ``logger.runtime.info(...)`` like the rest of PR-1's trace
      telemetry).
    - Wandb (when a run is active) — logged to the current wandb run.
    - The benchmark harness's per-cell aggregator (composes this into
      its richer ``CandidateMetrics`` schema).

    Every measurement field is optional; producers should populate what
    they can capture and leave the rest at the default. Wandb drops
    ``None`` values silently which is the right behavior for "this
    metric wasn't reachable on this generation".
    """

    model_config = ConfigDict(extra="forbid")

    # GPU memory — NVML-sampled peak across the whole generation.
    peak_vram_gb: float | None = Field(
        default=None,
        description=(
            "Peak device-wide VRAM usage in GiB, sampled by NVML "
            "(``pynvml.nvmlDeviceGetMemoryInfo``) across the whole generation. "
            "``None`` when NVML is unavailable. Device-wide reading; on a "
            "shared GPU it includes other processes."
        ),
    )

    # KV cache + prefix cache — vLLM's Prometheus surface, end-of-generate.
    kv_cache_usage_perc: float | None = Field(
        default=None,
        description=(
            "vLLM's ``vllm:kv_cache_usage_perc`` gauge (fraction 0..1 of "
            "KV cache blocks in use) at end of generation. ``None`` when "
            "the engine doesn't expose the gauge or the call failed. "
            "Approximates peak; vLLM only publishes the instantaneous "
            "value, not a max-over-time."
        ),
    )
    prefix_cache_hit_rate: float | None = Field(
        default=None,
        description=(
            "Derived from ``vllm:prefix_cache_hits / vllm:prefix_cache_queries`` "
            "at end of generation. ``None`` when either counter is absent or "
            "queries==0. Surfaces whether shared schema prefixes actually "
            "amortized across the batch."
        ),
    )

    # Speculative decoding — vLLM's Prometheus surface, end-of-generate.
    spec_accept_rate: float | None = Field(
        default=None,
        description=(
            "Derived from ``vllm:spec_decode_num_accepted_tokens / "
            "num_draft_tokens`` at end of generation. ``None`` when "
            "speculative decoding wasn't enabled on this generation (counters "
            "absent) or no drafts were proposed (denominator==0)."
        ),
    )

    # Host load — /proc/loadavg pre/post.
    loadavg_pre: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Host ``/proc/loadavg`` snapshot captured at the start of "
            "this generation (1-min, 5-min, 15-min averages). ``None`` when "
            "``/proc/loadavg`` is unavailable (non-Linux)."
        ),
    )
    loadavg_post: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Host ``/proc/loadavg`` snapshot captured at the end of this "
            "generation. Drift from ``loadavg_pre`` signals load change during "
            "the generation."
        ),
    )

    # Engine config introspection — what the engine actually does, not what was asked.
    engine_runtime_config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Best-effort probe of the engine's effective runtime config "
            "(``enable_prefix_caching``, ``enable_chunked_prefill``, "
            "``max_num_seqs``, ``max_num_batched_tokens``, "
            "``kv_cache_dtype``, ``speculative_method`` when populated). "
            "Empty dict on probe failure."
        ),
    )
    flag_did_not_engage: bool = Field(
        default=False,
        description=(
            "``True`` when ``engine_runtime_config`` disagrees with the "
            "candidate/caller's intended setting on any checked field — "
            "an unsupported knob silently ignored, a default-on flag "
            "overriding an explicit-off intent, etc."
        ),
    )

    def to_wandb_payload(self, prefix: str = "vllm_gen") -> dict[str, Any]:
        """Flatten this event into a wandb-friendly ``wandb.log(...)`` dict.

        Wandb plots scalars cleanly but renders tuples/dicts as opaque
        blobs, so this method:

        - Drops ``None`` values (wandb would drop them anyway; explicit
          here for documentation).
        - Unpacks ``loadavg_pre`` / ``loadavg_post`` 3-tuples to per-
          duration scalars (``loadavg_pre_1m`` / ``_5m`` / ``_15m``).
        - Flattens ``engine_runtime_config`` to ``engine_runtime/<key>``
          scalars (mirrors the existing flattening pattern in the
          benchmark harness).

        All keys are namespaced under ``prefix`` so production generation
        events don't collide with other wandb metrics in the same run.
        """
        payload: dict[str, Any] = {}
        for scalar_field in (
            "peak_vram_gb",
            "kv_cache_usage_perc",
            "prefix_cache_hit_rate",
            "spec_accept_rate",
            "flag_did_not_engage",
        ):
            value = getattr(self, scalar_field)
            if value is not None:
                payload[f"{prefix}/{scalar_field}"] = value
        for side in ("pre", "post"):
            tup = getattr(self, f"loadavg_{side}")
            if tup is None:
                continue
            for label, value in zip(_LOADAVG_HORIZON_LABELS, tup, strict=False):
                payload[f"{prefix}/loadavg_{side}_{label}"] = value
        for key, value in self.engine_runtime_config.items():
            if value is not None:
                payload[f"{prefix}/engine_runtime/{key}"] = value
        return payload


# ---------------------------------------------------------------------------
# Engine config introspection
# ---------------------------------------------------------------------------


def _identity(value: Any) -> Any:
    """Default field transform — return the probed value unchanged."""
    return value


@dataclass(frozen=True)
class _ProbeField:
    """One extraction rule for :func:`probe_engine_runtime_config`.

    Declarative replacement for per-field control flow: each row says where
    to read (``section`` + precedence-ordered ``sources``), what to call the
    result (``out_key``), how to normalize it (``transform``), and whether
    the flag-engagement check compares it (``checked``).
    """

    section: str
    """``vllm_config`` sub-object attribute, e.g. ``"scheduler_config"``."""
    sources: tuple[str, ...]
    """Candidate source attribute names, modern-name-first precedence."""
    out_key: str
    """Key under which the value is surfaced in the probe result."""
    transform: Callable[[Any], Any] = _identity
    """Normalizer applied to the first present source value."""
    checked: bool = True
    """Whether :data:`ENGINE_CONFIG_CHECKED_FIELDS` includes this field."""


# The single source of truth for what the probe extracts. Adding a probed
# field is one row here — no new control flow. ``sources`` ordering encodes
# vLLM's cross-version attribute renames (modern name first, legacy fallback).
_PROBE_FIELDS: tuple[_ProbeField, ...] = (
    _ProbeField("scheduler_config", ("max_num_seqs",), "max_num_seqs"),
    _ProbeField("scheduler_config", ("max_num_batched_tokens",), "max_num_batched_tokens"),
    _ProbeField(
        "scheduler_config",
        ("chunked_prefill_enabled", "enable_chunked_prefill"),
        "enable_chunked_prefill",
        transform=bool,
    ),
    _ProbeField("cache_config", ("enable_prefix_caching",), "enable_prefix_caching"),
    _ProbeField("cache_config", ("kv_cache_dtype", "cache_dtype"), "kv_cache_dtype"),
    # Probed for observability but excluded from the flag-engagement check.
    _ProbeField("speculative_config", ("method",), "speculative_method", checked=False),
)


# Engine-config fields the flag-engagement check compares intended-vs-actual.
# Derived from the probe table so the two cannot drift: a field is checked iff
# the probe can surface it and the row opts in via ``checked=True``.
ENGINE_CONFIG_CHECKED_FIELDS: tuple[str, ...] = tuple(f.out_key for f in _PROBE_FIELDS if f.checked)


def _engine_vllm_config(llm: object) -> Any | None:
    """Resolve ``llm.llm_engine.vllm_config`` (or the v0 ``llm.engine`` name).

    Pure ``getattr`` traversal; returns ``None`` when any link is absent.
    """
    engine = getattr(llm, "llm_engine", None) or getattr(llm, "engine", None)
    return getattr(engine, "vllm_config", None) if engine is not None else None


def _first_present(obj: object, names: tuple[str, ...]) -> Any | None:
    """Return the first non-``None`` attribute among ``names``, else ``None``."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def probe_engine_runtime_config(llm: object) -> dict[str, Any]:
    """Best-effort introspection of the engine's effective runtime config.

    Returns a flat dict of the load-bearing scheduler/cache/speculative
    settings drawn from :data:`_PROBE_FIELDS`. Empty dict when the engine
    config can't be reached — this is observability, not a correctness gate.

    Degrades at field granularity: a malformed individual attribute skips
    that one field rather than emptying the whole result.

    Typed ``object`` (not ``LLM``) on purpose: the probe is pure defensive
    ``getattr`` introspection and degrades on any shape, so it does not
    require — and must not claim to require — the concrete engine type.
    """
    try:
        vcfg = _engine_vllm_config(llm)
    except Exception:  # noqa: BLE001 — degraded mode by design
        logger.debug("engine-probe: vllm_config unreachable; returning empty probe", exc_info=True)
        return {}
    if vcfg is None:
        return {}
    out: dict[str, Any] = {}
    for spec in _PROBE_FIELDS:
        try:
            section = getattr(vcfg, spec.section, None)
            if section is None:
                continue
            value = _first_present(section, spec.sources)
            if value is not None:
                out[spec.out_key] = spec.transform(value)
        except Exception:  # noqa: BLE001 — degrade one field, not the whole probe
            logger.debug("engine-probe: field %r failed; skipping", spec.out_key, exc_info=True)
            continue
    return out


def flag_engagement_mismatches(
    intended: dict[str, Any],
    actual: dict[str, Any],
    checked_fields: tuple[str, ...] = ENGINE_CONFIG_CHECKED_FIELDS,
) -> list[str]:
    """Return human-readable mismatch descriptions; empty list means clean engagement.

    Only checks fields the caller explicitly set in ``intended`` (i.e.,
    fields whose value is not ``None``); a ``None`` on the intended side
    means "use engine default" so there's no reference value to compare
    against. Fields missing from ``actual`` are skipped — the probe is
    best-effort and may not expose every flag.

    The dict-vs-dict shape (rather than a typed pydantic model) is
    deliberate so this helper works regardless of whether the caller
    has a ``VllmEngineParameters`` instance or just raw vLLM kwargs.
    """
    mismatches: list[str] = []
    for field in checked_fields:
        intended_val = intended.get(field)
        if intended_val is None:
            continue
        actual_val = actual.get(field)
        if actual_val is None:
            continue
        if intended_val != actual_val:
            mismatches.append(f"{field}: intended={intended_val!r} actual={actual_val!r}")
    return mismatches


# ---------------------------------------------------------------------------
# vLLM runtime metrics
# ---------------------------------------------------------------------------


# vLLM metric names this module knows about. Kept module-level so a future
# renaming in vLLM only needs one update site.
METRIC_KV_CACHE_USAGE_PERC = "vllm:kv_cache_usage_perc"
METRIC_PREFIX_CACHE_QUERIES = "vllm:prefix_cache_queries"
METRIC_PREFIX_CACHE_HITS = "vllm:prefix_cache_hits"
METRIC_SPEC_NUM_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens"
METRIC_SPEC_NUM_ACCEPTED_TOKENS = "vllm:spec_decode_num_accepted_tokens"

# Raw counters pulled from ``llm.get_metrics()``. The output fields below are
# either passed through (kv-cache usage) or derived as ratios of these.
_COLLECTED_METRICS: frozenset[str] = frozenset(
    {
        METRIC_KV_CACHE_USAGE_PERC,
        METRIC_PREFIX_CACHE_QUERIES,
        METRIC_PREFIX_CACHE_HITS,
        METRIC_SPEC_NUM_DRAFT_TOKENS,
        METRIC_SPEC_NUM_ACCEPTED_TOKENS,
    }
)


class VllmRuntimeMetrics(TypedDict):
    """End-of-generation vLLM metric snapshot with a fixed key set.

    Every value is ``float | None``; ``None`` means the engine did not
    surface that counter on this generation (distinct from a measured zero).

    A ``TypedDict`` rather than a dataclass on purpose: the value stays a
    plain ``dict`` at runtime, so dict-style consumers (e.g. the benchmark
    harness) are unaffected, while callers gain static key checking and
    ``float | None`` value typing instead of ``dict[str, float | None]``.
    """

    kv_cache_usage_perc: float | None
    prefix_cache_hit_rate: float | None
    spec_accept_rate: float | None


def _empty_runtime_metrics() -> VllmRuntimeMetrics:
    """The degraded-mode snapshot: every field ``None`` (nothing measured)."""
    return VllmRuntimeMetrics(kv_cache_usage_perc=None, prefix_cache_hit_rate=None, spec_accept_rate=None)


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    """``numerator / denominator``, or ``None`` when either is absent or the denominator is non-positive."""
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _collect_raw_metrics(llm: LLM) -> dict[str, float]:
    """Read ``llm.get_metrics()`` into ``{name: value}`` for the counters in :data:`_COLLECTED_METRICS`.

    Isolates the only failure-prone work (the engine call plus numeric
    coercion) so the caller's degraded-mode guard can wrap exactly this.
    """
    raw: dict[str, float] = {}
    for m in llm.get_metrics():
        name = getattr(m, "name", "")
        value = getattr(m, "value", None)
        if value is not None and name in _COLLECTED_METRICS:
            raw[name] = float(value)
    return raw


def read_vllm_runtime_metrics(llm: LLM | None) -> VllmRuntimeMetrics:
    """Snapshot ``llm.get_metrics()`` for known metrics; degraded-mode on failure.

    Returns a :class:`VllmRuntimeMetrics` with stable keys regardless of
    which metrics the engine actually exposed — missing metrics map to
    ``None``. Callers should treat ``None`` as "engine didn't surface this
    counter" and not crash on missing data.

    Currently captures:

    - ``kv_cache_usage_perc`` — vLLM gauge, fraction (0..1) of used KV
      cache blocks at the moment of read.
    - ``prefix_cache_hit_rate`` — derived from
      ``vllm:prefix_cache_hits / vllm:prefix_cache_queries``.
    - ``spec_accept_rate`` — derived from
      ``vllm:spec_decode_num_accepted_tokens / num_draft_tokens``.
      ``None`` when speculative decoding wasn't enabled (counters
      registered at runtime by the spec-decode subsystem; absent
      otherwise) — distinguishes "not measured" from "measured zero".
    """
    if llm is None:
        return _empty_runtime_metrics()

    # Best-effort probe: the engine call or a non-numeric metric value is the
    # only thing that can fail, so the guard wraps exactly that. The ratio
    # derivation below is pure dict arithmetic and cannot raise.
    try:
        raw = _collect_raw_metrics(llm)
    except Exception:  # noqa: BLE001 — degraded mode by design
        logger.debug("read_vllm_runtime_metrics failed", exc_info=True)
        return _empty_runtime_metrics()

    return VllmRuntimeMetrics(
        kv_cache_usage_perc=raw.get(METRIC_KV_CACHE_USAGE_PERC),
        prefix_cache_hit_rate=_safe_ratio(raw.get(METRIC_PREFIX_CACHE_HITS), raw.get(METRIC_PREFIX_CACHE_QUERIES)),
        spec_accept_rate=_safe_ratio(raw.get(METRIC_SPEC_NUM_ACCEPTED_TOKENS), raw.get(METRIC_SPEC_NUM_DRAFT_TOKENS)),
    )
