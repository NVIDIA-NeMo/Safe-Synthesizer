# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM observability primitives for production + benchmark.

Schema-frozen cell-observability events emitted by ``VllmBackend.generate()``
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

Plus the :class:`CellObservability` pydantic model — the schema for the
``vllm.cell.complete`` structured event emitted at the end of each
generation invocation. Forward-compatible: new optional fields can be
added without breaking existing consumers because the model uses
``extra="forbid"`` (so producers are forced to update when they add new
fields) but every existing field has a default of ``None`` or empty.

References (design):
- HuggingFace train_memory blog covers in-process torch profiling, which
  is the gap NVML fills here (out-of-process VRAM visibility).
- spark-dashboard (Rust) demonstrates the NVML + ``/metrics`` pattern at
  ~1s polling cadence — the precedent for combining NVML's driver-level
  reading with vLLM's Prometheus surface.
- vLLM's metrics design doc enumerates the gauges/counters
  :func:`read_vllm_runtime_metrics` reads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from ..observability import get_logger

if TYPE_CHECKING:
    from vllm import LLM

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema for the cell-observability event
# ---------------------------------------------------------------------------


class CellObservability(BaseModel):
    """One ``vllm.cell.complete`` event payload.

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
    metric wasn't reachable on this cell".
    """

    model_config = ConfigDict(extra="forbid")

    # GPU memory — NVML-sampled peak across the whole cell.
    peak_vram_gb: float | None = Field(
        default=None,
        description=(
            "Peak device-wide VRAM usage in GiB, sampled by NVML "
            "(``pynvml.nvmlDeviceGetMemoryInfo``) across the whole cell. "
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
            "speculative decoding wasn't enabled on this cell (counters "
            "absent) or no drafts were proposed (denominator==0)."
        ),
    )

    # Host load — /proc/loadavg pre/post.
    loadavg_pre: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Host ``/proc/loadavg`` snapshot captured at the start of "
            "this cell (1-min, 5-min, 15-min averages). ``None`` when "
            "``/proc/loadavg`` is unavailable (non-Linux)."
        ),
    )
    loadavg_post: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Host ``/proc/loadavg`` snapshot captured at the end of this "
            "cell. Drift from ``loadavg_pre`` signals load change during "
            "the cell."
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

    def to_wandb_payload(self, prefix: str = "vllm_cell") -> dict[str, Any]:
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

        All keys are namespaced under ``prefix`` so production cell
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
            payload[f"{prefix}/engine_runtime/{key}"] = value
        return payload


_LOADAVG_HORIZON_LABELS: tuple[str, str, str] = ("1m", "5m", "15m")
"""Labels for unpacked ``/proc/loadavg`` triples on the wandb side.

Used by :meth:`CellObservability.to_wandb_payload`.
"""


# ---------------------------------------------------------------------------
# NVML peak sampler
# ---------------------------------------------------------------------------


class NvmlPeakSampler:
    """Daemon-thread sampler tracking peak device VRAM via NVML.

    Use as a context manager wrapping the engine-build + generation
    block::

        with NvmlPeakSampler() as vram:
            ...  # build engine, run generate
        peak_gb = vram.peak_gb  # float | None

    Returns ``None`` from :attr:`peak_gb` when NVML isn't available
    (driver missing, pynvml import failed, device index invalid).
    Reports device-wide VRAM — on a dedicated host that's the same as
    the vLLM worker's allocation; on a shared GPU it would include
    other process allocations (filter via PID externally if needed).
    """

    def __init__(self, device_index: int = 0, interval_seconds: float = 0.25) -> None:
        self._device_index = device_index
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._peak_bytes = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._handle: Any = None
        self._pynvml: Any = None

    def __enter__(self) -> NvmlPeakSampler:
        try:
            import pynvml  # noqa: PLC0415 — soft dep, no top-level cost
        except ImportError:
            logger.debug("nvml-sampler: pynvml unavailable; peak VRAM will be None")
            return self
        try:
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._device_index)
        except pynvml.NVMLError as exc:  # driver missing, bad index, etc. — degraded mode
            logger.warning("nvml-sampler: init failed; peak VRAM will be None: %s", exc)
            self._pynvml = None
            return self
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"nvml-sampler[{self._device_index}]")
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except self._pynvml.NVMLError:
                pass

    @property
    def peak_gb(self) -> float | None:
        """Peak device-wide VRAM (GiB) observed during sampling; ``None`` if NVML unavailable."""
        if self._pynvml is None:
            return None
        with self._lock:
            return self._peak_bytes / (1024**3)

    def _run(self) -> None:
        """Poll loop. Tolerates transient NVML errors without dying."""
        assert self._pynvml is not None
        while not self._stop.is_set():
            try:
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used = int(info.used)
                with self._lock:
                    if used > self._peak_bytes:
                        self._peak_bytes = used
            except self._pynvml.NVMLError:  # transient driver hiccup — keep sampling
                pass
            self._stop.wait(self._interval)


# ---------------------------------------------------------------------------
# Host load
# ---------------------------------------------------------------------------


def read_loadavg() -> tuple[float, float, float] | None:
    """Return ``/proc/loadavg`` as a (1m, 5m, 15m) triple; ``None`` when unavailable.

    Linux-only. Cheap (one syscall). Safe to call from any process —
    the read is host-scoped, not process-scoped. Designed to bracket a
    generation call: caller reads pre + post, the pair is informative
    about whether host load drifted during the cell.
    """
    try:
        with open("/proc/loadavg", encoding="utf-8") as f:
            parts = f.read().split()
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except (OSError, ValueError, IndexError):
        return None


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


class VllmRuntimeMetrics(TypedDict):
    """End-of-generation vLLM metric snapshot with a fixed key set.

    Every value is ``float | None``; ``None`` means the engine did not
    surface that counter on this cell (distinct from a measured zero).

    A ``TypedDict`` rather than a dataclass on purpose: the value stays a
    plain ``dict`` at runtime, so dict-style consumers (e.g. the benchmark
    harness) are unaffected, while callers gain static key checking and
    ``float | None`` value typing instead of ``dict[str, float | None]``.
    """

    kv_cache_usage_perc: float | None
    prefix_cache_hit_rate: float | None
    spec_accept_rate: float | None


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
    out: VllmRuntimeMetrics = {
        "kv_cache_usage_perc": None,
        "prefix_cache_hit_rate": None,
        "spec_accept_rate": None,
    }
    if llm is None:
        return out

    # Whole-body guard: this is a best-effort probe, so any failure (the
    # engine call, an unexpected metric shape, a non-numeric value) degrades
    # to the stable all-``None`` dict rather than propagating. Callers can
    # therefore trust the return contract without re-wrapping the call.
    try:
        kv_usage: float | None = None
        prefix_hits: float | None = None
        prefix_queries: float | None = None
        spec_accepted: float | None = None
        spec_drafted: float | None = None
        for m in llm.get_metrics():
            name = getattr(m, "name", "")
            value = getattr(m, "value", None)
            if value is None:
                continue
            if name == METRIC_KV_CACHE_USAGE_PERC:
                kv_usage = float(value)
            elif name == METRIC_PREFIX_CACHE_HITS:
                prefix_hits = float(value)
            elif name == METRIC_PREFIX_CACHE_QUERIES:
                prefix_queries = float(value)
            elif name == METRIC_SPEC_NUM_ACCEPTED_TOKENS:
                spec_accepted = float(value)
            elif name == METRIC_SPEC_NUM_DRAFT_TOKENS:
                spec_drafted = float(value)

        if kv_usage is not None:
            out["kv_cache_usage_perc"] = kv_usage
        if prefix_hits is not None and prefix_queries is not None and prefix_queries > 0:
            out["prefix_cache_hit_rate"] = prefix_hits / prefix_queries
        if spec_accepted is not None and spec_drafted is not None and spec_drafted > 0:
            out["spec_accept_rate"] = spec_accepted / spec_drafted
    except Exception as exc:  # noqa: BLE001 — degraded mode by design
        logger.debug("read_vllm_runtime_metrics failed: %s", exc)
    return out
