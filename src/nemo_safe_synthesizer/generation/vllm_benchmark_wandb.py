# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WandB metrics sink for vLLM benchmark runs.

Each benchmark candidate run becomes one WandB run (grouped by sweep ID, tagged
by condition_label + dataset). Distinct from PR-A's production
``log_observability_event`` pattern, which logs to the *currently
active* WandB run: production logs generations as a time-series within one
run, while benchmark mode logs each candidate run as its own run for per-condition
isolation in the wandb UI.

Reuses :class:`WandbSettings` from ``nemo_safe_synthesizer.cli.wandb_setup``
so env-var handling (``WANDB_MODE`` defaults to ``disabled``,
``WANDB_PROJECT`` / ``NSS_WANDB_PROJECT`` precedence) matches the rest
of the pipeline.

Soft dependency: wandb is observability, not a hard requirement. Any
failure (missing package, missing netrc, ``wandb.init`` exception)
logs a warning and the harness continues; the benchmark JSON output
is still written.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..cli.wandb_setup import WandbMode, WandbPhase, WandbSettings
from ..observability import get_logger

if TYPE_CHECKING:
    from .vllm_benchmark import CandidateMetrics

logger = get_logger(__name__)

# A benchmark candidate run is structurally a GENERATE invocation that's
# measured rather than consumed. Phase aligns with production GENERATE
# runs; job_type distinguishes benchmark from production at the
# wandb-UI level.
_BENCHMARK_JOB_TYPE: str = "benchmark"


def resolve_sweep_id() -> str:
    """Resolve the wandb group identifier for the current sweep.

    Reads ``WANDB_RUN_GROUP`` when set (the orchestrator sets this once
    per sweep to group all candidate runs). Falls back to an auto-generated
    timestamp so single-run invocations don't all collapse into one
    bucket.
    """
    return os.environ.get("WANDB_RUN_GROUP") or f"sweep-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"


def init_cell_run(
    *,
    candidate_name: str,
    candidate_idx: int,
    total: int,
    corpus_run_id: str,
    corpus_size: int,
    sweep_id: str,
    candidate_condition_label: str = "",
    candidate_bracket_position: int = 0,
) -> Any:
    """Open one WandB run for a single benchmark candidate run.

    Returns the run object on success, ``None`` when wandb is disabled,
    misconfigured, or the package is unavailable. Callers must treat
    ``None`` as "no wandb"; subsequent log/finish helpers are no-ops
    on ``None``.

    Precedence for ``condition_label`` / ``bracket_position``:
    candidate-carried values win when non-empty/non-zero; otherwise
    falls back to ``BENCHMARK_CONDITION_LABEL`` /
    ``BENCHMARK_BRACKET_POSITION`` env vars; finally falls back to
    ``candidate_name`` / ``candidate_idx - 1``. The candidate-carried
    path is set by the ``bracketed_ab`` preset family.

    Auth via ``~/.netrc`` (set up by ``wandb login``).
    ``WANDB_API_KEY`` is NOT read; passing the key via env leaks it
    via ``ps auxe``.
    """
    settings = WandbSettings()
    if settings.wandb_mode == WandbMode.DISABLED:
        return None
    try:
        import wandb  # noqa: PLC0415 - soft dependency
    except ImportError:
        logger.warning("wandb not installed; skipping wandb metrics sink for this candidate run")
        return None

    condition_label = candidate_condition_label or os.environ.get("BENCHMARK_CONDITION_LABEL") or candidate_name
    dataset = os.environ.get("BENCHMARK_DATASET", "unknown")
    bracket_position = (
        candidate_bracket_position
        if candidate_bracket_position > 0
        else int(os.environ.get("BENCHMARK_BRACKET_POSITION", str(candidate_idx - 1)))
    )
    tags = [t for t in (condition_label, dataset, _BENCHMARK_JOB_TYPE) if t and t != "unknown"]
    try:
        return wandb.init(
            project=settings.effective_wandb_project,
            name=candidate_name,
            group=sweep_id,
            job_type=_BENCHMARK_JOB_TYPE,
            tags=tags,
            mode=settings.wandb_mode.value,
            reinit=True,
            config={
                "candidate_name": candidate_name,
                "candidate_idx": candidate_idx,
                "total_candidates": total,
                "corpus_run_id": corpus_run_id,
                "corpus_size": corpus_size,
                "condition_label": condition_label,
                "dataset": dataset,
                "bracket_position": bracket_position,
                "sweep_id": sweep_id,
                "phase": WandbPhase.GENERATE.value,
            },
        )
    except Exception as exc:  # noqa: BLE001 - degraded mode by design
        logger.warning("wandb.init failed; continuing without wandb", exc_info=exc)
        return None


def _flatten_metrics(metrics: CandidateMetrics) -> dict[str, Any]:
    """Project ``CandidateMetrics`` into a flat dict for ``wandb.log``.

    Direct fields land at the top level. ``finish_reason_distribution``
    is flattened to ``finish_reason/<key>`` scalars. The composed
    ``observability`` field is delegated to
    :meth:`GenerationObservability.to_wandb_payload` with the
    ``vllm_gen`` prefix so production + benchmark agree on
    observability key namespacing.
    """
    payload = metrics.model_dump(exclude={"observability", "finish_reason_distribution"})
    fr_dist = metrics.finish_reason_distribution or {}
    payload.update({f"finish_reason/{k}": v for k, v in fr_dist.items()})
    payload.update(metrics.observability.to_wandb_payload())
    return payload


def log_and_finish(run: Any, metrics: CandidateMetrics | None, exit_code: int = 0) -> None:
    """Log metrics (if any) and close the wandb run. Swallows wandb errors.

    Called on both success (``metrics`` populated, ``exit_code=0``) and
    skip paths (``metrics=None``, ``exit_code=1``). No-op when ``run``
    is ``None`` so callers don't need to branch.
    """
    if run is None:
        return
    try:
        if metrics is not None:
            import wandb  # noqa: PLC0415

            wandb.log(_flatten_metrics(metrics))
        run.finish(exit_code=exit_code)
    except Exception as exc:  # noqa: BLE001 - degraded mode
        logger.warning("wandb finish failed; continuing", exc_info=exc)
