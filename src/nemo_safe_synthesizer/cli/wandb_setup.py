# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WandB integration for Safe Synthesizer.

This module provides WandB (Weights & Biases) integration for experiment tracking,
including run initialization, configuration logging, and failure reporting.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings

import wandb

from ..config import SafeSynthesizerParameters, SafeSynthesizerSummary
from ..observability import get_logger
from .artifact_structure import Workdir

logger = get_logger(__name__)

_EVALUATION_PUBLISHING_FINGERPRINT_KEY = "_nss_evaluation_publishing_fingerprint"
_EVALUATION_SCORECARD_FINGERPRINT_KEY = "_nss_evaluation_scorecard_fingerprint"
_EVALUATION_REPORT_FINGERPRINT_KEY = "_nss_evaluation_report_fingerprint"
_EVALUATION_ARTIFACT_FINGERPRINT_KEY = "_nss_evaluation_artifact_fingerprint"


@dataclass(frozen=True)
class _PublishingOutcome:
    """Result of one idempotent W&B publishing operation."""

    published: bool
    recorded: bool


@dataclass(frozen=True)
class _EvaluationPublishingPayload:
    """Local evaluation data prepared for W&B publishing."""

    eval_metrics: dict[str, float | int | None]
    report_path: Path
    metrics_path: Path
    report_bytes: bytes | None
    metrics_bytes: bytes | None
    report_sha256: str | None
    metrics_sha256: str | None
    fingerprint: str


@dataclass(frozen=True)
class _PublishedReportState:
    """Durable state of the latest report known to exist in W&B."""

    uploaded: bool
    sha256: str | None


def resolve_wandb_run_id(id_or_path: str) -> str:
    """Resolve a wandb run ID from a string or file path.

    Args:
        id_or_path: Either a wandb run ID string, or a path to a file containing the ID.

    Returns:
        The resolved wandb run ID.
    """
    path = Path(id_or_path)
    if path.exists() and path.is_file():
        return path.read_text().strip()
    return id_or_path


class WandbMode(str, Enum):
    """WandB run mode."""

    ONLINE = "online"
    OFFLINE = "offline"
    DISABLED = "disabled"


class WandbPhase(str, Enum):
    """Phase of the Safe Synthesizer pipeline."""

    TRAIN = "train"
    GENERATE = "generate"
    END_TO_END = "end_to_end"
    UNKNOWN = "unknown"
    PROCESS_DATA = "process_data"
    EVALUATE = "evaluate"


class WandbSettings(BaseSettings):
    """WandB configuration for Safe Synthesizer.

    All settings can be configured via environment variables.
    """

    wandb_mode: WandbMode = Field(
        default=WandbMode.DISABLED,
        description="Run mode, one of online, offline, or disabled.",
        validation_alias=AliasChoices("WANDB_MODE", "NSS_WANDB_MODE"),
    )
    """Run mode, one of online, offline, or disabled (env variable: ``WANDB_MODE`` or ``NSS_WANDB_MODE``)."""

    wandb_project: str | None = Field(
        default=None,
        description="WandB project name override.",
        validation_alias=AliasChoices("WANDB_PROJECT", "NSS_WANDB_PROJECT"),
    )
    """WandB project name override (env variable: ``WANDB_PROJECT`` or ``NSS_WANDB_PROJECT``)."""

    exp_name: str = Field(
        default="nss_experiments", description="Fallback project name when ``wandb_project`` is not set."
    )
    """Fallback project name when ``wandb_project`` is not set."""

    phase: WandbPhase = Field(default=WandbPhase.UNKNOWN, description="Current pipeline phase for WandB grouping.")
    """Current pipeline phase for WandB grouping."""

    model_config = {"env_prefix": "NSS_", "env_file": ".env", "extra": "ignore"}

    @field_validator("wandb_mode", mode="before")
    @classmethod
    def validate_wandb_mode(cls, v: str | WandbMode | None) -> WandbMode:
        """Coerce string or None to ``WandbMode`` enum, defaulting to DISABLED."""
        if v is None:
            return WandbMode.DISABLED
        if isinstance(v, WandbMode):
            return v
        return WandbMode(v)

    @field_validator("phase", mode="before")
    @classmethod
    def validate_phase(cls, v: str | WandbPhase | None) -> WandbPhase:
        """Coerce string or None to ``WandbPhase``, defaulting to UNKNOWN."""
        if v is None:
            return WandbPhase.UNKNOWN
        if isinstance(v, WandbPhase):
            return v
        return WandbPhase(v)

    @property
    def effective_wandb_project(self) -> str:
        """Effective wandb project name, falling back to ``exp_name``."""
        return self.wandb_project or self.exp_name


def log_failure_to_wandb(error: Exception, phase: str) -> None:
    """Log failure to wandb before exiting.

    Args:
        error: The exception that caused the failure
        phase: The phase where failure occurred (e.g., "train", "generation", "end_to_end")
    """
    try:
        if wandb.run is not None:
            wandb.run.summary.update(
                {
                    "eval/success": 0,
                    f"{phase}/error_type": type(error).__name__,
                    f"{phase}/error_message": str(error),
                }
            )
            logger.runtime.info("Updated wandb failure summary for %s phase", phase)
    except Exception as exc:  # noqa: BLE001 -- observability is best-effort
        logger.runtime.warning("Failed to log error to wandb: %s", exc)


def publish_evaluation_report(
    workdir: Workdir,
    summary: SafeSynthesizerSummary,
    upload_report: bool,
) -> None:
    """Best-effort publish final evaluation media for a CLI-managed run.

    The scorecard is always sent when W&B is active. HTML and files leave the
    local machine only when ``upload_report`` is explicitly enabled.

    Args:
        workdir: Run artifact paths containing the saved report and metrics.
        summary: Final pipeline summary used to construct the scorecard.
        upload_report: Whether report HTML and artifact egress is permitted.
    """
    run = wandb.run
    if run is None:
        return
    payload = _prepare_evaluation_publishing_payload(workdir, summary, upload_report)

    if _publishing_marker_matches(run.summary, _EVALUATION_PUBLISHING_FINGERPRINT_KEY, payload.fingerprint):
        return

    previous_report_state = _read_published_report_state(run.summary)
    scorecard_outcome = _publish_evaluation_scorecard(run, payload.eval_metrics)
    report_outcome = _publish_evaluation_report_panel(run, payload)
    artifact_outcome = _publish_evaluation_artifact(run, payload)
    report_state = _updated_report_state(previous_report_state, report_outcome, payload.report_sha256)
    publishing_recorded = all(outcome.recorded for outcome in (scorecard_outcome, report_outcome, artifact_outcome))
    _update_evaluation_publishing_summary(
        run.summary,
        report_state=report_state,
        publishing_fingerprint=payload.fingerprint if publishing_recorded else None,
    )


def _prepare_evaluation_publishing_payload(
    workdir: Workdir,
    summary: SafeSynthesizerSummary,
    upload_report: bool,
) -> _EvaluationPublishingPayload:
    """Read and fingerprint the local evaluation data selected for publishing."""
    eval_metrics = {key: value for key, value in summary._wandb_metrics().items() if key.startswith("eval/")}
    report_path = workdir.evaluation_report
    metrics_path = workdir.evaluation_metrics
    report_bytes = _read_optional_file(report_path, "report") if upload_report else None
    metrics_bytes = _read_optional_file(metrics_path, "metrics") if upload_report else None
    report_sha256 = _optional_sha256(report_bytes)
    metrics_sha256 = _optional_sha256(metrics_bytes)
    fingerprint = _fingerprint_payload(
        {
            "eval_metrics": eval_metrics,
            "upload_report": upload_report,
            "report_sha256": report_sha256,
            "metrics_sha256": metrics_sha256,
        }
    )
    return _EvaluationPublishingPayload(
        eval_metrics=eval_metrics,
        report_path=report_path,
        metrics_path=metrics_path,
        report_bytes=report_bytes,
        metrics_bytes=metrics_bytes,
        report_sha256=report_sha256,
        metrics_sha256=metrics_sha256,
        fingerprint=fingerprint,
    )


def _publish_evaluation_scorecard(
    run: Any,
    eval_metrics: dict[str, float | int | None],
) -> _PublishingOutcome:
    """Publish the scorecard once for the current evaluation metrics."""
    fingerprint = _fingerprint_payload({"eval_metrics": eval_metrics})
    return _publish_once(
        run.summary,
        _EVALUATION_SCORECARD_FINGERPRINT_KEY,
        fingerprint,
        "scorecard",
        lambda: _log_evaluation_scorecard(run, eval_metrics),
    )


def _publish_evaluation_report_panel(run: Any, payload: _EvaluationPublishingPayload) -> _PublishingOutcome:
    """Publish the HTML report panel when report bytes are available."""
    report_bytes = payload.report_bytes
    if report_bytes is None or payload.report_sha256 is None:
        return _PublishingOutcome(published=False, recorded=True)
    fingerprint = _fingerprint_payload({"report_sha256": payload.report_sha256})
    return _publish_once(
        run.summary,
        _EVALUATION_REPORT_FINGERPRINT_KEY,
        fingerprint,
        "report",
        lambda: _log_evaluation_report(run, report_bytes),
    )


def _publish_evaluation_artifact(run: Any, payload: _EvaluationPublishingPayload) -> _PublishingOutcome:
    """Publish an artifact when at least one evaluation file is available."""
    if payload.report_bytes is None and payload.metrics_bytes is None:
        return _PublishingOutcome(published=False, recorded=True)
    fingerprint = _fingerprint_payload(
        {
            "report_sha256": payload.report_sha256,
            "metrics_sha256": payload.metrics_sha256,
        }
    )
    return _publish_once(
        run.summary,
        _EVALUATION_ARTIFACT_FINGERPRINT_KEY,
        fingerprint,
        "report artifact",
        lambda: _log_evaluation_artifact(run, payload),
    )


def _publish_once(
    summary: Any,
    marker_key: str,
    fingerprint: str,
    description: str,
    publish: Callable[[], None],
) -> _PublishingOutcome:
    """Publish one W&B object unless its durable marker already matches."""
    if _publishing_marker_matches(summary, marker_key, fingerprint):
        return _PublishingOutcome(published=True, recorded=True)
    try:
        publish()
    except Exception as exc:  # noqa: BLE001 -- W&B publishing is optional
        logger.runtime.warning("Failed to publish W&B evaluation %s: %s", description, exc)
        return _PublishingOutcome(published=False, recorded=False)
    return _PublishingOutcome(
        published=True,
        recorded=_record_publishing_marker(summary, marker_key, fingerprint),
    )


def _log_evaluation_scorecard(run: Any, eval_metrics: dict[str, float | int | None]) -> None:
    """Log the final evaluation metrics as a W&B table panel."""
    run.log(
        {
            "evaluation/scorecard": wandb.Table(
                columns=["metric", "value"],
                data=[[key, value] for key, value in eval_metrics.items()],
            )
        }
    )


def _log_evaluation_report(run: Any, report_bytes: bytes) -> None:
    """Log the saved evaluation report as a W&B HTML panel."""
    run.log({"evaluation/report": wandb.Html(report_bytes.decode("utf-8"), inject=False)})


def _log_evaluation_artifact(run: Any, payload: _EvaluationPublishingPayload) -> None:
    """Log the available evaluation files as a W&B artifact."""
    artifact = wandb.Artifact(f"safe-synthesizer-evaluation-report-{run.id}", type="evaluation-report")
    if payload.report_bytes is not None:
        artifact.add_file(str(payload.report_path), name="evaluation_report.html")
    if payload.metrics_bytes is not None:
        artifact.add_file(str(payload.metrics_path), name="evaluation_metrics.json")
    run.log_artifact(artifact)


def _update_evaluation_publishing_summary(
    summary: Any,
    *,
    report_state: _PublishedReportState,
    publishing_fingerprint: str | None,
) -> None:
    """Record durable evaluation publishing state without failing the pipeline."""
    try:
        publishing_summary = {
            "evaluation/report_uploaded_post_run": report_state.uploaded,
            "evaluation/report_sha256": report_state.sha256,
        }
        if publishing_fingerprint is not None:
            publishing_summary[_EVALUATION_PUBLISHING_FINGERPRINT_KEY] = publishing_fingerprint
        summary.update(publishing_summary)
    except Exception as exc:  # noqa: BLE001 -- W&B publishing is optional
        logger.runtime.warning("Failed to update W&B evaluation publishing summary: %s", exc)


def _read_published_report_state(summary: Any) -> _PublishedReportState:
    """Read the report state already persisted in the W&B summary."""
    uploaded = _read_publishing_summary_value(summary, "evaluation/report_uploaded_post_run", False)
    sha256 = _read_publishing_summary_value(summary, "evaluation/report_sha256", None)
    return _PublishedReportState(
        uploaded=uploaded is True,
        sha256=sha256 if isinstance(sha256, str) else None,
    )


def _updated_report_state(
    previous: _PublishedReportState,
    outcome: _PublishingOutcome,
    report_sha256: str | None,
) -> _PublishedReportState:
    """Return the latest report state known to have reached W&B."""
    if outcome.published and report_sha256 is not None:
        return _PublishedReportState(uploaded=True, sha256=report_sha256)
    return previous


def _read_publishing_summary_value(summary: Any, key: str, default: object) -> object:
    """Read a durable W&B publishing value without failing the pipeline."""
    try:
        value = summary.get(key)
    except Exception as exc:  # noqa: BLE001 -- publishing remains best-effort
        logger.runtime.warning("Failed to read W&B evaluation publishing value %s: %s", key, exc)
        return default
    return default if value is None else value


def _read_optional_file(path: Path, file_description: str) -> bytes | None:
    """Read an opted-in evaluation file, warning instead of failing the run."""
    try:
        if not path.is_file():
            logger.runtime.warning(
                "W&B evaluation %s upload requested but file is missing: %s",
                file_description,
                path,
            )
            return None
        return path.read_bytes()
    except Exception as exc:  # noqa: BLE001 -- local output inspection is optional
        logger.runtime.warning("Failed to read W&B evaluation %s: %s", file_description, exc)
        return None


def _optional_sha256(content: bytes | None) -> str | None:
    """Hash optional evaluation content."""
    return hashlib.sha256(content).hexdigest() if content is not None else None


def _fingerprint_payload(payload: dict[str, object]) -> str:
    """Return a stable SHA-256 marker for a JSON-serializable payload."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _publishing_marker_matches(summary: Any, key: str, fingerprint: str) -> bool:
    """Return whether a best-effort W&B publishing marker matches."""
    try:
        return summary.get(key) == fingerprint
    except Exception as exc:  # noqa: BLE001 -- publishing remains best-effort
        logger.runtime.warning("Failed to read W&B evaluation publishing marker %s: %s", key, exc)
        return False


def _record_publishing_marker(summary: Any, key: str, fingerprint: str) -> bool:
    """Persist a best-effort W&B publishing marker after an operation succeeds."""
    try:
        summary.update({key: fingerprint})
        return True
    except Exception as exc:  # noqa: BLE001 -- publishing remains best-effort
        logger.runtime.warning("Failed to update W&B evaluation publishing marker %s: %s", key, exc)
        return False


def update_wandb_config(
    cfg: SafeSynthesizerParameters | None = None,
    additional_configs: dict[str, Any] | None = None,
) -> None:
    """Update the wandb config with the given configuration.

    Args:
        cfg: SafeSynthesizerParameters to log
        additional_configs: Additional key-value pairs to log
    """
    if wandb.run is None:
        return

    if additional_configs is None:
        additional_configs = {}

    if cfg is not None:
        config_dict = cfg.model_dump()
        config_dict.update(additional_configs)
        wandb.config.update(config_dict, allow_val_change=True)


def initialize_wandb_run(
    workdir: Workdir,
    resume_job_id: str | None = None,
    cfg: SafeSynthesizerParameters | None = None,
) -> None:
    """Initialize or resume a wandb run with consistent configuration.

    This function handles four cases (in priority order):
    1. WandB already initialized - just save the run ID
    2. Explicit resume_job_id provided - resume that run (ID or file path)
    3. Resume existing run from saved run_id file in workdir
    4. Create new run

    Args:
        workdir: Workdir structure containing paths for run ID files
        resume_job_id: Optional wandb run ID or path to file containing the ID
        cfg: Optional SafeSynthesizerParameters to log to wandb config
    """
    settings = WandbSettings()

    logger.info(f"WANDB_MODE: {settings.wandb_mode}")
    if settings.wandb_mode == WandbMode.DISABLED:
        return

    wandb_project = settings.effective_wandb_project
    logger.info(f"WANDB_PROJECT: {wandb_project}")

    phase = settings.phase
    run_id_file = workdir.wandb_run_id_file

    if TYPE_CHECKING:
        assert isinstance(run_id_file, Path)

    # WandB settings to prevent console log issues
    wandb_settings = wandb.Settings(
        console="wrap",  # Wrap console output instead of redirecting
    )

    # Make a dictionary of additional configs to log to wandb
    additional_configs = {
        "dataset_name": workdir.dataset_name,
        "config_name": workdir.config_name,
        "dataset_name-config_name": f"{workdir.dataset_name}-{workdir.config_name}",  # wandb charts can only group by one variable
        "run_name": workdir.run_name,
        "phase": phase,
    }

    # Case 1: WandB already initialized
    if wandb.run is not None:
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        run_id_file.write_text(wandb.run.id, encoding="utf-8")

    # Case 2: Explicit resume_job_id provided (ID or file path)
    elif resume_job_id is not None:
        resolved_run_id = resolve_wandb_run_id(resume_job_id)
        logger.info(f"Resuming wandb run: {resolved_run_id} (from --wandb-resume-job-id)")
        wandb.init(
            project=wandb_project,
            id=resolved_run_id,
            resume="allow",
            mode=settings.wandb_mode.value,
            settings=wandb_settings,
            dir=workdir.run_dir,
        )
        if wandb.run is not None:
            run_id_file.parent.mkdir(parents=True, exist_ok=True)
            run_id_file.write_text(wandb.run.id, encoding="utf-8")

    # Case 3: Resume existing run from saved run_id file in workdir
    elif run_id_file.exists():
        saved_run_id = run_id_file.read_text().strip()
        logger.info(f"Resuming wandb run: {saved_run_id} (from {run_id_file.name})")
        wandb.init(
            project=wandb_project,
            id=saved_run_id,
            resume="allow",
            mode=settings.wandb_mode.value,
            settings=wandb_settings,
            dir=workdir.run_dir,
        )
        if wandb.run is not None:
            run_id_file.write_text(wandb.run.id, encoding="utf-8")

    # Case 4: Create new run
    else:
        logger.info(f"Creating new wandb run: {workdir.run_name}")
        run_id_file.parent.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=wandb_project,
            name=workdir.run_name,
            mode=settings.wandb_mode.value,
            settings=wandb_settings,
            dir=workdir.run_dir,
        )
        if wandb.run is not None:
            run_id_file.write_text(wandb.run.id, encoding="utf-8")
        logger.info(f"Saved wandb run ID to {workdir.wandb_run_id_file}")

        # Log config to wandb (only for new runs - resumed runs already have config)
        update_wandb_config(cfg, additional_configs=additional_configs)

    # Log run info
    logger.info(f"Wandb run name: {wandb.run.name if wandb.run else 'None'}")
    logger.info(f"Wandb run id: {wandb.run.id if wandb.run else 'None'}")
    if settings.wandb_mode != WandbMode.DISABLED:
        logger.info(f"Wandb run url: {wandb.run.url if wandb.run else 'None'}")


class WandbLoggable(Protocol):
    """Structural type for observability events that can be logged to wandb.

    Any event exposing ``to_wandb_payload(prefix) -> dict`` satisfies this --
    e.g. ``TrainingObservability`` and the generation-side
    ``GenerationObservability``. Using a Protocol keeps :func:`log_observability_event`
    decoupled from the concrete event types (no import of the training/generation
    subpackages from this CLI module).
    """

    def to_wandb_payload(self, prefix: str = "") -> dict[str, Any]:
        """Return wandb metrics for this event, namespaced under ``prefix``."""


def log_observability_event(event: WandbLoggable, prefix: str) -> None:
    """Log an observability event to the currently active wandb run.

    Generic sink shared by final training and generation observability paths.
    No-op when no wandb run is active (``WANDB_MODE=disabled`` or the pipeline
    hasn't called :func:`initialize_wandb_run`). Errors during summary updates are
    swallowed at warning level -- observability is best-effort and a wandb
    failure must not break the run.

    Args:
        event: Any object exposing ``to_wandb_payload(prefix) -> dict`` (see
            :class:`WandbLoggable`).
        prefix: wandb key namespace for this event's metrics (e.g. ``"training"``
            or ``"vllm_gen"``).
    """
    if wandb.run is None:
        return
    try:
        wandb.run.summary.update(event.to_wandb_payload(prefix=prefix))
    except Exception as exc:  # noqa: BLE001 -- degraded mode
        logger.warning(f"failed to log observability event ({prefix!r}) to wandb: {exc}")
