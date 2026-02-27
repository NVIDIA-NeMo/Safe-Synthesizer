# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""WandB integration for Safe Synthesizer.

This module provides WandB (Weights & Biases) integration for experiment tracking,
including run initialization, configuration logging, and failure reporting.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import wandb
from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings

from ..config import SafeSynthesizerParameters
from ..observability import get_logger
from .artifact_structure import Workdir

logger = get_logger(__name__)

PathT = str | Path


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
        default=WandbMode.DISABLED, validation_alias=AliasChoices("WANDB_MODE", "NSS_WANDB_MODE")
    )
    wandb_project: str | None = Field(default=None, validation_alias=AliasChoices("WANDB_PROJECT", "NSS_WANDB_PROJECT"))
    exp_name: str = Field(default="nss_experiments")  # fallback for wandb_project
    phase: WandbPhase = WandbPhase.UNKNOWN

    model_config = {"env_prefix": "NSS_", "env_file": ".env", "extra": "ignore"}

    @field_validator("wandb_mode", mode="before")
    @classmethod
    def validate_wandb_mode(cls, v: str | WandbMode | None) -> WandbMode:
        """Validate the wandb mode."""
        if v is None:
            return WandbMode.DISABLED
        if isinstance(v, WandbMode):
            return v
        return WandbMode(v)

    @field_validator("phase", mode="before")
    @classmethod
    def validate_phase(cls, v: str | WandbPhase | None) -> WandbPhase:
        """Validate the wandb phase."""
        if v is None:
            return WandbPhase.UNKNOWN
        if isinstance(v, WandbPhase):
            return v
        return WandbPhase(v)

    @property
    def effective_wandb_project(self) -> str:
        """The effective wandb project name."""
        return self.wandb_project or self.exp_name


def log_failure_to_wandb(error: Exception, phase: str) -> None:
    """Log failure to wandb before exiting.

    Args:
        error: The exception that caused the failure
        phase: The phase where failure occurred (e.g., "train", "generation", "end_to_end")
    """
    try:
        if wandb.run is not None:
            wandb.log(
                {
                    "eval/success": 0,
                    f"{phase}/error_type": type(error).__name__,
                    f"{phase}/error_message": str(error),
                }
            )
            logger.info(f"Logged failure to wandb for {phase} phase")
    except Exception as e:
        logger.warning(f"Failed to log error to wandb: {e}")


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
