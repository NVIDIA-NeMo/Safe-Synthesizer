# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI utility functions for Safe Synthesizer.

This module provides utility functions for CLI commands including:

- Logging initialization
- Dataset loading
- Configuration merging
- Result saving
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import click
import pandas as pd
from pydantic import ValidationError

from ..config import SafeSynthesizerParameters
from ..defaults import DEFAULT_ARTIFACTS_PATH
from ..observability import configure_logging_from_workdir, get_logger, initialize_observability
from ..utils import merge_dicts
from .artifact_structure import Workdir
from .datasets import DatasetRegistry
from .wandb_setup import WandbMode, initialize_wandb_run

if TYPE_CHECKING:
    from ..observability import CategoryLogger
    from .settings import CLISettings

logger = get_logger(__name__)

PathT = str | Path
CLI_NESTED_FIELD_SEPARATOR = "__"
"""Separator used to denote nested fields in CLI options. e.g., --data__holdout=0.1"""

VERBOSITY_TO_LOG_LEVEL: dict[int, Literal["INFO", "DEBUG", "DEBUG_DEPENDENCIES"]] = {
    0: "INFO",
    1: "DEBUG",
    2: "DEBUG_DEPENDENCIES",
}
"""Mapping from CLI verbosity level to log level."""


def _create_workdir(
    artifacts_path: PathT | None,
    run_path: PathT | None,
    data_source: str | None,
    config_path: PathT | None,
    resume: bool = False,
    phase: str | None = None,
    auto_discover_adapter: bool = False,
) -> Workdir:
    """Create Workdir from CLI arguments.

    This is called first in common_setup() to establish the artifact directory
    structure before initializing logging or other services.

    Args:
        artifacts_path: Base directory for all runs. Runs are created as
            <artifacts-path>/<config>---<dataset>/<timestamp>/
        run_path: Explicit path for this run's output directory.
            When specified, outputs go directly to this path.
            Overrides artifacts_path.
        data_source: Dataset name, URL, or path (used to extract dataset_name). Can be
            None for resume mode (generate) where the dataset is loaded from cached files.
        config_path: Path to config file (used to extract config_name)
        resume: If True, attempt to load an existing workdir from run_path.
            Raises an error if no existing workdir is found. Also starts a new
            generation run to create unique output files.
        phase: The current phase (train, generate, end_to_end). Defaults to NSS_PHASE env var.
        auto_discover_adapter: If True and resume=True, automatically find the latest
            trained adapter in artifacts_path. Without this flag, run_path must be
            explicitly specified for generation.

    Returns:
        Configured Workdir

    Raises:
        click.ClickException: If resume=True but no existing workdir is found
        click.ClickException: If run_path already contains a training run
        click.ClickException: If data_source is None and not in resume mode
    """
    if data_source is None and not resume:
        raise click.ClickException("--data-source is required for new runs")

    dataset_name = Path(data_source).stem if data_source else "unknown"
    config_name = Path(config_path).stem if config_path else "default"
    current_phase = phase or os.getenv("NSS_PHASE", "unknown")

    # Handle conflicting options: run_path takes precedence
    if artifacts_path is not None and run_path is not None:
        logger.warning(f"--artifacts-path is ignored when --run-path is specified. Using: {run_path}")

    if resume:
        # For resume mode (generation), we need an existing workdir with a trained adapter
        if run_path is not None:
            # Explicit run path provided - use it directly without inferring structure
            path = Path(run_path).resolve()
            if not path.is_dir():
                raise click.ClickException(f"--run-path does not exist: {run_path}")

            # Create workdir with explicit path (don't infer structure from path)
            # This preserves the exact path without trying to parse config---dataset structure
            parent_workdir = Workdir(
                base_path=path.parent,
                config_name=config_name,
                dataset_name=dataset_name,
                run_name=path.name,
                _current_phase="generate",
                _explicit_run_path=path,
            )
            logger.info(f"Using explicit run path for resume: {path}")

        elif auto_discover_adapter:
            # Auto-discovery enabled - search in artifacts_path or default
            search_path = Path(artifacts_path) if artifacts_path else Path(DEFAULT_ARTIFACTS_PATH)
            if not search_path.is_dir():
                raise click.ClickException(
                    f"Cannot auto-discover adapter: directory does not exist: {search_path}\n\n"
                    "Either create the directory and run training first, or specify --run-path."
                )
            logger.info(f"Auto-discovering latest trained adapter in: {search_path}")

            try:
                parent_workdir = Workdir.from_path(path=search_path)
                logger.info(f"Found trained adapter at: {parent_workdir.adapter_path}")
            except ValueError as e:
                raise click.ClickException(
                    f"No trained adapter found in {search_path}.\n\n"
                    "Run training first:\n"
                    f"  safe-synthesizer run train --data-source data.csv --artifact-path {search_path}"
                ) from e
        else:
            # No run_path and no auto-discover - error with helpful message
            raise click.ClickException(
                "--run-path is required for 'generate' command.\n\n"
                "Specify the path to a trained run:\n"
                "  safe-synthesizer run generate --data-source data.csv --run-path ./artifacts/<project>/<run>\n\n"
                "Or use --auto-discover-adapter to find the latest trained run:\n"
                "  safe-synthesizer run generate --data-source data.csv --auto-discover-adapter"
            )

        # Verify adapter exists using the workdir's adapter_path property
        adapter_path = parent_workdir.adapter_path
        if not adapter_path.exists() or not list(adapter_path.glob("*.safetensors")):
            raise click.ClickException(
                f"No trained adapter found in {adapter_path}.\n"
                "Run training first or specify a path with an existing trained adapter."
            )

        # When --run-path is explicitly provided, use that path directly for generation
        # (don't create a new timestamped run). Otherwise, create a new generation run.
        if run_path is not None:
            return parent_workdir
        workdir = parent_workdir.new_generation_run()
        return workdir

    # For new runs (train or end_to_end)
    if run_path is not None:
        # Use explicit run path
        try:
            return Workdir.from_explicit_run_path(
                run_path=Path(run_path),
                config_name=config_name,
                dataset_name=dataset_name,
                current_phase=current_phase,
            )
        except ValueError as e:
            raise click.ClickException(str(e)) from e

    # Use artifacts_path (or default) with auto-generated run name
    base_path = Path(artifacts_path) if artifacts_path else Path(DEFAULT_ARTIFACTS_PATH)
    return Workdir(
        base_path=base_path,
        dataset_name=dataset_name,
        config_name=config_name,
        _current_phase=current_phase,
    )


def common_setup(
    settings: "CLISettings",
    resume: bool = False,
    phase: str | None = None,
    auto_discover_adapter: bool = False,
    wandb_resume_job_id: str | None = None,
) -> tuple["CategoryLogger", SafeSynthesizerParameters, pd.DataFrame | None, Workdir]:
    """Common setup for all run commands using unified CLISettings.

    The setup order is:
    1. Create Workdir (establishes artifact paths)
    2. Initialize logging (using workdir.log_file)
    3. Create DatasetRegistry from settings.dataset_registry if present, otherwise create an empty registry
    4. Load dataset from registry if settings.data_source is a known name, otherwise from data_source
    5. Load config with overrides from dataset overrides and command line overrides
    6. Initialize wandb

    Args:
        settings: Unified CLI settings (includes all config from env vars and CLI args)
        resume: If True, attempt to resume from an existing workdir
        phase: The current phase (train, generate, end_to_end)
        auto_discover_adapter: If True and resume=True, auto-discover the latest trained adapter
        wandb_resume_job_id: Optional wandb run ID or path to file containing the ID to resume

    Returns:
        Tuple of (logger, config, dataframe, workdir). For generate-only runs with
        cached datasets, dataframe may be None (loaded from cached files by SafeSynthesizer).
    """
    # 1. Create workdir FIRST - this establishes all artifact paths
    workdir = _create_workdir(
        settings.artifact_path,
        settings.run_path,
        settings.data_source,
        settings.config_path,
        resume=resume,
        phase=phase,
        auto_discover_adapter=auto_discover_adapter,
    )

    # Ensure directories exist
    workdir.ensure_directories()

    # 2. Initialize logging using the workdir structure and settings
    run_logger = _initialize_logging_for_cli_from_settings(
        settings=settings,
        workdir=workdir,
    )

    # 3. Create DatasetRegistry
    if settings.dataset_registry:
        dataset_registry = DatasetRegistry.from_yaml(settings.dataset_registry)
    else:
        dataset_registry = DatasetRegistry()

    # 4. Load dataset (or check for cached dataset in resume mode)

    # synthesis_overrides collects config overrides from dataset registry and
    # CLI, which is then combined with the config file when calling
    # merge_overrides(). CLI takes top precedence, then dataset registry, and
    # finally the config file. See test_utils.py and especially
    # test_overrides_config_registry_and_cli for examples of how the resolution
    # is expected to work.
    synthesis_overrides: dict[str, Any] | None = dict()
    df: pd.DataFrame | None = None
    if settings.data_source:
        dataset_info = dataset_registry.get_dataset(settings.data_source)
        synthesis_overrides = merge_dicts(synthesis_overrides, dataset_info.overrides or dict())
        df = dataset_info.fetch()
    elif resume:
        # For generate-only runs without --data-source, verify cached dataset exists
        cached_training: Path = workdir.source_dataset.training  # type: ignore[assignment]
        cached_test: Path = workdir.source_dataset.test  # type: ignore[assignment]
        if not cached_training.exists() or not cached_test.exists():
            raise click.ClickException(
                f"No cached dataset found in workdir: {workdir.source_dataset.path}\n\n"
                "Either provide --data-source to load a dataset, or ensure the workdir "
                "contains cached training/test data from a previous run."
            )
        run_logger.info(f"Using cached dataset from: {workdir.source_dataset.path}")
        # df is None - SafeSynthesizer.load_from_save_path() will load from cached files
    else:
        # Should not happen - _create_workdir already validates this
        raise click.ClickException("--data-source is required for new runs")

    # 5. Load config with overrides from settings
    synthesis_overrides = merge_dicts(synthesis_overrides, settings.synthesis_overrides)
    config = merge_overrides(settings.config_path, synthesis_overrides)

    # 6. Initialize wandb (uses workdir for run ID files)
    initialize_wandb_run(workdir, resume_job_id=wandb_resume_job_id, cfg=config)

    return run_logger, config, df, workdir


def _set_wandb_env_vars(
    wandb_mode: WandbMode | str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    """Set environment variables related to wandb. Called before initializing wandb."""
    if wandb_mode:
        os.environ["WANDB_MODE"] = wandb_mode if isinstance(wandb_mode, str) else wandb_mode.value
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = wandb_run_name


def _initialize_logging_for_cli_from_settings(
    settings: "CLISettings",
    workdir: Workdir,
) -> "CategoryLogger":
    """Initialize logging using CLISettings.

    Uses the unified settings object to configure logging, with the workdir
    providing the log file path.

    Args:
        settings: Unified CLI settings
        workdir: Workdir for artifact paths (logs go to workdir.log_file)

    Returns:
        The configured logger
    """
    import structlog

    # Disable vLLM's default logging configuration so it uses our handlers
    # This MUST be set before vLLM is imported anywhere in the application
    # See: https://docs.vllm.ai/en/latest/examples/others/logging_configuration/
    os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")

    # Set wandb environment variables from settings
    _set_wandb_env_vars(
        settings.effective_wandb_mode,
        settings.effective_wandb_project,
    )

    # Determine log level from verbosity
    verbose = min(settings.verbose, 2)
    log_level = VERBOSITY_TO_LOG_LEVEL[verbose]

    # Use workdir structure for logging
    workdir.ensure_directories()
    log_file_path = configure_logging_from_workdir(
        workdir,
        log_level=log_level,
        log_format=settings.effective_log_format,
        log_color=settings.effective_log_color,
    )

    # Initialize the logging system
    structlog.reset_defaults()
    initialize_observability()

    run_logger = get_logger("nemo_safe_synthesizer")

    run_logger.system.info(f"Project directory: {workdir.project_dir}")
    run_logger.system.info(f"Run directory: {workdir.run_dir}")
    run_logger.system.info(f"Phase directory: {workdir.phase_dir()}")
    run_logger.system.info(f"Log file: {log_file_path}")

    return run_logger


def merge_overrides(config_path: str | Path | None, overrides: dict) -> SafeSynthesizerParameters:
    """Merge overrides into a SafeSynthesizerParameters object.

    If config_path is None, use the overrides to create a new SafeSynthesizerParameters object.
    Otherwise, merge the overrides into the config file.

    Args:
        config_path: Path to config file (YAML)
        overrides: Dictionary of override values

    Returns:
        Merged SafeSynthesizerParameters
    """
    try:
        if config_path is None:
            my_config = SafeSynthesizerParameters.model_validate(overrides)
        else:
            params = merge_dicts(
                SafeSynthesizerParameters.from_yaml(config_path).model_dump(exclude_unset=False), overrides
            )
            my_config = SafeSynthesizerParameters.model_validate(params)
    except ValidationError as e:
        click.echo(f"{config_path} is invalid:\n{e}")
        sys.exit(1)
    return my_config
