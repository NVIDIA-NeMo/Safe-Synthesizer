# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI run commands for Safe Synthesizer.

This module provides the run command group for the Safe Synthesizer pipeline:

- ``run`` (default): Full end-to-end pipeline
- ``run train``: Training stage only
- ``run generate``: Generation stage only (requires trained model)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import click

from ..config import SafeSynthesizerParameters
from ..configurator.pydantic_click_options import (
    parse_overrides,
    pydantic_options,
)
from ..observability import traced_user
from .settings import CLISettings
from .utils import (
    CLI_NESTED_FIELD_SEPARATOR,
    PathT,
    common_setup,
)

if TYPE_CHECKING:
    from ..sdk.library_builder import SafeSynthesizer


def common_run_options(f):
    """Decorator to add common options for run commands.

    Note: Environment variable handling is done by CLISettings, not Click's envvar=.
    This keeps env var logic in one place (pydantic-settings) rather than duplicating it.
    """
    options = []
    options.append(
        click.option("--config", "config_path", default=None, required=False, help="path to a yaml config file")
    )
    options.append(
        click.option(
            "--data-source",
            type=str,
            default=None,
            required=False,
            help="Dataset name, URL, or path to CSV dataset. "
            "For 'run generate', this is optional if a cached dataset exists in the workdir.",
        )
    )
    options.append(
        click.option(
            "--artifact-path",
            type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
            default=None,
            required=False,
            help="Base directory for all runs. Runs are created as "
            "<artifact-path>/<config>---<dataset>/<timestamp>/. "
            "Can also be set via NSS_ARTIFACTS_PATH env var. "
            "[default: ./safe-synthesizer-artifacts]",
        )
    )
    options.append(
        click.option(
            "--run-path",
            type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
            default=None,
            required=False,
            help="Explicit path for this run's output directory. "
            "When specified, outputs go directly to this path. "
            "Overrides --artifact-path.",
        )
    )
    options.append(
        click.option(
            "--output-file",
            type=click.Path(exists=False),
            default=None,
            required=False,
            help="Path to output CSV file. Overrides the default workdir output location.",
        )
    )
    options.append(
        click.option(
            "--log-format",
            type=click.Choice(["json", "plain"]),
            default=None,
            required=False,
            help="Log format for console output. File logging will always be JSON. "
            "Can also be set via NSS_LOG_FORMAT env var. [default: plain]",
        )
    )
    options.append(
        click.option(
            "--log-color/--no-log-color",
            type=click.BOOL,
            default=None,
            required=False,
            help="Whether to colorize the log output on the console. [default: --log-color]",
        )
    )
    options.append(
        click.option(
            "--log-file",
            type=click.Path(exists=False),
            default=None,
            required=False,
            help="Path to log file. Defaults to a file nested under the run directory. "
            "Can also be set via NSS_LOG_FILE env var.",
        )
    )
    options.append(
        click.option(
            "--wandb-mode",
            type=click.Choice(["online", "offline", "disabled"]),
            default=None,
            required=False,
            help="Wandb mode. 'online' will upload logs to wandb, 'offline' will save logs to a local file, 'disabled' will not upload logs to wandb. Can also be set via WANDB_MODE env var. [default: disabled]",
        )
    )
    options.append(
        click.option(
            "--wandb-project",
            type=str,
            default=None,
            required=False,
            help="Wandb project. Can also be set via WANDB_PROJECT env var.",
        )
    )
    options.append(
        click.option(
            "-v",
            "verbose",
            required=False,
            help="Verbose logging. 'v' shows debug info from main program, 'vv' shows debug from dependencies too",
            count=True,
        )
    )
    options.append(
        click.option(
            "--dataset-registry",
            type=str,
            required=False,
            default=None,
            help="URL or path of a dataset registry YAML file. If provided, "
            "datasets in the registry may be referenced by name in --data-source. "
            "Can also be set via NSS_DATASET_REGISTRY env var. "
            "If both env var and CLI option are provided, the CLI option takes precedence.",
        )
    )
    # Apply each option decorator in reverse order (decorators apply bottom-up)
    for option in reversed(options):
        f = option(f)
    return f


@click.group(invoke_without_command=True)
@click.pass_context
@common_run_options
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def run(
    ctx: click.Context,
    config_path: PathT | None,
    data_source: str,
    artifact_path: PathT | None,
    run_path: PathT | None,
    output_file: PathT | None,
    log_file: PathT | None,
    log_color: bool | None,
    log_format: str | None,
    verbose: int = 0,
    wandb_mode: str | None = None,
    wandb_project: str | None = None,
    dataset_registry: str | None = None,
    **kwargs,
):
    """Run the Safe Synthesizer end-to-end pipeline.

    Without a subcommand, runs the full end-to-end pipeline.
    Use 'run train' or 'run generate' for individual stages.
    """
    # If a subcommand is invoked, skip the default behavior
    if ctx.invoked_subcommand is not None:
        return

    # Create unified settings from CLI kwargs (CLI values override env vars)
    settings = CLISettings.from_cli_kwargs(
        data_source=data_source,
        config_path=config_path,
        artifact_path=artifact_path,
        run_path=run_path,
        output_file=output_file,
        log_file=log_file,
        log_color=log_color,
        log_format=log_format,
        verbose=verbose,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        synthesis_overrides=parse_overrides(kwargs),
        dataset_registry=dataset_registry,
    )

    # Full pipeline execution
    os.environ["NSS_PHASE"] = "end_to_end"
    run_logger, config, df, workdir = common_setup(
        settings=settings,
        phase="end_to_end",
    )
    run_logger.warning("Nemo Safe Synthesizer starting")
    run_logger.debug("running with: ", extra={"config": config.model_dump()})

    with traced_user("SafeSynthesizer"):
        from ..sdk.library_builder import SafeSynthesizer

        ss: SafeSynthesizer = SafeSynthesizer(config=config, workdir=workdir).with_data_source(df)
        # ss.run() calls train + generate + evaluate + save_results. The generate step has its own
        # try/finally, but train or evaluate failures leave the generator loaded; this guard
        # ensures teardown on all exit paths of the full pipeline.
        try:
            ss.run(output_file=settings.output_file)
            ss.results.summary.log_summary(run_logger)
            ss.results.summary.timing.log_timing(run_logger)
            ss.results.summary.log_wandb()
        finally:
            if hasattr(ss, "generator") and ss.generator is not None:
                ss.generator.teardown()


@run.command("train")
@common_run_options
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def run_train(
    config_path: PathT,
    data_source: str,
    artifact_path: PathT | None,
    run_path: PathT | None,
    output_file: PathT | None,
    log_format: str | None,
    log_color: bool | None,
    log_file: PathT | None,
    verbose: int,
    wandb_mode: str | None = None,
    wandb_project: str | None = None,
    dataset_registry: str | None = None,
    **kwargs,
):
    """Run the training stage only.

    This command processes data and trains the model, saving the adapter to the run directory.
    Use 'run generate' afterwards to generate synthetic data from the trained adapter.
    """
    # Create unified settings from CLI kwargs
    settings = CLISettings.from_cli_kwargs(
        data_source=data_source,
        config_path=config_path,
        artifact_path=artifact_path,
        run_path=run_path,
        output_file=output_file,
        log_file=log_file,
        log_color=log_color,
        log_format=log_format,
        verbose=verbose,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        synthesis_overrides=parse_overrides(kwargs),
        dataset_registry=dataset_registry,
    )

    os.environ["NSS_PHASE"] = "train"
    run_logger, config, df, workdir = common_setup(
        settings=settings,
        phase="train",
    )
    from ..sdk.library_builder import SafeSynthesizer

    with traced_user("SafeSynthesizer"):
        SafeSynthesizer(config, workdir=workdir).with_data_source(df).process_data().train()
        run_logger.info(f"Training complete. Adapter saved to: {workdir.adapter_path}")


@run.command("generate")
@common_run_options
@click.option(
    "--auto-discover-adapter",
    is_flag=True,
    default=False,
    help="Automatically find the latest trained adapter in --artifacts-path. "
    "Without this flag, --run-path must point to a specific trained run.",
)
@click.option(
    "--wandb-resume-job-id",
    type=str,
    default=None,
    required=False,
    help="Wandb run ID to resume, or path to a file containing the run ID. "
    "Overrides file-based run ID detection from workdir.",
)
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def run_generate(
    config_path: PathT,
    data_source: str,
    run_path: PathT | None,
    artifact_path: PathT | None,
    output_file: PathT | None,
    log_format: str | None,
    log_color: bool | None,
    log_file: PathT | None,
    verbose: int,
    wandb_mode: str | None = None,
    wandb_project: str | None = None,
    auto_discover_adapter: bool = False,
    wandb_resume_job_id: str | None = None,
    dataset_registry: str | None = None,
    **kwargs,
):
    """Run the generation stage only.

    This command loads a trained adapter and generates synthetic data.
    Requires 'run train' to have been executed first.

    Use --run-path to specify the exact run directory containing the trained model,
    or use --auto-discover-adapter with --artifact-path to automatically find
    the latest trained run.
    """
    # Create unified settings from CLI kwargs
    settings = CLISettings.from_cli_kwargs(
        data_source=data_source,
        config_path=config_path,
        artifact_path=artifact_path,
        run_path=run_path,
        output_file=output_file,
        log_file=log_file,
        log_color=log_color,
        log_format=log_format,
        verbose=verbose,
        wandb_mode=wandb_mode,
        wandb_project=wandb_project,
        synthesis_overrides=parse_overrides(kwargs),
        dataset_registry=dataset_registry,
    )

    os.environ["NSS_PHASE"] = "generate"
    # Generation always resumes from an existing workdir with a trained model
    run_logger, config, df, workdir = common_setup(
        settings=settings,
        resume=True,
        phase="generate",
        auto_discover_adapter=auto_discover_adapter,
        wandb_resume_job_id=wandb_resume_job_id,
    )
    from ..sdk.library_builder import SafeSynthesizer

    final_output_file = settings.output_file or workdir.output_file
    with traced_user("SafeSynthesizer"):
        ss = SafeSynthesizer(config, workdir=workdir)

        # Only set data source if provided via --data-source
        # Otherwise, load_from_save_path() will load from cached files
        if df is not None:
            ss = ss.with_data_source(df)

        try:
            ss = (
                ss.load_from_save_path()
                .process_data()
                .generate()
                .evaluate()
                .save_results(output_file=final_output_file)
            )
            ss.results.summary.log_summary(run_logger)
            ss.results.summary.timing.log_timing(run_logger)
            run_logger.info(f"Generation complete. Results saved to: {final_output_file}")
            ss.results.summary.log_wandb()
        finally:
            if hasattr(ss, "generator") and ss.generator is not None:
                ss.generator.teardown()
