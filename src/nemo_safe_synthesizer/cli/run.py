# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI run commands for Safe Synthesizer."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

from ..config import SafeSynthesizerParameters
from ..configurator.pydantic_click_options import (
    parse_overrides,
    pydantic_options,
)
from ..errors import UserError
from ..observability import traced_user
from ..telemetry import DeploymentTypeEnum, TaskStatusEnum
from ..tooling import PreflightRenderContext, render_preflight_report
from .settings import CLISettings
from .utils import (
    CLI_NESTED_FIELD_SEPARATOR,
    PathT,
    common_setup,
)

if TYPE_CHECKING:
    import pandas as pd

    from ..sdk.library_builder import SafeSynthesizer
    from .artifact_structure import Workdir


def common_run_options(f: Callable[..., object]) -> Callable[..., object]:
    """Decorator to add common options for run commands.

    Apply this above ``@pydantic_options`` in source order. Python applies
    decorators bottom-up, so the shared command options are added after the
    generated parameter options. Environment-variable handling stays in
    ``CLISettings`` rather than Click ``envvar=`` declarations, so precedence is
    centralized in one settings model.
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
    options.append(
        click.option(
            "--inference-endpoint-url",
            type=str,
            required=False,
            default=None,
            help="OpenAI-compatible inference endpoint URL for PII column classification. "
            "Can also be set via NSS_INFERENCE_ENDPOINT env var.",
        )
    )
    options.append(
        click.option(
            "--inference-api-key",
            type=str,
            required=False,
            default=None,
            help="API key for the inference endpoint used in PII column classification. "
            "Can also be set via NSS_INFERENCE_KEY env var.",
        )
    )
    options.append(
        click.option(
            "--inference-model-id",
            type=str,
            required=False,
            default=None,
            help="Model ID sent to the inference endpoint for PII column classification. "
            "Can also be set via NSS_INFERENCE_MODEL env var. "
            "[default: qwen/qwen3-next-80b-a3b-instruct]",
        )
    )
    options.append(
        click.option(
            "--enable-huggingface-remote/--disable-huggingface-remote",
            "huggingface_remote",
            required=False,
            default=None,
            help="Allow or block Hugging Face remote downloads for both the base model "
            "and GLiNER. --disable-huggingface-remote forces a fully offline run by "
            "setting HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE; both must already be "
            "cached. Equivalent to setting HF_HUB_OFFLINE in the environment. When "
            "neither flag is given, the run inherits HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE "
            "from the environment (remote downloads enabled when unset). "
            "[default: --enable-huggingface-remote]",
        )
    )
    options.append(
        click.option(
            "--cpu-count",
            type=int,
            required=False,
            default=None,
            help="Number of CPU worker processes used for NER (PII replacement). "
            "Can also be set via NSS_PII_REPLACER_CPU_COUNT env var. "
            "[default: max(1, cpu_count - 1)]",
        )
    )
    # Apply each option decorator in reverse order (decorators apply bottom-up)
    for option in reversed(options):
        f = option(f)
    return f


def _parse_run_overrides(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse generated config options plus manual run aliases into config overrides."""
    return parse_overrides(kwargs)


# CLISettings fields populated from common_run_options flags. ``synthesis_overrides``
# is excluded -- it is derived from the leftover pydantic_options kwargs, not bound
# to a single flag. ``observability``/``wandb`` are nested sub-settings with no CLI
# flag, so they never appear in command kwargs.
_CLI_SETTINGS_FIELDS: frozenset[str] = frozenset(CLISettings.model_fields) - {"synthesis_overrides"}


def _settings_from_run_kwargs(kwargs: dict[str, Any]) -> CLISettings:
    """Build ``CLISettings`` from a run command's ``**kwargs``.

    ``common_run_options`` binds each infrastructure flag to a kwarg whose name
    matches a ``CLISettings`` field; those are pulled out here. Everything left
    (the ``pydantic_options`` ``--section__field`` options) becomes synthesis
    overrides. This keeps the three run commands from re-listing the shared flag
    set in both their signature and their settings construction -- adding a flag
    now means editing ``common_run_options`` and ``CLISettings`` only.

    ``kwargs`` is mutated: matched settings keys are popped before the remainder
    is parsed into overrides.
    """
    settings_kwargs = {name: kwargs.pop(name) for name in _CLI_SETTINGS_FIELDS if name in kwargs}
    settings_kwargs["synthesis_overrides"] = _parse_run_overrides(kwargs)
    return CLISettings.from_cli_kwargs(**settings_kwargs)


def _set_cli_deployment_type_default() -> None:
    """Default telemetry deployment type for CLI commands without overriding Slurm or explicit settings."""
    os.environ.setdefault("NEMO_DEPLOYMENT_TYPE", DeploymentTypeEnum.CLI.value)


def _format_dataset_runtime_info(data: pd.DataFrame) -> str:
    """Format dataset size summary for validate runtime info."""
    return f"{len(data):,} rows, {len(data.columns):,} columns"


def _build_validate_run_info(
    *,
    version: str,
    model_name: str,
    data: pd.DataFrame,
    training_records: int | None = None,
) -> dict[str, str]:
    """Build runtime info displayed in validate output.

    ``training_records`` is the size of the training split that preflight
    actually checked; it is shown alongside the input dataset size so the
    scope of the report is obvious.
    """
    info: dict[str, str] = {
        "nemo-safe-synthesizer": version,
        "model": model_name,
        "input data": _format_dataset_runtime_info(data),
    }
    if training_records is not None:
        info["training split"] = f"{training_records:,} rows (pre-flight scope)"
    info["log level"] = os.environ.get("NSS_LOG_LEVEL", "INFO")
    return info


def _run_validate_and_render(
    nss: SafeSynthesizer,
    *,
    settings: CLISettings,
    workdir: Workdir,
    config: SafeSynthesizerParameters,
    data: pd.DataFrame,
) -> None:
    """Run preflight in validate mode and render the resulting report.

    Shared by the ``run`` (end-to-end) and ``run train`` command paths so
    the ``--validate`` branch in each stays a single line.

    Note: ``process_data(check_only=True)`` deliberately skips PII
    replacement, so preflight runs against the pre-replacement training
    split. This is why ``--validate`` is documented as a best-effort
    fail-fast gate rather than a full-run guarantee.
    """
    click.echo("Running pre-flight validation...", nl=False)
    # ``process_data(check_only=True)`` raises ``UserError`` (specifically
    # ``ParameterError``) when preflight surfaces errors, but it populates
    # ``nss.preflight_report`` first. Catch the raise so the Rich report still
    # renders before we propagate the failure -- otherwise the user gets only
    # the bare traceback text.
    error: UserError | None = None
    try:
        nss.process_data(check_only=True)
    except UserError as exc:
        error = exc
    finally:
        _clear_progress_line()

    # intentionally deferred import to avoid delay in user startup
    from ..package_info import __version__
    from ..preflight import get_registry

    if nss.preflight_report is not None:
        render_preflight_report(
            nss.preflight_report,
            registry=get_registry(),
            context=_build_validate_render_context(
                config_path=nss._preflight_config_path,
                data_source=settings.data_source,
                artifact_dir=workdir.run_dir,
                log_file=workdir.log_file,
                version=__version__,
                model_name=config.training.pretrained_model,
                data=data,
                training_records=len(nss._training_df) if nss._training_df is not None else None,
            ),
        )

    if error is not None:
        raise error


def _clear_progress_line() -> None:
    r"""Clear the ``Running pre-flight validation...`` progress line.

    Emits the ANSI clear sequence only when stdout is a TTY so CI logs and
    other non-terminal sinks don't show raw ``\\r\\x1b[K`` control bytes.
    On non-TTYs, just drop to a new line.
    """
    if sys.stdout.isatty():
        click.echo("\r\033[K", nl=False)
    else:
        click.echo()


def _build_validate_render_context(
    *,
    config_path: PathT | None,
    data_source: str | None,
    artifact_dir: PathT | None,
    log_file: PathT | None,
    version: str,
    model_name: str,
    data: pd.DataFrame,
    training_records: int | None = None,
) -> PreflightRenderContext:
    """Build display context for validate-mode preflight rendering."""
    return PreflightRenderContext(
        config_path=Path(config_path) if config_path is not None else None,
        data_source=data_source,
        artifact_dir=Path(artifact_dir) if artifact_dir is not None else None,
        log_file=Path(log_file) if log_file is not None else None,
        run_info=_build_validate_run_info(
            version=version,
            model_name=model_name,
            data=data,
            training_records=training_records,
        ),
    )


@click.group(invoke_without_command=True)
@click.pass_context
@common_run_options
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="Run pre-flight validation only, then exit without training or generating.",
)
def run(
    ctx: click.Context,
    validate: bool = False,
    **kwargs: Any,
) -> None:
    """Run the Safe Synthesizer end-to-end pipeline.

    Without a subcommand, runs the full end-to-end pipeline.
    Use 'run train' or 'run generate' for individual stages.
    """
    # If a subcommand is invoked, skip the default behavior
    if ctx.invoked_subcommand is not None:
        return

    _set_cli_deployment_type_default()

    settings = _settings_from_run_kwargs(kwargs)

    if validate:
        os.environ["NSS_PHASE"] = "process_data"
    else:
        os.environ["NSS_PHASE"] = "end_to_end"

    run_logger, config, df, workdir = common_setup(
        settings=settings,
        phase="process_data" if validate else "end_to_end",
        skip_wandb=validate,
        quiet=validate,
        run_name="validate" if validate else None,
    )

    try:
        run_logger.warning("Nemo Safe Synthesizer starting")
        run_logger.debug("running with: ", extra={"config": config.model_dump()})

        with traced_user("SafeSynthesizer"):
            from ..sdk.library_builder import SafeSynthesizer

            assert df is not None
            nss: SafeSynthesizer = SafeSynthesizer(
                config=config,
                workdir=workdir,
                emit_telemetry=config.emit_telemetry,
            ).with_data_source(df)

            if validate:
                _run_validate_and_render(nss, settings=settings, workdir=workdir, config=config, data=df)
                return

            try:
                nss.run(output_file=settings.output_file)
                nss.results.summary.log_summary(run_logger)
                nss.results.summary.timing.log_timing(run_logger)
                nss.results.summary.log_wandb()
            finally:
                if hasattr(nss, "generator") and nss.generator is not None:
                    nss.generator.teardown()
    except UserError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1)


@run.command("train")
@common_run_options
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="Run pre-flight validation only, then exit without training or generating.",
)
def run_train(
    validate: bool = False,
    **kwargs: Any,
) -> None:
    """Run the training stage only.

    This command processes data and trains the model, saving the adapter to the run directory.
    Use 'run generate' afterwards to generate synthetic data from the trained adapter.
    """
    _set_cli_deployment_type_default()

    settings = _settings_from_run_kwargs(kwargs)

    if validate:
        os.environ["NSS_PHASE"] = "process_data"
    else:
        os.environ["NSS_PHASE"] = "train"

    run_logger, config, df, workdir = common_setup(
        settings=settings,
        phase="process_data" if validate else "train",
        skip_wandb=validate,
        quiet=validate,
        run_name="validate" if validate else None,
    )
    from ..sdk.library_builder import SafeSynthesizer

    try:
        with traced_user("SafeSynthesizer"):
            assert df is not None
            nss = SafeSynthesizer(
                config,
                workdir=workdir,
                emit_telemetry=config.emit_telemetry,
            ).with_data_source(df)

            if validate:
                _run_validate_and_render(nss, settings=settings, workdir=workdir, config=config, data=df)
                return

            nss.process_data().train()
            run_logger.info(f"Training complete. Adapter saved to: {workdir.adapter_path}")
    except UserError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1)


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
    auto_discover_adapter: bool = False,
    wandb_resume_job_id: str | None = None,
    **kwargs: Any,
) -> None:
    """Run the generation stage only.

    This command loads a trained adapter and generates synthetic data.
    Requires 'run train' to have been executed first.

    Use --run-path to specify the exact run directory containing the trained model,
    or use --auto-discover-adapter with --artifact-path to automatically find
    the latest trained run.
    """
    _set_cli_deployment_type_default()

    # Create unified settings from CLI kwargs
    settings = _settings_from_run_kwargs(kwargs)

    os.environ["NSS_PHASE"] = "generate"
    # Generation always resumes from an existing workdir with a trained model
    run_logger, config, df, workdir = common_setup(
        settings=settings,
        resume=True,
        phase="generate",
        auto_discover_adapter=auto_discover_adapter,
        wandb_resume_job_id=wandb_resume_job_id,
    )
    from ..sdk.library_builder import SafeSynthesizer, _emit_nss_telemetry

    final_output_file = settings.output_file or workdir.output_file
    with traced_user("SafeSynthesizer"):
        nss = SafeSynthesizer(
            config,
            workdir=workdir,
            emit_telemetry=config.emit_telemetry,
        )

        # Only set data source if provided via --data-source
        # Otherwise, load_from_save_path() will load from cached files
        if df is not None:
            nss = nss.with_data_source(df)

        try:
            nss = (
                nss.load_from_save_path(runtime_config=config)
                .process_data()
                .generate()
                .evaluate()
                .save_results(output_file=final_output_file)
            )
            _emit_nss_telemetry(nss, TaskStatusEnum.COMPLETED)
            nss.results.summary.log_summary(run_logger)
            nss.results.summary.timing.log_timing(run_logger)
            run_logger.info(f"Generation complete. Results saved to: {final_output_file}")
            nss.results.summary.log_wandb()
        except KeyboardInterrupt:
            _emit_nss_telemetry(nss, TaskStatusEnum.CANCELED)
            raise
        except Exception:
            _emit_nss_telemetry(nss, TaskStatusEnum.ERROR)
            raise
        finally:
            if hasattr(nss, "generator") and nss.generator is not None:
                nss.generator.teardown()
