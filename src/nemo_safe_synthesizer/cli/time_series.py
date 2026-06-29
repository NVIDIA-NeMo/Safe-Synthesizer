# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time-series exploration commands."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Literal, cast

import click
import pandas as pd

from ..config import SafeSynthesizerParameters
from ..configurator.pydantic_click_options import pydantic_options
from ..errors import UserError
from ..generation.timeseries_backend import TimeseriesBackend
from ..observability import traced_user
from ..utils import write_json
from .run import _set_cli_deployment_type_default, _settings_from_run_kwargs, common_run_options
from .utils import CLI_NESTED_FIELD_SEPARATOR, common_setup

TimeSeriesInitStrategy = Literal["training_prefill", "empty", "start_instruction", "partial_record_prefix"]

DEFAULT_VARIANTS: tuple[TimeSeriesInitStrategy, ...] = (
    "training_prefill",
    "empty",
    "start_instruction",
    "partial_record_prefix",
)


@click.group("time-series")
def time_series() -> None:
    """Time-series exploration utilities."""


@time_series.command("cold-start")
@common_run_options
@click.option(
    "--variant",
    "variants",
    multiple=True,
    type=click.Choice(DEFAULT_VARIANTS),
    help="Initialization variant to run. Repeat to run several. Defaults to all variants.",
)
@click.option(
    "--experiment-dir",
    type=click.Path(exists=False, dir_okay=True, file_okay=False, resolve_path=True),
    default=None,
    help="Directory for experiment artifacts. Defaults to <run>/generate/time_series_cold_start.",
)
@click.option(
    "--auto-discover-adapter",
    is_flag=True,
    default=False,
    help="Automatically find the latest trained adapter in --artifacts-path.",
)
@click.option(
    "--wandb-resume-job-id",
    type=str,
    default=None,
    required=False,
    help="Wandb run ID to resume, or path to a file containing the run ID.",
)
@click.option(
    "--plot-column",
    "plot_columns",
    multiple=True,
    help="Numeric column to include in the HTML traces. Defaults to up to three numeric columns.",
)
@click.option(
    "--max-plot-groups",
    type=int,
    default=5,
    show_default=True,
    help="Maximum number of groups to show per variant in the HTML traces.",
)
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def cold_start(
    variants: tuple[str, ...],
    experiment_dir: str | None,
    auto_discover_adapter: bool,
    wandb_resume_job_id: str | None,
    plot_columns: tuple[str, ...],
    max_plot_groups: int,
    **kwargs: Any,
) -> None:
    """Run time-series cold-start generation variants against an existing adapter."""
    _set_cli_deployment_type_default()
    settings = _settings_from_run_kwargs(kwargs)
    selected_variants = cast(tuple[TimeSeriesInitStrategy, ...], variants or DEFAULT_VARIANTS)

    os.environ["NSS_PHASE"] = "generate"
    run_logger, config, df, workdir = common_setup(
        settings=settings,
        resume=True,
        phase="generate",
        auto_discover_adapter=auto_discover_adapter,
        wandb_resume_job_id=wandb_resume_job_id,
    )

    if not config.time_series.is_timeseries:
        raise click.ClickException("time-series cold-start experiments require time_series.is_timeseries=true.")

    output_dir = (
        Path(experiment_dir) if experiment_dir is not None else workdir.generate.path / "time_series_cold_start"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        variant_outputs = []
        with traced_user("SafeSynthesizer.time_series_cold_start"):
            for variant in selected_variants:
                run_logger.info(f"Running time-series cold-start variant: {variant}")
                variant_output = _run_variant(
                    base_config=config,
                    variant=variant,
                    df=df,
                    workdir=workdir,
                    output_dir=output_dir / variant,
                )
                variant_outputs.append(variant_output)

        _write_summary_artifacts(
            output_dir=output_dir,
            variant_outputs=variant_outputs,
            plot_columns=list(plot_columns),
            max_plot_groups=max_plot_groups,
        )
        run_logger.info(f"Time-series cold-start experiment artifacts saved to: {output_dir}")
    except UserError as exc:
        click.secho(str(exc), fg="red", err=True)
        raise SystemExit(1) from exc


def _run_variant(
    *,
    base_config: SafeSynthesizerParameters,
    variant: TimeSeriesInitStrategy,
    df: pd.DataFrame | None,
    workdir: Any,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one initialization variant and write per-variant artifacts."""
    from ..sdk.library_builder import SafeSynthesizer

    output_dir.mkdir(parents=True, exist_ok=True)
    config = base_config.model_copy(deep=True)
    config.time_series = config.time_series.model_copy(update={"initialization_strategy": variant})

    nss = SafeSynthesizer(config, workdir=workdir, emit_telemetry=config.emit_telemetry)
    if df is not None:
        nss = nss.with_data_source(df)

    synthetic_path = output_dir / "synthetic_data.csv"
    try:
        nss = nss.load_from_save_path(runtime_config=config).process_data().generate()
        generator = nss.generator
        results = generator.gen_results
        result_df = results.df
        result_df.to_csv(synthetic_path, index=False)

        diagnostics: dict[str, Any] = {}
        if isinstance(generator, TimeseriesBackend):
            diagnostics = generator.generation_diagnostics()
        _write_variant_diagnostics(output_dir, diagnostics)

        metrics = _build_variant_metrics(variant, results, diagnostics)
    except Exception as exc:
        generator = getattr(nss, "generator", None)
        diagnostics = {"error": str(exc), "error_type": type(exc).__name__}
        if isinstance(generator, TimeseriesBackend):
            diagnostics.update(generator.generation_diagnostics())
        _write_variant_diagnostics(output_dir, diagnostics)

        result_df = pd.DataFrame()
        result_df.to_csv(synthetic_path, index=False)
        metrics = _build_failed_variant_metrics(variant, exc, diagnostics)

    write_json(metrics, output_dir / "metrics.json", indent=2)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    return {
        "variant": variant,
        "output_dir": output_dir,
        "synthetic_path": synthetic_path,
        "df": result_df,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def _write_variant_diagnostics(output_dir: Path, diagnostics: dict[str, Any]) -> None:
    """Write prompt, first-record, and batch diagnostics for one variant."""
    write_json(diagnostics, output_dir / "diagnostics.json", indent=2)
    write_json(
        {"prompt_previews": diagnostics.get("prompt_previews", [])},
        output_dir / "prompt_previews.json",
        indent=2,
    )

    first_records = diagnostics.get("first_record_diagnostics", [])
    pd.DataFrame(first_records).to_csv(output_dir / "first_record_diagnostics.csv", index=False)

    batch_rows = []
    for row in diagnostics.get("batch_diagnostics", []):
        flat = dict(row)
        flat["finish_reasons"] = json.dumps(flat.get("finish_reasons", {}), sort_keys=True)
        flat["errors"] = json.dumps(flat.get("errors", []), sort_keys=True)
        batch_rows.append(flat)
    pd.DataFrame(batch_rows).to_csv(output_dir / "batch_diagnostics.csv", index=False)

    completion_rows = []
    for row in diagnostics.get("completion_diagnostics", []):
        completion_rows.append(dict(row))
    pd.DataFrame(completion_rows).to_csv(output_dir / "completion_diagnostics.csv", index=False)


def _build_variant_metrics(variant: str, results: Any, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Build a compact metrics row for one variant."""
    first_records = diagnostics.get("first_record_diagnostics", [])
    comparable = [row for row in first_records if row.get("matched") is not None]
    first_match_rate = (
        sum(1 for row in comparable if row.get("matched")) / len(comparable) if comparable else None
    )

    error_counter: Counter[str] = Counter()
    finish_counter: Counter[str] = Counter()
    for row in diagnostics.get("batch_diagnostics", []):
        finish_counter.update(row.get("finish_reasons", {}))
        for error in row.get("errors", []):
            label = f"{error.get('validator')}: {error.get('message')}"
            error_counter[label] += int(error.get("count", 0))

    return {
        "variant": variant,
        "status": str(results.status),
        "num_valid_records": results.num_valid_records,
        "num_invalid_records": results.num_invalid_records,
        "num_prompts": results.num_prompts,
        "valid_record_fraction": results.valid_record_fraction,
        "first_timestamp_match_rate": first_match_rate,
        "elapsed_time": results.elapsed_time,
        "tokens_per_prompt": results.tokens_per_prompt,
        "finish_reasons": json.dumps(dict(finish_counter), sort_keys=True),
        "top_invalid_reasons": json.dumps(dict(error_counter.most_common(10)), sort_keys=True),
    }


def _build_failed_variant_metrics(variant: str, exc: Exception, diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Build a metrics row when generation fails before producing results."""
    first_records = diagnostics.get("first_record_diagnostics", [])
    batch_diagnostics = diagnostics.get("batch_diagnostics", [])
    valid_records = sum(int(row.get("num_valid_records", 0)) for row in batch_diagnostics)
    invalid_records = sum(int(row.get("num_invalid_records", 0)) for row in batch_diagnostics)
    total_records = valid_records + invalid_records
    comparable = [row for row in first_records if row.get("matched") is not None]
    first_match_rate = (
        sum(1 for row in comparable if row.get("matched")) / len(comparable) if comparable else None
    )

    return {
        "variant": variant,
        "status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "num_valid_records": valid_records,
        "num_invalid_records": invalid_records,
        "num_prompts": len(diagnostics.get("completion_diagnostics", [])),
        "valid_record_fraction": valid_records / total_records if total_records else 0.0,
        "first_timestamp_match_rate": first_match_rate,
        "elapsed_time": None,
        "tokens_per_prompt": None,
        "finish_reasons": "{}",
        "top_invalid_reasons": "{}",
    }


def _write_summary_artifacts(
    *,
    output_dir: Path,
    variant_outputs: list[dict[str, Any]],
    plot_columns: list[str],
    max_plot_groups: int,
) -> None:
    """Write top-level experiment summary artifacts."""
    metrics_df = pd.DataFrame([item["metrics"] for item in variant_outputs])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    write_json({"variants": [item["metrics"] for item in variant_outputs]}, output_dir / "metrics.json", indent=2)
    _write_html_report(output_dir / "report.html", variant_outputs, metrics_df, plot_columns, max_plot_groups)


def _write_html_report(
    path: Path,
    variant_outputs: list[dict[str, Any]],
    metrics_df: pd.DataFrame,
    plot_columns: list[str],
    max_plot_groups: int,
) -> None:
    """Write a lightweight HTML report for quick visual inspection."""
    sections = [
        "<html><head><title>Time-Series Cold-Start Experiment</title></head><body>",
        "<h1>Time-Series Cold-Start Experiment</h1>",
        "<h2>Metrics</h2>",
        metrics_df.to_html(index=False, escape=True),
    ]
    sections.extend(_build_plot_sections(variant_outputs, plot_columns, max_plot_groups))
    sections.append("</body></html>")
    path.write_text("\n".join(sections), encoding="utf-8")


def _build_plot_sections(
    variant_outputs: list[dict[str, Any]],
    plot_columns: list[str],
    max_plot_groups: int,
) -> list[str]:
    """Build Plotly sections, falling back to text when Plotly is unavailable."""
    try:
        import plotly.graph_objects as go
        from plotly.io import to_html
    except ImportError:
        return ["<p>Plotly is not installed; only CSV/JSON diagnostics were written.</p>"]

    first_df = next((item["df"] for item in variant_outputs if not item["df"].empty), pd.DataFrame())
    if first_df.empty:
        return ["<p>No synthetic records were generated for plotting.</p>"]

    time_column = _infer_time_column(variant_outputs)
    group_column = _infer_group_column(variant_outputs, time_column)
    columns = plot_columns or [
        column
        for column in first_df.select_dtypes(include="number").columns
        if column not in {time_column, group_column}
    ][:3]
    if not columns or time_column is None:
        return ["<p>No timestamp and numeric value columns were available for plotting.</p>"]

    sections = ["<h2>Time-Series Traces</h2>"]
    for column in columns:
        fig = go.Figure()
        for item in variant_outputs:
            df = item["df"]
            if column not in df.columns or time_column not in df.columns:
                continue
            groups = [None]
            if group_column is not None and group_column in df.columns:
                groups = list(df[group_column].dropna().astype(str).unique()[:max_plot_groups])
            for group in groups:
                trace_df = df if group is None else df[df[group_column].astype(str) == group]
                name = item["variant"] if group is None else f"{item['variant']} / {group}"
                fig.add_trace(
                    go.Scatter(x=trace_df[time_column], y=trace_df[column], mode="lines+markers", name=name)
                )
        fig.update_layout(title=f"{column} over time", xaxis_title=time_column, yaxis_title=column)
        sections.append(to_html(fig, full_html=False, include_plotlyjs="cdn"))
    return sections


def _infer_time_column(variant_outputs: list[dict[str, Any]]) -> str | None:
    """Infer the timestamp column from prompt diagnostics or generated data."""
    for item in variant_outputs:
        time_column = item.get("diagnostics", {}).get("timestamp_column")
        if time_column:
            return str(time_column)
    for item in variant_outputs:
        df = item["df"]
        for column in df.columns:
            if "time" in column.lower() or "date" in column.lower():
                return column
    return None


def _infer_group_column(variant_outputs: list[dict[str, Any]], time_column: str | None) -> str | None:
    """Infer a likely group column from generated data."""
    for item in variant_outputs:
        group_column = item.get("diagnostics", {}).get("group_column")
        if group_column:
            return str(group_column)
    for item in variant_outputs:
        df = item["df"]
        for column in df.columns:
            if column == time_column:
                continue
            if "group" in column.lower() or column.lower().endswith("_id") or column.lower() == "id":
                return column
    return None
