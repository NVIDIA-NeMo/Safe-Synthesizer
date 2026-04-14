# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-flight validation checks for dataset and configuration compatibility."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .observability import get_logger

if TYPE_CHECKING:
    from rich.console import Console

    from .config.parameters import SafeSynthesizerParameters
    from .llm.metadata import ModelMetadata

logger = get_logger(__name__)


@dataclass(frozen=True)
class PreflightIssue:
    code: str
    severity: Literal["error", "warning"]
    check: str
    message: str


@dataclass(frozen=True)
class PreflightCheckResult:
    name: str
    label: str
    issues: list[PreflightIssue]

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)


@dataclass(frozen=True)
class PreflightReport:
    checks: list[PreflightCheckResult]

    @property
    def issues(self) -> list[PreflightIssue]:
        return [i for c in self.checks for i in c.issues]

    @property
    def errors(self) -> list[PreflightIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[PreflightIssue]:
        return [i for i in self.issues if i.severity == "warning"]


# ---------------------------------------------------------------------------
# Check functions
# ---------------------------------------------------------------------------


def check_gpu_resources(config: SafeSynthesizerParameters) -> list[PreflightIssue]:
    """Validate GPU availability and estimate VRAM headroom."""
    issues: list[PreflightIssue] = []

    try:
        import torch
    except ImportError:
        issues.append(
            PreflightIssue(
                code="no_gpu",
                severity="error",
                check="check_gpu_resources",
                message="PyTorch is not installed; cannot verify GPU availability.",
            )
        )
        return issues

    cuda_available = torch.cuda.is_available()

    if not cuda_available:
        issues.append(
            PreflightIssue(
                code="no_gpu",
                severity="error",
                check="check_gpu_resources",
                message="No CUDA GPU detected. Safe Synthesizer requires a CUDA-capable GPU.",
            )
        )

    if config.training.use_unsloth is True and not cuda_available:
        issues.append(
            PreflightIssue(
                code="unsloth_no_gpu",
                severity="error",
                check="check_gpu_resources",
                message="Unsloth training requires a CUDA GPU, but none is available.",
            )
        )

    if cuda_available:
        try:
            from .llm.utils import get_max_vram

            vram_map = get_max_vram()
            autoconfig = getattr(config, "_metadata_autoconfig", None)
            if autoconfig is None:
                autoconfig = getattr(config, "autoconfig", None)

            if vram_map and autoconfig is not None:
                hidden_size = getattr(autoconfig, "hidden_size", None)
                num_hidden_layers = getattr(autoconfig, "num_hidden_layers", None)
                if hidden_size and num_hidden_layers:
                    estimated_bytes = hidden_size * num_hidden_layers * 4 * 1.5
                    estimated_gib = estimated_bytes / (1024**3)
                    max_free_gib = max(
                        frac
                        * torch.cuda.get_device_properties(dev).total_mem
                        / (1024**3)
                        for dev, frac in vram_map.items()
                    )
                    if max_free_gib < estimated_gib:
                        issues.append(
                            PreflightIssue(
                                code="low_vram",
                                severity="warning",
                                check="check_gpu_resources",
                                message=(
                                    f"Estimated VRAM need ~{estimated_gib:.1f} GiB "
                                    f"exceeds available ~{max_free_gib:.1f} GiB. "
                                    "Training may OOM."
                                ),
                            )
                        )
        except Exception:
            pass

    return issues


_VALID_LOG_LEVELS = frozenset({"INFO", "WARNING", "ERROR", "CRITICAL", "DEBUG_DEPENDENCIES", "DEBUG"})


def check_env(config: SafeSynthesizerParameters) -> list[PreflightIssue]:
    """Check environment variables: inference keys, HF token, log level."""
    issues: list[PreflightIssue] = []

    if config.replace_pii is not None:
        if config.replace_pii.globals.classify.enable_classify is not False:
            if not os.environ.get("NSS_INFERENCE_KEY"):
                issues.append(
                    PreflightIssue(
                        code="inference_key_missing",
                        severity="warning",
                        check="check_env",
                        message=(
                            "NSS_INFERENCE_KEY is not set. "
                            "PII column classification will run in degraded mode."
                        ),
                    )
                )

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        issues.append(
            PreflightIssue(
                code="hf_token_missing",
                severity="warning",
                check="check_env",
                message=(
                    "HF_TOKEN is not set. Model downloads from gated repos will fail. "
                    "Set HF_TOKEN or HUGGING_FACE_HUB_TOKEN in your environment."
                ),
            )
        )

    log_level = os.environ.get("NSS_LOG_LEVEL")
    if log_level is not None and log_level.upper() not in _VALID_LOG_LEVELS:
        issues.append(
            PreflightIssue(
                code="invalid_log_level",
                severity="warning",
                check="check_env",
                message=(
                    f"NSS_LOG_LEVEL='{log_level}' is not valid. "
                    f"Expected one of: {', '.join(sorted(_VALID_LOG_LEVELS))}."
                ),
            )
        )

    log_format = os.environ.get("NSS_LOG_FORMAT")
    if log_format is not None and log_format.lower() not in ("json", "plain"):
        issues.append(
            PreflightIssue(
                code="invalid_log_format",
                severity="warning",
                check="check_env",
                message=f"NSS_LOG_FORMAT='{log_format}' is not valid. Expected 'json' or 'plain'.",
            )
        )

    return issues


def check_config(config: SafeSynthesizerParameters) -> list[PreflightIssue]:
    """Validate resolved configuration values."""
    issues: list[PreflightIssue] = []

    if config.training.num_input_records_to_sample == "auto":
        issues.append(
            PreflightIssue(
                code="auto_unresolved",
                severity="error",
                check="check_config",
                message="num_input_records_to_sample is still 'auto'; AutoConfigResolver may have failed.",
            )
        )

    if isinstance(config.training.num_input_records_to_sample, int):
        effective_batch = (
            config.training.batch_size * config.training.gradient_accumulation_steps
        )
        if effective_batch > config.training.num_input_records_to_sample:
            issues.append(
                PreflightIssue(
                    code="batch_exceeds_data",
                    severity="warning",
                    check="check_config",
                    message=(
                        f"Effective batch size ({effective_batch}) exceeds "
                        f"num_input_records_to_sample ({config.training.num_input_records_to_sample})."
                    ),
                )
            )

    return issues


def check_columns(
    data: pd.DataFrame, config: SafeSynthesizerParameters
) -> list[PreflightIssue]:
    """Validate that configured columns exist and are well-formed."""
    from .defaults import PSEUDO_GROUP_COLUMN

    issues: list[PreflightIssue] = []

    group_col = config.data.group_training_examples_by
    if group_col is not None:
        if group_col not in data.columns:
            issues.append(
                PreflightIssue(
                    code="column_not_found",
                    severity="error",
                    check="check_columns",
                    message=f"group_training_examples_by column '{group_col}' not found in data.",
                )
            )
        elif data[group_col].isna().any():
            issues.append(
                PreflightIssue(
                    code="column_nulls",
                    severity="error",
                    check="check_columns",
                    message=f"group_training_examples_by column '{group_col}' contains null values.",
                )
            )

    order_col = config.data.order_training_examples_by
    if order_col is not None:
        if order_col not in data.columns:
            issues.append(
                PreflightIssue(
                    code="column_not_found",
                    severity="error",
                    check="check_columns",
                    message=f"order_training_examples_by column '{order_col}' not found in data.",
                )
            )

    if PSEUDO_GROUP_COLUMN in data.columns:
        issues.append(
            PreflightIssue(
                code="pseudo_column_collision",
                severity="error",
                check="check_columns",
                message=f"Data contains reserved internal column '{PSEUDO_GROUP_COLUMN}'.",
            )
        )

    for col in data.columns:
        if data[col].dropna().nunique() == 1:
            issues.append(
                PreflightIssue(
                    code="constant_column",
                    severity="warning",
                    check="check_columns",
                    message=f"Column '{col}' has only 1 unique value.",
                )
            )

    return issues


def check_token_budget(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    metadata: ModelMetadata,
) -> list[PreflightIssue]:
    """Verify that records and groups fit within the model's context window."""
    from .defaults import PSEUDO_GROUP_COLUMN

    issues: list[PreflightIssue] = []

    if metadata.tokenizer is None:
        return [
            PreflightIssue(
                code="tokenizer_unavailable",
                severity="warning",
                check="check_token_budget",
                message="Tokenizer not available; token budget checks skipped.",
            )
        ]

    columns = [c for c in data.columns if c != PSEUDO_GROUP_COLUMN]
    schema_prompt = ",".join(f'"{c}":<unk>' for c in columns)
    schema_prompt_ids = metadata.tokenizer.encode(
        schema_prompt, add_special_tokens=False
    )

    max_new_tokens = metadata.max_seq_length - len(schema_prompt_ids) - 10
    if max_new_tokens <= 0:
        issues.append(
            PreflightIssue(
                code="schema_exceeds_context",
                severity="error",
                check="check_token_budget",
                message=(
                    f"Schema prompt ({len(schema_prompt_ids)} tokens) "
                    f"exceeds model context window ({metadata.max_seq_length})."
                ),
            )
        )
        return issues

    sample_size = min(len(data), 5000)
    sample = (
        data.sample(n=sample_size, random_state=42) if sample_size < len(data) else data
    )
    exceeded = 0
    for _, row in sample.iterrows():
        row_json = json.dumps({str(k): v for k, v in row.items()}, default=str)
        token_ids = metadata.tokenizer.encode(row_json, add_special_tokens=False)
        if len(token_ids) > max_new_tokens:
            exceeded += 1
    if exceeded:
        issues.append(
            PreflightIssue(
                code="record_exceeds_context",
                severity="error",
                check="check_token_budget",
                message=f"{exceeded} of {sample_size} sampled records exceed the token budget ({max_new_tokens} tokens).",
            )
        )

    group_col = config.data.group_training_examples_by
    if group_col is not None and group_col in data.columns:
        group_sizes = data.groupby(group_col).size().sort_values(ascending=False)
        top_groups = group_sizes.head(100).index
        groups_exceeded = 0
        for grp_key in top_groups:
            grp_df = data[data[group_col] == grp_key]
            concat_json = "".join(
                json.dumps({str(k): v for k, v in row.items()}, default=str)
                for _, row in grp_df.iterrows()
            )
            token_ids = metadata.tokenizer.encode(concat_json, add_special_tokens=False)
            if len(token_ids) > max_new_tokens:
                groups_exceeded += 1
        if groups_exceeded:
            issues.append(
                PreflightIssue(
                    code="group_exceeds_context",
                    severity="error",
                    check="check_token_budget",
                    message=(
                        f"{groups_exceeded} of the {len(top_groups)} largest groups "
                        f"exceed the token budget ({max_new_tokens} tokens)."
                    ),
                )
            )

    return issues


def check_timeseries(
    data: pd.DataFrame, config: SafeSynthesizerParameters
) -> list[PreflightIssue]:
    """Validate time-series column presence and integrity."""
    issues: list[PreflightIssue] = []

    if not config.time_series.is_timeseries:
        return issues

    ts_col = config.time_series.timestamp_column
    if ts_col is not None:
        if ts_col not in data.columns:
            issues.append(
                PreflightIssue(
                    code="timestamp_not_found",
                    severity="error",
                    check="check_timeseries",
                    message=f"Timestamp column '{ts_col}' not found in data.",
                )
            )
        elif data[ts_col].isna().any():
            issues.append(
                PreflightIssue(
                    code="timestamp_nulls",
                    severity="error",
                    check="check_timeseries",
                    message=f"Timestamp column '{ts_col}' contains null values.",
                )
            )

    return issues


def check_dataset_size(
    data: pd.DataFrame, config: SafeSynthesizerParameters
) -> list[PreflightIssue]:
    """Warn when the dataset is unusually small."""
    issues: list[PreflightIssue] = []
    n = len(data)

    if n < 200:
        issues.append(
            PreflightIssue(
                code="dataset_too_small",
                severity="error",
                check="check_dataset_size",
                message=f"Dataset has {n} rows; at least 200 are needed for meaningful training.",
            )
        )
    elif n < 1000:
        issues.append(
            PreflightIssue(
                code="dataset_small",
                severity="warning",
                check="check_dataset_size",
                message=f"Dataset has {n} rows; consider using more data for better quality.",
            )
        )

    group_col = config.data.group_training_examples_by
    if group_col and group_col in data.columns:
        group_sizes = data.groupby(group_col).size()
        tiny = (group_sizes < 3).sum()
        if tiny > 0:
            issues.append(
                PreflightIssue(
                    code="tiny_groups",
                    severity="warning",
                    check="check_dataset_size",
                    message=f"{tiny} group(s) have fewer than 3 rows.",
                )
            )

    return issues


def check_training_adequacy(
    data: pd.DataFrame, config: SafeSynthesizerParameters
) -> list[PreflightIssue]:
    """Flag extreme over/under-sampling and insufficient training steps."""
    issues: list[PreflightIssue] = []
    n_records = config.training.num_input_records_to_sample

    if not isinstance(n_records, int):
        return issues

    n = len(data)
    data_fraction = n_records / n if n > 0 else 0

    if data_fraction > 25:
        issues.append(
            PreflightIssue(
                code="extreme_oversampling",
                severity="warning",
                check="check_training_adequacy",
                message=f"num_input_records_to_sample is {data_fraction:.0f}x the dataset size; risk of overfitting.",
            )
        )

    if data_fraction < 1.0 and n_records < n:
        issues.append(
            PreflightIssue(
                code="undersampling",
                severity="warning",
                check="check_training_adequacy",
                message=(
                    f"num_input_records_to_sample ({n_records}) is less than "
                    f"dataset size ({n}); the model will not see all records."
                ),
            )
        )

    effective_batch = (
        config.training.batch_size * config.training.gradient_accumulation_steps
    )
    if effective_batch > 0:
        effective_steps = n_records / effective_batch
        if effective_steps < 10:
            issues.append(
                PreflightIssue(
                    code="few_training_steps",
                    severity="warning",
                    check="check_training_adequacy",
                    message=f"Effective training steps (~{effective_steps:.0f}) is very low; model may underfit.",
                )
            )

    return issues


def check_column_cardinality(
    data: pd.DataFrame, config: SafeSynthesizerParameters
) -> list[PreflightIssue]:
    """Flag non-numeric columns with near-unique values (likely identifiers)."""
    issues: list[PreflightIssue] = []
    n = len(data)
    if n == 0:
        return issues

    whitelist: set[str] = set()
    if config.data.group_training_examples_by:
        whitelist.add(config.data.group_training_examples_by)
    if config.data.order_training_examples_by:
        whitelist.add(config.data.order_training_examples_by)
    if config.time_series.timestamp_column:
        whitelist.add(config.time_series.timestamp_column)

    for col in data.columns:
        if col in whitelist:
            continue
        if pd.api.types.is_numeric_dtype(data[col]):
            continue
        if data[col].nunique() / n > 0.95:
            issues.append(
                PreflightIssue(
                    code="high_cardinality",
                    severity="warning",
                    check="check_column_cardinality",
                    message=f"Column '{col}' has >95% unique values; may be an identifier column.",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

from .observability import LogCategory, traced


@traced("preflight", category=LogCategory.USER)
def run_preflight(
    data: pd.DataFrame,
    config: SafeSynthesizerParameters,
    metadata: ModelMetadata,
) -> PreflightReport:
    """Execute all pre-flight checks and return a structured report."""
    checks: list[PreflightCheckResult] = []

    def _run(name: str, label: str, issues: list[PreflightIssue]) -> list[PreflightIssue]:
        checks.append(PreflightCheckResult(name=name, label=label, issues=issues))
        return issues

    _run("gpu", "GPU resources", check_gpu_resources(config))
    _run("env", "Environment variables", check_env(config))
    _run("config", "Configuration", check_config(config))

    col_issues = _run("columns", "Column validation", check_columns(data, config))
    if not any(i.severity == "error" for i in col_issues):
        _run("token_budget", "Token budget", check_token_budget(data, config, metadata))

    if config.time_series.is_timeseries:
        _run("timeseries", "Time series", check_timeseries(data, config))
    _run("dataset_size", "Dataset size", check_dataset_size(data, config))
    _run("training", "Training adequacy", check_training_adequacy(data, config))
    _run("cardinality", "Column cardinality", check_column_cardinality(data, config))

    report = PreflightReport(checks=checks)
    logger.runtime.debug(
        "Preflight complete",
        extra={
            "errors": len(report.errors),
            "warnings": len(report.warnings),
        },
    )
    return report


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def format_preflight_report(
    report: PreflightReport | None,
    config_path: Path | None = None,
    data_source: str | None = None,
    artifact_dir: Path | None = None,
    log_file: Path | None = None,
    run_info: dict[str, str] | None = None,
    console: "Console | None" = None,
) -> None:
    """Print a Rich-formatted validation report to the console."""
    from rich import box
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    if console is None:
        console = Console(soft_wrap=True)
    if report is None:
        report = PreflightReport(checks=[])

    n_errors = len(report.errors)
    n_warns = len(report.warnings)

    if n_errors:
        header = Text(
            f"Pre-flight validation failed  {n_errors} error(s), {n_warns} warning(s)",
            style="bold red",
        )
    elif n_warns:
        header = Text(f"Pre-flight validation passed  {n_warns} warning(s)", style="bold yellow")
    else:
        header = Text("Pre-flight validation passed", style="bold green")

    console.print()
    if run_info:
        info_parts = [f"[dim]{k}:[/] {v}" for k, v in run_info.items()]
        console.print("  ".join(info_parts))
        console.print()
    console.print(header)
    console.print()

    width = max(int(console.width * 0.8), 80)
    tbl = Table(
        box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
        padding=(0, 1), width=width,
    )
    tbl.add_column("Check", style="bold", min_width=20, no_wrap=True)
    tbl.add_column("", width=1, justify="center")
    tbl.add_column("Code", style="dim", min_width=16, no_wrap=True)
    tbl.add_column("Message", ratio=1)

    for check in report.checks:
        check_errors = [i for i in check.issues if i.severity == "error"]
        check_warns = [i for i in check.issues if i.severity == "warning"]

        if not check.issues:
            tbl.add_row(check.label, Text("✓", style="green"), "", "")
        else:
            first = True
            for issue in check_errors:
                tbl.add_row(
                    check.label if first else "",
                    Text("✗", style="bold red"),
                    issue.code,
                    issue.message,
                )
                first = False
            for issue in check_warns:
                tbl.add_row(
                    check.label if first else "",
                    Text("⚠", style="yellow"),
                    issue.code,
                    issue.message,
                )
                first = False

    if n_errors:
        tbl.caption = f"[bold red]{n_errors} error(s)[/], {n_warns} warning(s) — fix errors before running"
    elif n_warns:
        tbl.caption = f"[yellow]{n_warns} warning(s)[/] — review before running"
    else:
        tbl.caption = "[green]all checks passed[/]"
    tbl.caption_justify = "left"

    console.print(tbl)
    console.print()

    if artifact_dir is not None:
        from rich.tree import Tree

        console.print(Text("Program output will be stored at:", style="dim"))
        tree = Tree(str(artifact_dir))
        if config_path is not None:
            try:
                rel = config_path.relative_to(artifact_dir)
            except ValueError:
                rel = config_path
            tree.add(f"[bold]{rel}[/]  [dim](resolved config)[/]")
        if log_file is not None:
            try:
                rel_log = log_file.relative_to(artifact_dir)
            except ValueError:
                rel_log = log_file
            tree.add(f"{rel_log}  [dim](log file)[/]")
        console.print(tree)
        console.print()
    elif config_path is not None:
        console.print(Text("Resolved config", style="dim"))
        console.print(f"  {config_path}")
        console.print()

    if not n_errors and config_path is not None and data_source is not None:
        console.print(Text("Run with the resolved configuration:", style="dim"))
        console.print(f"  safe-synthesizer run --data-source {data_source} \\")
        console.print(f"    --config {config_path}")
        console.print()
        console.print(
            Text(
                "Note: the full run will create a new timestamped directory for its artifacts unless you specify a path.",
                style="dim italic",
            )
        )
