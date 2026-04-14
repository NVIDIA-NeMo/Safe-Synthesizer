# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from rich.console import Console

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.config.training import TrainingHyperparams
from nemo_safe_synthesizer.preflight import (
    PreflightCheckResult,
    PreflightIssue,
    PreflightReport,
    check_column_cardinality,
    check_columns,
    check_config,
    check_dataset_size,
    check_env,
    check_gpu_resources,
    check_timeseries,
    check_token_budget,
    check_training_adequacy,
    format_preflight_report,
    run_preflight,
)


# ---------------------------------------------------------------------------
# check_gpu_resources
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_gpu_resources_no_gpu(default_config):
    with patch("torch.cuda.is_available", return_value=False):
        issues = check_gpu_resources(default_config)
    assert any(i.code == "no_gpu" and i.severity == "error" for i in issues)


@pytest.mark.unit
def test_check_gpu_resources_with_gpu(default_config):
    with patch("torch.cuda.is_available", return_value=True):
        issues = check_gpu_resources(default_config)
    assert not any(i.code == "no_gpu" for i in issues)


@pytest.mark.unit
def test_check_gpu_resources_unsloth_no_gpu():
    config = SafeSynthesizerParameters(training=TrainingHyperparams(use_unsloth=True))
    with patch("torch.cuda.is_available", return_value=False):
        issues = check_gpu_resources(config)
    assert any(i.code == "unsloth_no_gpu" for i in issues)


# ---------------------------------------------------------------------------
# check_env
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_env_inference_key_missing(default_config):
    with patch.dict("os.environ", {}, clear=True):
        issues = check_env(default_config)
    assert any(i.code == "inference_key_missing" for i in issues)


@pytest.mark.unit
def test_check_env_inference_key_present(default_config):
    with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "test-key", "HF_TOKEN": "hf_xxx"}):
        issues = check_env(default_config)
    assert not any(i.code == "inference_key_missing" for i in issues)


@pytest.mark.unit
def test_check_env_pii_disabled():
    config = SafeSynthesizerParameters(replace_pii=None)
    with patch.dict("os.environ", {"HF_TOKEN": "hf_xxx"}, clear=True):
        issues = check_env(config)
    assert not any(i.code == "inference_key_missing" for i in issues)


@pytest.mark.unit
def test_check_env_hf_token_missing(default_config):
    with patch.dict("os.environ", {}, clear=True):
        issues = check_env(default_config)
    assert any(i.code == "hf_token_missing" for i in issues)


@pytest.mark.unit
def test_check_env_hf_token_present(default_config):
    with patch.dict("os.environ", {"HF_TOKEN": "hf_xxx"}, clear=True):
        issues = check_env(default_config)
    assert not any(i.code == "hf_token_missing" for i in issues)


@pytest.mark.unit
def test_check_env_hugging_face_hub_token(default_config):
    with patch.dict("os.environ", {"HUGGING_FACE_HUB_TOKEN": "hf_xxx"}, clear=True):
        issues = check_env(default_config)
    assert not any(i.code == "hf_token_missing" for i in issues)


@pytest.mark.unit
def test_check_env_invalid_log_level(default_config):
    with patch.dict("os.environ", {"NSS_LOG_LEVEL": "foo", "HF_TOKEN": "hf_xxx"}, clear=True):
        issues = check_env(default_config)
    assert any(i.code == "invalid_log_level" for i in issues)


@pytest.mark.unit
def test_check_env_valid_log_level(default_config):
    with patch.dict("os.environ", {"NSS_LOG_LEVEL": "DEBUG", "HF_TOKEN": "hf_xxx"}, clear=True):
        issues = check_env(default_config)
    assert not any(i.code == "invalid_log_level" for i in issues)


@pytest.mark.unit
def test_check_env_invalid_log_format(default_config):
    with patch.dict("os.environ", {"NSS_LOG_FORMAT": "xml", "HF_TOKEN": "hf_xxx"}, clear=True):
        issues = check_env(default_config)
    assert any(i.code == "invalid_log_format" for i in issues)


@pytest.mark.unit
def test_check_env_valid_log_format(default_config):
    with patch.dict("os.environ", {"NSS_LOG_FORMAT": "json", "HF_TOKEN": "hf_xxx"}, clear=True):
        issues = check_env(default_config)
    assert not any(i.code == "invalid_log_format" for i in issues)


# ---------------------------------------------------------------------------
# check_config
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_config_auto_unresolved():
    config = SafeSynthesizerParameters()
    config = config.model_copy(update={"training": config.training.model_copy(update={"num_input_records_to_sample": "auto"})})
    issues = check_config(config)
    assert any(i.code == "auto_unresolved" for i in issues)


@pytest.mark.unit
def test_check_config_batch_exceeds_data():
    config = SafeSynthesizerParameters(training=TrainingHyperparams(
        num_input_records_to_sample=10,
        batch_size=8,
        gradient_accumulation_steps=4,
    ))
    issues = check_config(config)
    assert any(i.code == "batch_exceeds_data" for i in issues)


# ---------------------------------------------------------------------------
# check_columns
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_columns_happy_path(sample_df, default_config):
    issues = check_columns(sample_df, default_config)
    assert not any(i.severity == "error" for i in issues)


@pytest.mark.unit
def test_check_columns_missing_group_by(sample_df):
    config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by="nonexistent_col"))
    issues = check_columns(sample_df, config)
    assert any(i.code == "column_not_found" for i in issues)


@pytest.mark.unit
def test_check_columns_nulls_in_group_by():
    df = pd.DataFrame({"grp": [1, None, 3], "val": [10, 20, 30]})
    config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by="grp"))
    issues = check_columns(df, config)
    assert any(i.code == "column_nulls" for i in issues)


@pytest.mark.unit
def test_check_columns_constant_column():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    config = SafeSynthesizerParameters()
    issues = check_columns(df, config)
    assert any(i.code == "constant_column" for i in issues)


@pytest.mark.unit
def test_check_columns_pseudo_column_collision():
    from nemo_safe_synthesizer.defaults import PSEUDO_GROUP_COLUMN

    df = pd.DataFrame({PSEUDO_GROUP_COLUMN: [1, 2, 3], "val": [10, 20, 30]})
    config = SafeSynthesizerParameters()
    issues = check_columns(df, config)
    assert any(i.code == "pseudo_column_collision" for i in issues)


# ---------------------------------------------------------------------------
# check_token_budget
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_token_budget_tokenizer_unavailable(sample_df, default_config):
    metadata = MagicMock()
    metadata.tokenizer = None
    issues = check_token_budget(sample_df, default_config, metadata)
    assert any(i.code == "tokenizer_unavailable" for i in issues)


@pytest.mark.unit
def test_check_token_budget_happy_path(sample_df, default_config):
    metadata = MagicMock()
    metadata.tokenizer.encode.return_value = list(range(50))
    metadata.max_seq_length = 2048
    issues = check_token_budget(sample_df, default_config, metadata)
    assert not any(i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# check_dataset_size
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_dataset_size_happy_path(sample_df, default_config):
    issues = check_dataset_size(sample_df, default_config)
    assert not any(i.severity == "error" for i in issues)


@pytest.mark.unit
def test_check_dataset_size_too_small(tiny_df, default_config):
    issues = check_dataset_size(tiny_df, default_config)
    assert any(i.code == "dataset_too_small" and i.severity == "error" for i in issues)


@pytest.mark.unit
def test_check_dataset_size_under_200(small_df, default_config):
    issues = check_dataset_size(small_df, default_config)
    assert any(i.code == "dataset_too_small" and i.severity == "error" for i in issues)


# ---------------------------------------------------------------------------
# check_training_adequacy
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_training_adequacy_extreme_oversampling(sample_df):
    config = SafeSynthesizerParameters(training=TrainingHyperparams(num_input_records_to_sample=50000))
    issues = check_training_adequacy(sample_df, config)
    assert any(i.code == "extreme_oversampling" for i in issues)


# ---------------------------------------------------------------------------
# check_column_cardinality
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_column_cardinality_high_cardinality():
    df = pd.DataFrame({"unique_col": [f"val_{i}" for i in range(200)], "num_col": range(200)})
    config = SafeSynthesizerParameters()
    issues = check_column_cardinality(df, config)
    assert any(i.code == "high_cardinality" for i in issues)


@pytest.mark.unit
def test_check_column_cardinality_whitelisted():
    df = pd.DataFrame({"grp": [f"val_{i}" for i in range(200)], "num_col": range(200)})
    config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by="grp"))
    issues = check_column_cardinality(df, config)
    assert not any(i.code == "high_cardinality" and "grp" in i.message for i in issues)


# ---------------------------------------------------------------------------
# run_preflight
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_run_preflight_clean_dataset(sample_df, default_config):
    metadata = MagicMock()
    metadata.tokenizer.encode.return_value = list(range(50))
    metadata.max_seq_length = 2048
    with patch("torch.cuda.is_available", return_value=True):
        with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "test", "HF_TOKEN": "hf_xxx"}):
            report = run_preflight(sample_df, default_config, metadata)
    assert len(report.errors) == 0
    assert len(report.checks) >= 7
    check_names = [c.name for c in report.checks]
    assert "timeseries" not in check_names  # default config has is_timeseries=False


@pytest.mark.unit
def test_run_preflight_skips_token_budget_on_column_errors():
    df = pd.DataFrame({"val": [1, 2, 3]})
    config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by="missing_col"))
    metadata = MagicMock()
    with patch("torch.cuda.is_available", return_value=True):
        report = run_preflight(df, config, metadata)
    assert any(i.code == "column_not_found" for i in report.issues)
    check_names = [c.name for c in report.checks]
    assert "token_budget" not in check_names


# ---------------------------------------------------------------------------
# format_preflight_report
# ---------------------------------------------------------------------------


def _make_report(*check_tuples: tuple[str, str, list[PreflightIssue]]) -> PreflightReport:
    """Build a PreflightReport from (name, label, issues) tuples."""
    return PreflightReport(
        checks=[PreflightCheckResult(name=n, label=l, issues=i) for n, l, i in check_tuples],
    )


def _capture_report(**kwargs) -> str:
    """Render format_preflight_report to a plain-text string for assertion."""
    buf = StringIO()
    format_preflight_report(**kwargs, console=Console(file=buf, force_terminal=False, no_color=True))
    return buf.getvalue()


@pytest.mark.unit
def test_format_report_no_issues():
    r = _make_report(("gpu", "GPU resources", []), ("env", "Environment variables", []))
    output = _capture_report(
        report=r, config_path=Path("/tmp/config.yaml"), data_source="/data.csv", artifact_dir=Path("/tmp/artifacts"),
    )
    assert "passed" in output
    assert "GPU resources" in output
    assert "/tmp/artifacts" in output
    assert "resolved config" in output
    assert "safe-synthesizer run" in output
    assert "/data.csv" in output


@pytest.mark.unit
def test_format_report_with_errors():
    issues = [PreflightIssue("no_gpu", "error", "check_gpu_resources", "No GPU")]
    r = _make_report(("gpu", "GPU resources", issues))
    output = _capture_report(
        report=r, config_path=Path("/tmp/config.yaml"), data_source="/data.csv", artifact_dir=Path("/tmp/artifacts"),
    )
    assert "✗" in output
    assert "no_gpu" in output
    assert "GPU resources" in output
    assert "/tmp/artifacts" in output
    assert "safe-synthesizer run" not in output


@pytest.mark.unit
def test_format_report_warnings_only():
    issues = [PreflightIssue("dataset_small", "warning", "check_dataset_size", "Small dataset")]
    r = _make_report(("size", "Dataset size", issues), ("gpu", "GPU resources", []))
    output = _capture_report(
        report=r, config_path=Path("/tmp/config.yaml"), data_source="/data.csv", artifact_dir=Path("/tmp/artifacts"),
    )
    assert "⚠" in output
    assert "GPU resources" in output
    assert "/tmp/artifacts" in output
    assert "safe-synthesizer run" in output


@pytest.mark.unit
def test_format_report_shows_all_checks():
    r = _make_report(
        ("gpu", "GPU resources", []),
        ("env", "Environment variables", [PreflightIssue("hf", "warning", "check_env", "no token")]),
        ("config", "Configuration", []),
    )
    output = _capture_report(report=r)
    assert "GPU resources" in output
    assert "Environment variables" in output
    assert "Configuration" in output


@pytest.mark.unit
def test_format_report_paths_not_truncated():
    long_path = Path("/root/ss-wt-preflight/safe-synthesizer-artifacts/default---financial_transactions/2026-04-14T20:00:30")
    config = long_path / "safe-synthesizer-config.yaml"
    r = _make_report()
    output = _capture_report(
        report=r, config_path=config, data_source="/data.csv", artifact_dir=long_path,
    )
    assert str(long_path) in output
    assert "safe-synthesizer-config.yaml" in output
    assert str(config) in output  # full path still appears in the follow-up command


# ---------------------------------------------------------------------------
# Pydantic validators (time-series config)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_timestamp_interval_negative():
    from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters

    with pytest.raises(ValueError, match="positive"):
        TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=-1)


@pytest.mark.unit
def test_timeseries_without_group_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SafeSynthesizerParameters(
            time_series=TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=5),
        )
    assert any("group_training_examples_by" in str(warning.message) for warning in w)
