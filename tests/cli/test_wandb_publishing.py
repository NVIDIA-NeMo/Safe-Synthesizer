# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused W&B evaluation publishing contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_safe_synthesizer.cli import wandb_setup
from nemo_safe_synthesizer.config.external_results import SafeSynthesizerSummary, SafeSynthesizerTiming


class FakeSummary:
    """Dictionary-like W&B summary recording update calls."""

    def __init__(self) -> None:
        self.values: dict[str, object] = {}
        self.update_calls: list[dict[str, object]] = []

    def get(self, key: str) -> object | None:
        return self.values.get(key)

    def update(self, values: dict[str, object]) -> None:
        self.update_calls.append(values)
        self.values.update(values)

    def __contains__(self, key: str) -> bool:
        return key in self.values

    def __getitem__(self, key: str) -> object:
        return self.values[key]


class FakeArtifact:
    """Hermetic replacement for W&B artifact staging."""

    def __init__(self, name: str, type: str) -> None:  # noqa: A002 - W&B API spelling
        self.name = name
        self.type = type
        self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, name: str) -> None:
        self.files.append((path, name))


def _summary() -> SafeSynthesizerSummary:
    return SafeSynthesizerSummary(
        timing=SafeSynthesizerTiming(),
        synthetic_data_quality_score=7.5,
        data_privacy_score=8.5,
    )


def _workdir(tmp_path: Path, *, report: bool = True, metrics: bool = True) -> MagicMock:
    workdir = MagicMock()
    workdir.evaluation_report = tmp_path / "evaluation_report.html"
    workdir.evaluation_metrics = tmp_path / "evaluation_metrics.json"
    if report:
        workdir.evaluation_report.write_text("<h1>report</h1>", encoding="utf-8")
    if metrics:
        workdir.evaluation_metrics.write_text('{"ok": true}', encoding="utf-8")
    return workdir


def _run(run_id: str = "run-123") -> MagicMock:
    run = MagicMock(id=run_id)
    run.summary = FakeSummary()
    return run


def test_publish_opt_in_has_hermetic_scorecard_media_artifact_and_sha(tmp_path: Path) -> None:
    """Opt-in writes only evaluation media history and records the exact artifact."""
    workdir, run = _workdir(tmp_path), _run()
    fake_artifacts: list[FakeArtifact] = []
    html = MagicMock()
    table = MagicMock()

    def make_artifact(name: str, type: str) -> FakeArtifact:  # noqa: A002 - W&B API spelling
        artifact = FakeArtifact(name, type)
        fake_artifacts.append(artifact)
        return artifact

    with (
        patch.object(wandb_setup.wandb, "run", run),
        patch.object(wandb_setup.wandb, "Table", return_value=table) as mock_table,
        patch.object(wandb_setup.wandb, "Html", return_value=html) as mock_html,
        patch.object(wandb_setup.wandb, "Artifact", side_effect=make_artifact),
    ):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)

    assert [set(call.args[0]) for call in run.log.call_args_list] == [{"evaluation/scorecard"}, {"evaluation/report"}]
    assert mock_table.call_args.kwargs == {
        "columns": ["metric", "value"],
        "data": [[key, value] for key, value in _summary()._wandb_metrics().items() if key.startswith("eval/")],
    }
    mock_html.assert_called_once_with("<h1>report</h1>", inject=False)
    assert [(artifact.name, artifact.type, artifact.files) for artifact in fake_artifacts] == [
        (
            "safe-synthesizer-evaluation-report-run-123",
            "evaluation-report",
            [
                (str(workdir.evaluation_report), "evaluation_report.html"),
                (str(workdir.evaluation_metrics), "evaluation_metrics.json"),
            ],
        )
    ]
    run.log_artifact.assert_called_once_with(fake_artifacts[0])
    assert run.summary["evaluation/report_sha256"] == hashlib.sha256(workdir.evaluation_report.read_bytes()).hexdigest()


def test_identical_resume_in_a_fresh_process_is_suppressed(tmp_path: Path) -> None:
    """The persisted W&B marker, not process memory, suppresses an identical repeat."""
    workdir, run = _workdir(tmp_path), _run()
    with patch.object(wandb_setup.wandb, "run", run):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=False)
        run.log.reset_mock()  # Simulate a fresh process retaining only W&B summary state.
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=False)
    run.log.assert_not_called()


def test_identical_opt_in_resume_does_not_duplicate_media_or_artifact(tmp_path: Path) -> None:
    """A completed opt-in publication is suppressed by its persisted marker."""
    workdir, run = _workdir(tmp_path), _run()
    with (
        patch.object(wandb_setup.wandb, "run", run),
        patch.object(wandb_setup.wandb, "Table", return_value=MagicMock()),
        patch.object(wandb_setup.wandb, "Html", return_value=MagicMock()),
        patch.object(wandb_setup.wandb, "Artifact", side_effect=FakeArtifact),
    ):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
        run.log.reset_mock()
        run.log_artifact.reset_mock()
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)

    run.log.assert_not_called()
    run.log_artifact.assert_not_called()


def test_changed_evaluation_output_publishes_a_new_report(tmp_path: Path) -> None:
    """A changed score keeps resumed CLI work observable."""
    workdir, run = _workdir(tmp_path), _run()
    changed = _summary().model_copy(update={"synthetic_data_quality_score": 7.6})
    with patch.object(wandb_setup.wandb, "run", run):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=False)
        wandb_setup.publish_evaluation_report(workdir, changed, upload_report=False)
    assert run.log.call_count == 2


def test_opt_out_then_opt_in_publishes_report_after_scorecard(tmp_path: Path) -> None:
    """Opting in later changes the fingerprint and permits report publishing."""
    workdir, run = _workdir(tmp_path), _run()
    with (
        patch.object(wandb_setup.wandb, "run", run),
        patch.object(wandb_setup.wandb, "Artifact", side_effect=FakeArtifact),
    ):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=False)
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
    assert [set(call.args[0]) for call in run.log.call_args_list] == [
        {"evaluation/scorecard"},
        {"evaluation/report"},
    ]


def test_failure_then_retry_is_not_suppressed(tmp_path: Path) -> None:
    """A transient W&B failure leaves no completion marker and therefore retries."""
    workdir, run = _workdir(tmp_path), _run()
    run.log.side_effect = [RuntimeError("down"), None]
    with patch.object(wandb_setup.wandb, "run", run):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=False)
        assert wandb_setup._EVALUATION_PUBLISHING_FINGERPRINT_KEY not in run.summary
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=False)
    assert run.log.call_count == 2


def test_artifact_failure_then_retry_does_not_republish_successful_media(tmp_path: Path) -> None:
    """Retry only the failed artifact operation after scorecard and report succeed."""
    workdir, run = _workdir(tmp_path), _run()
    artifact_attempts = 0

    def make_artifact(name: str, type: str) -> FakeArtifact:  # noqa: A002 - W&B API spelling
        nonlocal artifact_attempts
        artifact_attempts += 1
        if artifact_attempts == 1:
            raise RuntimeError("down")
        return FakeArtifact(name, type)

    with (
        patch.object(wandb_setup.wandb, "run", run),
        patch.object(wandb_setup.wandb, "Artifact", side_effect=make_artifact),
    ):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)

    assert [set(call.args[0]) for call in run.log.call_args_list] == [
        {"evaluation/scorecard"},
        {"evaluation/report"},
    ]
    run.log_artifact.assert_called_once()
    assert run.summary["evaluation/report_uploaded_post_run"] is True
    assert run.summary["evaluation/report_sha256"] == hashlib.sha256(workdir.evaluation_report.read_bytes()).hexdigest()


def test_missing_report_and_metrics_warn_without_raising(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Absent opted-in local files are non-fatal and explain both omissions."""
    workdir, run = _workdir(tmp_path, report=False, metrics=False), _run()
    with patch.object(wandb_setup.wandb, "run", run):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
    assert "report upload requested but file is missing" in caplog.text
    assert "metrics upload requested but file is missing" in caplog.text


def test_missing_local_report_on_resume_preserves_uploaded_summary(tmp_path: Path) -> None:
    """A missing local file does not erase the durable state of an earlier upload."""
    workdir, run = _workdir(tmp_path), _run()
    with (
        patch.object(wandb_setup.wandb, "run", run),
        patch.object(wandb_setup.wandb, "Artifact", side_effect=FakeArtifact),
    ):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
        uploaded_sha = run.summary["evaluation/report_sha256"]
        workdir.evaluation_report.unlink()
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)

    assert run.summary["evaluation/report_uploaded_post_run"] is True
    assert run.summary["evaluation/report_sha256"] == uploaded_sha


def test_opt_out_after_upload_preserves_uploaded_summary(tmp_path: Path) -> None:
    """Opting out later does not erase the durable state of an earlier upload."""
    workdir, run = _workdir(tmp_path), _run()
    with (
        patch.object(wandb_setup.wandb, "run", run),
        patch.object(wandb_setup.wandb, "Artifact", side_effect=FakeArtifact),
    ):
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
        uploaded_sha = run.summary["evaluation/report_sha256"]
        run.log_artifact.reset_mock()
        wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=False)

    run.log_artifact.assert_not_called()
    assert run.summary["evaluation/report_uploaded_post_run"] is True
    assert run.summary["evaluation/report_sha256"] == uploaded_sha


@pytest.mark.parametrize("failure", ["media", "artifact", "summary"])
def test_publishing_operation_failures_never_escape(tmp_path: Path, failure: str) -> None:
    """Media, artifact, and final-summary W&B failures preserve synthesis success."""
    workdir, run = _workdir(tmp_path), _run()
    if failure == "summary":
        run.summary.update = MagicMock(side_effect=RuntimeError("summary down"))

    with patch.object(wandb_setup.wandb, "run", run):
        if failure == "media":
            with patch.object(wandb_setup.wandb, "Html", side_effect=RuntimeError("media down")) as mock_html:
                wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
            mock_html.assert_called_once_with("<h1>report</h1>", inject=False)
        elif failure == "artifact":
            with patch.object(
                wandb_setup.wandb, "Artifact", side_effect=RuntimeError("artifact down")
            ) as mock_artifact:
                wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
            mock_artifact.assert_called_once_with(
                "safe-synthesizer-evaluation-report-run-123",
                type="evaluation-report",
            )
        else:
            with patch.object(wandb_setup.wandb, "Artifact", side_effect=FakeArtifact):
                wandb_setup.publish_evaluation_report(workdir, _summary(), upload_report=True)
            run.summary.update.assert_called()
