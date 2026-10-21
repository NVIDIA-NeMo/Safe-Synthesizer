# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).parents[1] / ".github/workflows/pr-linked-issue.yml"


def _step(name: str, job: str = "check") -> dict:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    return next(step for step in workflow["jobs"][job]["steps"] if step["name"] == name)


def _run_step(
    tmp_path: Path, name: str, mock_gh: str, *, job: str = "check", **variables: str
) -> tuple[subprocess.CompletedProcess[str], str]:
    gh = tmp_path / "gh"
    gh.write_text("#!/bin/sh\n" + mock_gh)
    gh.chmod(0o755)
    output = tmp_path / "github-output"
    output.write_text("")
    script = _step(name, job)["run"].replace("${{ github.repository }}", "NVIDIA-NeMo/Safe-Synthesizer")
    env = (
        os.environ
        | {
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "REPO": "NVIDIA-NeMo/Safe-Synthesizer",
            "PR_NUMBER": "42",
        }
        | variables
    )
    result = subprocess.run(["bash", "-e", "-c", script], env=env, capture_output=True, text=True)
    return result, output.read_text()


@pytest.mark.parametrize(
    ("response", "expected_exit", "expected_output"),
    [
        ('{"state":"open","labels":[{"name":"triaged"}]}', 0, "issue_open=true\nis_triaged=true"),
        ('{"state":"closed","labels":[{"name":"triaged"}]}', 0, "issue_open=false\nis_triaged=false"),
        ('{"state":"open","labels":[]}', 0, "issue_open=true\nis_triaged=false"),
        ('{"pull_request":{},"state":"open","labels":[{"name":"triaged"}]}', 0, "issue_exists=false"),
        ("HTTP 404", 0, "issue_exists=false"),
        ("HTTP 503", 1, ""),
    ],
)
def test_issue_validation_distinguishes_policy_failure_from_api_outage(
    tmp_path: Path, response: str, expected_exit: int, expected_output: str
) -> None:
    mock = (
        'if [ "$SCENARIO" = "HTTP 404" ] || [ "$SCENARIO" = "HTTP 503" ]; then\n'
        '  echo "gh: $SCENARIO" >&2; exit 1\n'
        "fi\n"
        'printf "%s\\n" "$SCENARIO"\n'
    )
    result, output = _run_step(
        tmp_path, "Validate issue is open and triaged", mock, SCENARIO=response, ISSUE_NUMS="123"
    )
    assert result.returncode == expected_exit, result.stderr
    assert expected_output in output


@pytest.mark.parametrize(
    ("response", "expected_exit", "expected_output"),
    [
        ("write", 0, "is_collaborator=true"),
        ("admin", 0, "is_collaborator=true"),
        ("read", 0, "is_collaborator=false"),
        ("HTTP 404", 0, "is_collaborator=false"),
        ("HTTP 503", 1, ""),
    ],
)
def test_author_permission_does_not_treat_api_outage_as_external(
    tmp_path: Path, response: str, expected_exit: int, expected_output: str
) -> None:
    mock = 'case "$SCENARIO" in\n  "HTTP "*) echo "gh: $SCENARIO" >&2; exit 1 ;;\n  *) echo "$SCENARIO" ;;\nesac\n'
    result, output = _run_step(tmp_path, "Check author permissions", mock, SCENARIO=response, PR_AUTHOR="contributor")
    assert result.returncode == expected_exit, result.stderr
    assert expected_output in output


def test_dependabot_is_exempt_without_a_permission_lookup(tmp_path: Path) -> None:
    result, output = _run_step(tmp_path, "Check author permissions", "exit 1\n", PR_AUTHOR="dependabot[bot]")
    assert result.returncode == 0, result.stderr
    assert "is_collaborator=true" in output


@pytest.mark.parametrize(
    ("body", "expected_issue"),
    [
        ("Fixes #123", "123"),
        ("<!-- Fixes #123 -->", ""),
        ("<!-- Fixes #123 -->\nResolves #456", "456"),
        ("<!--\nFixes #123\n-->\ncLoSeS #456", "456"),
        ("No linked issue required: docs cleanup", ""),
        ("", ""),
    ],
)
def test_issue_parser_reads_current_pr_body(tmp_path: Path, body: str, expected_issue: str) -> None:
    mock = 'printf "%s\\n" "$CURRENT_BODY"\n'
    result, output = _run_step(
        tmp_path, "Parse issue reference from PR body", mock, CURRENT_BODY=body, PR_BODY="Fixes #999"
    )
    assert result.returncode == 0, result.stderr
    assert f"issue_num={expected_issue}\n" in output


def test_issue_parser_keeps_all_distinct_references(tmp_path: Path) -> None:
    result, output = _run_step(
        tmp_path,
        "Parse issue reference from PR body",
        'printf "%s\\n" "$CURRENT_BODY"\n',
        CURRENT_BODY="Fixes #123\nCloses #456\nResolves #123\n<!-- Fixes #789 -->",
    )
    assert result.returncode == 0, result.stderr
    assert "issue_nums=123 456\n" in output


def test_second_valid_issue_prevents_close(tmp_path: Path) -> None:
    mock = (
        'if [ "$1" = "pr" ] && [ "$2" = "view" ]; then\n'
        '  printf "%s\\n" \'{"state":"OPEN","labels":[],"body":"Fixes #123\\nResolves #456"}\'; exit 0\n'
        "fi\n"
        'if [ "$1" = "pr" ] && [ "$2" = "close" ]; then echo closed > "$CALL_LOG"; exit 0; fi\n'
        'if [ "$1" = "api" ] && [ "$2" = "-X" ]; then exit 0; fi\n'
        'case "$*" in\n'
        '  *"/123"*) printf "%s\\n" \'{"state":"closed","labels":[]}\' ;;\n'
        '  *"/456"*) printf "%s\\n" \'{"state":"open","labels":[{"name":"triaged"}]}\' ;;\n'
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    result, output = _run_step(tmp_path, "Validate issue is open and triaged", mock, ISSUE_NUMS="123 456")
    assert result.returncode == 0, result.stderr
    assert "is_triaged=true" in output

    call_log = tmp_path / "close-call"
    result, output = _run_step(
        tmp_path,
        "Close new or reopened PR without an open, triaged issue",
        mock,
        CALL_LOG=str(call_log),
        COMMENT_ID="77",
    )
    assert result.returncode == 0, result.stderr
    assert "current_valid=true" in output
    assert not call_log.exists()


def test_issue_changes_recheck_passing_prs_without_closing_them() -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    assert set(workflow["on"]["issues"]["types"]) == {"labeled", "unlabeled", "closed", "reopened"}
    retrigger = workflow["jobs"]["retrigger"]
    assert "github.event.action == 'closed'" in retrigger["if"]
    assert "github.event.action == 'reopened'" in retrigger["if"]
    rerun = next(step for step in retrigger["steps"] if step["name"] == "Rerun linked issue checks")
    assert '[ "$run_conclusion" = "success" ] && [ "$ISSUE_ACTION" = "labeled" ]' in rerun["run"]
    assert "github.run_attempt == '1'" in _step("Close new or reopened PR without an open, triaged issue")["if"]


@pytest.mark.parametrize(("action", "should_rerun"), [("closed", True), ("unlabeled", True), ("labeled", False)])
def test_issue_change_reruns_previously_passing_check(tmp_path: Path, action: str, should_rerun: bool) -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    step = next(step for step in workflow["jobs"]["retrigger"]["steps"] if step["name"] == "Rerun linked issue checks")
    gh = tmp_path / "gh"
    gh.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = "pr" ]; then echo abc; exit 0; fi\n'
        'if [ "$2" = "--paginate" ]; then\n'
        '  printf "%s\\n" \'[{"workflow_runs":[{"id":10,"event":"pull_request_target",'
        '"head_sha":"abc","status":"completed","conclusion":"success",'
        '"created_at":"2026-01-01T00:00:00Z"}]}]\'; exit 0\n'
        "fi\n"
        'if [ "$2" = "-X" ]; then echo "$*" > "$CALL_LOG"; exit 0; fi\n'
        "exit 1\n"
    )
    gh.chmod(0o755)
    call_log = tmp_path / "rerun-call"
    env = os.environ | {
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "REPO": "NVIDIA-NeMo/Safe-Synthesizer",
        "PR_NUMBERS": "42",
        "ISSUE_ACTION": action,
        "CALL_LOG": str(call_log),
    }
    result = subprocess.run(["bash", "-e", "-c", step["run"]], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert call_log.exists() == should_rerun


def test_issue_change_selects_visible_references_only(tmp_path: Path) -> None:
    prs = [
        {"number": 1, "body": "Fixes #123"},
        {"number": 2, "body": "<!-- Fixes #123 -->"},
        {"number": 3, "body": "Fixes #1234"},
        {"number": 4, "body": "<!-- Fixes #456 -->\nResolves #123"},
    ]
    result, output = _run_step(
        tmp_path,
        "Find PRs referencing this issue",
        'printf "%s\\n" "$PRS_JSON"\n',
        job="retrigger",
        ISSUE_NUMBER="123",
        PRS_JSON=json.dumps(prs),
    )
    assert result.returncode == 0, result.stderr
    assert "prs=1 4 \n" in output


@pytest.mark.parametrize(
    ("collaborator", "issues", "triaged", "existing_comment", "status", "operation"),
    [
        ("false", "", "", "", "fail", "create"),
        ("false", "123", "false", "77", "fail", "PATCH"),
        ("false", "123", "true", "77", "pass", "DELETE"),
        ("true", "", "", "77", "pass", "DELETE"),
    ],
)
def test_policy_comment_explains_failure_and_clears_on_success(
    tmp_path: Path,
    collaborator: str,
    issues: str,
    triaged: str,
    existing_comment: str,
    status: str,
    operation: str,
) -> None:
    call_log = tmp_path / "comment-call"
    body_log = tmp_path / "comment-body"
    mock = (
        'if [ "$2" = "--paginate" ]; then printf "%s\\n" "$EXISTING_COMMENT"; exit 0; fi\n'
        'if [ "$2" = "-X" ]; then echo "$3" > "$CALL_LOG"; else echo create > "$CALL_LOG"; fi\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in body=*) printf "%s\\n" "${arg#body=}" > "$BODY_LOG" ;; esac\n'
        "done\n"
        "echo 77\n"
    )
    result, output = _run_step(
        tmp_path,
        "Build comment body and post result",
        mock,
        IS_COLLABORATOR=collaborator,
        ISSUE_NUMS=issues,
        IS_TRIAGED=triaged,
        EXISTING_COMMENT=existing_comment,
        CALL_LOG=str(call_log),
        BODY_LOG=str(body_log),
    )
    assert result.returncode == 0, result.stderr
    assert f"status={status}\n" in output
    assert call_log.read_text().strip() == operation
    if status == "fail":
        assert "comment_id=77\n" in output
        assert "reopen" in body_log.read_text()
        assert "triaged" in body_log.read_text()


@pytest.mark.parametrize(
    ("state", "labels", "body", "issue_response", "expected_exit", "should_close"),
    [
        ("OPEN", [], "", "", 0, True),
        ("OPEN", [], "Fixes #123", '{"state":"open","labels":[]}', 0, True),
        ("OPEN", [], "Fixes #123", '{"state":"open","labels":[{"name":"triaged"}]}', 0, False),
        ("OPEN", [], "<!-- Fixes #123 -->", '{"state":"open","labels":[{"name":"triaged"}]}', 0, True),
        ("OPEN", ["keep-open"], "", "", 0, False),
        ("CLOSED", [], "", "", 0, False),
        ("OPEN", [], "Fixes #123", "HTTP 503", 1, False),
    ],
)
def test_close_step_rechecks_current_pr(
    tmp_path: Path,
    state: str,
    labels: list[str],
    body: str,
    issue_response: str,
    expected_exit: int,
    should_close: bool,
) -> None:
    step = _step("Close new or reopened PR without an open, triaged issue")
    assert "github.event.action == 'opened'" in step["if"]
    assert "github.event.action == 'reopened'" in step["if"]
    assert "github.run_attempt == '1'" in step["if"]

    call_log = tmp_path / "close-call"
    mock = (
        'if [ "$1" = "pr" ] && [ "$2" = "view" ]; then printf "%s\\n" "$PR_JSON"; exit 0; fi\n'
        'if [ "$1" = "pr" ] && [ "$2" = "close" ]; then echo "$*" > "$CALL_LOG"; exit 0; fi\n'
        'if [ "$1" = "api" ] && [ "$2" = "-X" ]; then exit 0; fi\n'
        'if [ "$ISSUE_RESPONSE" = "HTTP 503" ]; then echo "gh: HTTP 503" >&2; exit 1; fi\n'
        'printf "%s\\n" "$ISSUE_RESPONSE"\n'
    )
    result, output = _run_step(
        tmp_path,
        "Close new or reopened PR without an open, triaged issue",
        mock,
        PR_JSON=json.dumps({"state": state, "labels": [{"name": label} for label in labels], "body": body}),
        ISSUE_RESPONSE=issue_response,
        CALL_LOG=str(call_log),
        COMMENT_ID="77",
    )
    assert result.returncode == expected_exit, result.stderr
    assert call_log.exists() == should_close
    if should_close:
        assert call_log.read_text().strip() == "pr close 42 --repo NVIDIA-NeMo/Safe-Synthesizer"
    if body == "Fixes #123" and '"triaged"' in issue_response:
        assert "current_valid=true" in output


def test_valid_pr_passes_when_stale_comment_cannot_be_deleted(tmp_path: Path) -> None:
    mock = (
        'if [ "$1" = "pr" ] && [ "$2" = "view" ]; then\n'
        '  printf "%s\\n" \'{"state":"OPEN","labels":[],"body":"Fixes #123"}\'; exit 0\n'
        "fi\n"
        'if [ "$1" = "api" ] && [ "$2" = "-X" ]; then exit 1; fi\n'
        'if [ "$1" = "api" ]; then\n'
        '  printf "%s\\n" \'{"state":"open","labels":[{"name":"triaged"}]}\'; exit 0\n'
        "fi\n"
        "exit 1\n"
    )
    result, output = _run_step(
        tmp_path, "Close new or reopened PR without an open, triaged issue", mock, COMMENT_ID="77"
    )
    assert result.returncode == 0, result.stderr
    assert "current_valid=true" in output
    assert "Could not remove the outdated linked-issue comment" in result.stdout
