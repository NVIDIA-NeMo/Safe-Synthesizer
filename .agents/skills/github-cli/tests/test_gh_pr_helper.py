# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError


def _load_helper() -> Any:
    helper_path = Path(__file__).parents[1] / "scripts" / "gh_pr_helper.py"
    spec = importlib.util.spec_from_file_location("gh_pr_helper", helper_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


helper = _load_helper()


def test_collect_comments_includes_submitted_review_bodies() -> None:
    pr = SimpleNamespace(
        get_review_comments=lambda: [
            SimpleNamespace(raw_data={"id": 10, "body": "Inline finding"})
        ],
        get_reviews=lambda: [
            SimpleNamespace(
                raw_data={
                    "id": 20,
                    "body": "Review summary with an existing finding",
                    "state": "COMMENTED",
                    "user": {"login": "reviewer"},
                }
            )
        ],
    )
    issue = SimpleNamespace(
        get_comments=lambda: [
            SimpleNamespace(raw_data={"id": 30, "body": "Conversation decision"})
        ]
    )

    output = helper._collect_comments_output(715, "owner/repo", pr, issue)

    assert output.inline[0].body == "Inline finding"
    assert output.reviews[0].body == "Review summary with an existing finding"
    assert output.top_level[0].body == "Conversation decision"


def test_submit_approved_review_pins_head_and_returns_comment_urls() -> None:
    calls: dict[str, Any] = {}
    review = SimpleNamespace(
        id=501,
        raw_data={"html_url": "https://github.com/owner/repo/pull/715#pullrequestreview-501"},
    )

    def create_review(**kwargs: Any) -> Any:
        calls["create_review"] = kwargs
        return review

    pr = SimpleNamespace(
        head=SimpleNamespace(sha="a" * 40),
        create_review=create_review,
        get_review_comments=lambda: [
            SimpleNamespace(raw_data={"id": 90, "pull_request_review_id": 500}),
            SimpleNamespace(
                raw_data={
                    "id": 91,
                    "pull_request_review_id": 501,
                    "path": "src/example.py",
                    "line": 42,
                    "html_url": "https://github.com/owner/repo/pull/715#discussion_r91",
                }
            ),
        ],
    )

    def get_commit(sha: str) -> Any:
        calls["get_commit"] = sha
        return SimpleNamespace(sha=sha)

    repo = SimpleNamespace(get_commit=get_commit)
    submission = helper.ReviewSubmission(
        head_sha="A" * 40,
        body="Approved summary",
        comments=[
            {
                "path": "src/example.py",
                "line": 42,
                "side": "RIGHT",
                "body": "Approved finding",
            }
        ],
    )

    output = helper._submit_approved_review(repo, pr, "owner/repo", 715, submission)

    assert calls["get_commit"] == "a" * 40
    assert calls["create_review"] == {
        "commit": SimpleNamespace(sha="a" * 40),
        "body": "Approved summary",
        "event": "COMMENT",
        "comments": [
            {
                "path": "src/example.py",
                "body": "Approved finding",
                "line": 42,
                "side": "RIGHT",
            }
        ],
    }
    assert output.review_url.endswith("#pullrequestreview-501")
    assert [comment.html_url for comment in output.comments] == [
        "https://github.com/owner/repo/pull/715#discussion_r91"
    ]


def test_submit_approved_review_rejects_a_stale_head_before_posting() -> None:
    pr = SimpleNamespace(head=SimpleNamespace(sha="b" * 40))
    repo = SimpleNamespace()
    submission = helper.ReviewSubmission(
        head_sha="a" * 40,
        comments=[{"path": "src/example.py", "line": 42, "body": "Approved finding"}],
    )

    with pytest.raises(ValueError, match="PR head changed"):
        helper._submit_approved_review(repo, pr, "owner/repo", 715, submission)


def test_review_comment_range_requires_both_start_fields() -> None:
    with pytest.raises(ValidationError, match="start_line and start_side must be set together"):
        helper.ReviewDraftComment(
            path="src/example.py",
            line=42,
            body="Approved finding",
            start_line=40,
        )
