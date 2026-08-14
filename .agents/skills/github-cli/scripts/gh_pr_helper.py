#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# gh-pr-helper: fetch PR discussion, submit approved reviews, and reply via GitHub API.
# Run: uv run --script scripts/gh_pr_helper.py -- [args]
# Auth: GITHUB_TOKEN env var (or --token). Optional fallback: gh auth token.
# Requires network; in Agent use required_permissions: ["all"].
# See: references/workflows.md § Fetch and Address Review Comments.
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "typer>=0.12",
#     "PyGithub>=2.4",
#     "pydantic>=2.0",
# ]
# ///
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal, Self

import typer  # provided by PEP 723 deps when run via uv run --script
from github import Auth, Github
from github.Issue import Issue
from github.PullRequest import PullRequest
from github.Repository import Repository
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

app = typer.Typer(
    name="gh-pr-helper",
    help="Fetch PR discussion, submit approved reviews, and reply to inline comments (GitHub API).",
)


# --- Output models: single JSON object per command ---


class Author(BaseModel):
    login: str
    avatar_url: str | None = None


class ReviewComment(BaseModel):
    """A review comment on a specific line(s) of the PR diff (inline comment)."""

    id: int
    body: str = ""
    path: str | None = None
    line: int | None = None
    side: str | None = None
    start_line: int | None = None
    start_side: str | None = None
    user: Author | None = None
    created_at: str = ""
    updated_at: str | None = None
    html_url: str | None = None
    in_reply_to_id: int | None = None
    pull_request_url: str | None = None


class IssueComment(BaseModel):
    """A comment on the PR conversation (issue comment)."""

    id: int
    body: str = ""
    user: Author | None = None
    created_at: str = ""
    updated_at: str | None = None
    html_url: str | None = None


class SubmittedReview(BaseModel):
    """A submitted PR review whose body can record findings or decisions."""

    id: int
    body: str = ""
    state: str = ""
    user: Author | None = None
    submitted_at: str | None = None
    html_url: str | None = None
    commit_id: str | None = None


class PRCommentsOutput(BaseModel):
    """Single JSON output for all three PR discussion surfaces."""

    pr_number: int
    repo: str
    inline: list[ReviewComment] = Field(default_factory=list)
    reviews: list[SubmittedReview] = Field(default_factory=list)
    top_level: list[IssueComment] = Field(default_factory=list)


class ReplyOutput(BaseModel):
    """Single JSON output for reply command after posting a reply."""

    comment_id: int
    body: str
    success: bool = True


class ReviewDraftComment(BaseModel):
    """One approved inline finding to submit in a PR review."""

    path: str
    body: str
    line: int = Field(gt=0)
    side: Literal["LEFT", "RIGHT"] = "RIGHT"
    start_line: int | None = Field(default=None, gt=0)
    start_side: Literal["LEFT", "RIGHT"] | None = None

    @field_validator("path", "body")
    @classmethod
    def _reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _validate_multiline_range(self) -> Self:
        if (self.start_line is None) != (self.start_side is None):
            raise ValueError("start_line and start_side must be set together")
        if self.start_line is not None and self.start_line > self.line:
            raise ValueError("start_line must not be after line")
        return self

    def github_payload(self) -> dict[str, object]:
        """Return the line-comment fields accepted by GitHub's review API."""
        return self.model_dump(exclude_none=True)


class ReviewSubmission(BaseModel):
    """Exact approved review artifact pinned to one PR head."""

    head_sha: str = Field(pattern=r"^[0-9a-fA-F]{40}$")
    body: str = ""
    comments: list[ReviewDraftComment] = Field(min_length=1)

    @field_validator("head_sha")
    @classmethod
    def _normalize_head_sha(cls, value: str) -> str:
        return value.lower()


class PostedReviewComment(BaseModel):
    """Direct link to one submitted inline finding."""

    id: int
    path: str | None = None
    line: int | None = None
    html_url: str


class SubmitReviewOutput(BaseModel):
    """Single JSON output after submitting an approved review."""

    pr_number: int
    repo: str
    head_sha: str
    review_id: int
    review_url: str
    comments: list[PostedReviewComment] = Field(default_factory=list)


def _get_token(token: str | None) -> str:
    if token:
        return token
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t
    out = subprocess.run(
        ["gh", "auth", "token"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode == 0 and out.stdout.strip():
        return out.stdout.strip()
    typer.echo("Set GITHUB_TOKEN or pass --token (or use gh auth login).", err=True)
    raise typer.Exit(1)


def _get_repo_from_git() -> str | None:
    out = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    url = out.stdout.strip()
    # https://github.com/owner/repo.git or git@github.com:owner/repo.git
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}"
    return None


def _get_branch() -> str | None:
    out = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None
    return out.stdout.strip()


def _collect_comments_output(
    pr_number: int,
    repo_name: str,
    pr: PullRequest,
    issue: Issue,
) -> PRCommentsOutput:
    """Collect every PR discussion surface used by duplicate detection."""
    inline = [ReviewComment.model_validate(comment.raw_data) for comment in pr.get_review_comments()]
    reviews = [SubmittedReview.model_validate(review.raw_data) for review in pr.get_reviews()]
    top_level = [IssueComment.model_validate(comment.raw_data) for comment in issue.get_comments()]
    return PRCommentsOutput(
        pr_number=pr_number,
        repo=repo_name,
        inline=inline,
        reviews=reviews,
        top_level=top_level,
    )


def _read_review_submission(review_file: Path) -> ReviewSubmission:
    """Load and validate the exact review artifact approved by the user."""
    return ReviewSubmission.model_validate(json.loads(review_file.read_text()))


def _submit_approved_review(
    repo_obj: Repository,
    pr: PullRequest,
    repo_name: str,
    pr_number: int,
    submission: ReviewSubmission,
) -> SubmitReviewOutput:
    """Submit one approved review after checking that its head is still current."""
    current_head = pr.head.sha.lower()
    if current_head != submission.head_sha:
        msg = (
            f"PR head changed: approved {submission.head_sha}, current {current_head}. "
            "Re-verify the findings and request approval again."
        )
        raise ValueError(msg)

    review = pr.create_review(
        commit=repo_obj.get_commit(submission.head_sha),
        body=submission.body,
        event="COMMENT",
        comments=[comment.github_payload() for comment in submission.comments],
    )
    review_url = review.raw_data.get("html_url") or (
        f"https://github.com/{repo_name}/pull/{pr_number}#pullrequestreview-{review.id}"
    )
    posted_comments = []
    for comment in pr.get_review_comments():
        data = comment.raw_data
        if data.get("pull_request_review_id") != review.id:
            continue
        posted_comments.append(
            PostedReviewComment(
                id=data["id"],
                path=data.get("path"),
                line=data.get("line"),
                html_url=data.get("html_url")
                or f"https://github.com/{repo_name}/pull/{pr_number}#discussion_r{data['id']}",
            )
        )

    return SubmitReviewOutput(
        pr_number=pr_number,
        repo=repo_name,
        head_sha=submission.head_sha,
        review_id=review.id,
        review_url=review_url,
        comments=posted_comments,
    )


@app.command()
def comments(
    pr_number: str | None = typer.Argument(None, help="PR number (default: from current branch)"),
    repo: str | None = typer.Option(None, "--repo", "-r", help="OWNER/REPO (default: from git remote)"),
    token: str | None = typer.Option(
        None, "--token", "-t", help="GitHub token (default: GITHUB_TOKEN or gh auth token)"
    ),
) -> None:
    """Fetch inline comments, submitted reviews, and conversation comments as JSON."""
    gh = Github(auth=Auth.Token(_get_token(token)))
    rep = repo or _get_repo_from_git()
    if not rep:
        typer.echo("Could not determine repo. Pass --repo OWNER/REPO.", err=True)
        raise typer.Exit(1)
    repo_obj = gh.get_repo(rep)
    owner = repo_obj.owner.login

    if not pr_number:
        branch = _get_branch()
        if not branch:
            typer.echo("No PR number and could not get current branch. Pass PR number.", err=True)
            raise typer.Exit(1)
        head = f"{owner}:{branch}"
        prs = list(repo_obj.get_pulls(state="open", head=head))
        if not prs:
            typer.echo(f"No open PR found for head {head}. Pass PR number.", err=True)
            raise typer.Exit(1)
        pr_number = str(prs[0].number)

    try:
        resolved_pr_number = int(pr_number)
    except ValueError as exc:
        typer.echo(f"Invalid PR number: {pr_number}", err=True)
        raise typer.Exit(1) from exc
    out = _collect_comments_output(
        pr_number=resolved_pr_number,
        repo_name=rep,
        pr=repo_obj.get_pull(resolved_pr_number),
        issue=repo_obj.get_issue(resolved_pr_number),
    )
    # Single JSON object, written once (no sys.stdout / multiple writes).
    print(out.model_dump_json(indent=2))  # noqa: T201


@app.command("submit-review")
def submit_review(
    pr_number: int = typer.Argument(..., help="PR number"),
    review_file: Path = typer.Option(
        ...,
        "--review-file",
        "-f",
        exists=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
        help="Approved review artifact in JSON format",
    ),
    repo: str | None = typer.Option(None, "--repo", "-r", help="OWNER/REPO (default: from git remote)"),
    token: str | None = typer.Option(
        None, "--token", "-t", help="GitHub token (default: GITHUB_TOKEN or gh auth token)"
    ),
) -> None:
    """Submit an approved inline review pinned to a verified PR head."""
    try:
        submission = _read_review_submission(review_file)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        typer.echo(f"Invalid review artifact: {exc}", err=True)
        raise typer.Exit(1) from exc

    gh = Github(auth=Auth.Token(_get_token(token)))
    rep = repo or _get_repo_from_git()
    if not rep:
        typer.echo("Could not determine repo. Pass --repo OWNER/REPO.", err=True)
        raise typer.Exit(1)
    repo_obj = gh.get_repo(rep)
    try:
        result = _submit_approved_review(
            repo_obj=repo_obj,
            pr=repo_obj.get_pull(pr_number),
            repo_name=rep,
            pr_number=pr_number,
            submission=submission,
        )
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    print(result.model_dump_json(indent=2))  # noqa: T201


@app.command()
def reply(
    comment_id: str = typer.Argument(..., help="Comment ID (from pulls/PR/comments id field)"),
    body: str | None = typer.Argument(None, help="Reply text (use --reply-file for pipe/file)"),
    reply_file: Path | None = typer.Option(
        None,
        "--reply-file",
        "-f",
        path_type=Path,
        help="Read reply body from file (use - for stdin)",
    ),
    repo: str | None = typer.Option(None, "--repo", "-r", help="OWNER/REPO (default: from git remote)"),
    token: str | None = typer.Option(
        None, "--token", "-t", help="GitHub token (default: GITHUB_TOKEN or gh auth token)"
    ),
) -> None:
    """Post a reply to an inline review comment."""
    if reply_file is not None:
        if str(reply_file) == "-":
            reply_text = sys.stdin.read()
        else:
            reply_text = reply_file.read_text()
    elif body is not None:
        reply_text = body
    else:
        typer.echo("Give either reply text or --reply-file.", err=True)
        raise typer.Exit(1)
    if not reply_text.strip():
        typer.echo("Reply body is empty.", err=True)
        raise typer.Exit(1)

    gh = Github(auth=Auth.Token(_get_token(token)))
    rep = repo or _get_repo_from_git()
    if not rep:
        typer.echo("Could not determine repo. Pass --repo OWNER/REPO.", err=True)
        raise typer.Exit(1)
    repo_obj = gh.get_repo(rep)
    # Reply endpoint uses comment_id only; we need a PR to get the pull object.
    # PyGithub: create_review_comment_reply is on PullRequest and needs comment_id (int) and body.
    # We have comment_id but not pr number. GitHub API: POST /repos/owner/repo/pulls/comments/comment_id/replies
    # So we don't need PR number. PyGithub's PullRequest.create_review_comment_reply(comment_id, body) - let me check
    # if we can get there without a PR. We need a PullRequest instance. So we need to find the PR that contains
    # this comment, or use the low-level API. Actually the REST endpoint is under pulls/comments/ID/replies - so
    # we don't need pull number. In PyGithub we might need to use the repository's _requester. I'll fetch the
    # comment first to get its pull request URL, or use requester.
    comment = repo_obj.get_pull_comment(int(comment_id))
    # PullRequestComment has create_reply? Let me check - the web said create_review_comment_reply is on PullRequest.
    # So we need pr number. We can get it from the comment: comment has pull_request_review_id or we can get
    # comment.raw_data and see if there's a pull_request url. Actually in GitHub API, the comment object has
    # "pull_request_url" which gives us the PR. So: get comment, parse pull_request_url to get pull number, then
    # pr.create_review_comment_reply(comment_id, body).
    pr_url = comment.raw_data.get("pull_request_url") or comment.raw_data.get("_links", {}).get("pull_request", {}).get(
        "href"
    )
    if not pr_url:
        # Fallback: comment might have pull_request in raw_data
        pr_url = (
            comment.raw_data.get("pull_request", {}).get("url")
            if isinstance(comment.raw_data.get("pull_request"), dict)
            else None
        )
    if not pr_url:
        typer.echo("Could not determine PR from comment.", err=True)
        raise typer.Exit(1)
    pr_number = int(pr_url.rstrip("/").split("/")[-1])
    pr = repo_obj.get_pull(pr_number)
    pr.create_review_comment_reply(int(comment_id), reply_text)

    result = ReplyOutput(comment_id=int(comment_id), body=reply_text)
    print(result.model_dump_json(indent=2))  # noqa: T201


if __name__ == "__main__":
    app()
