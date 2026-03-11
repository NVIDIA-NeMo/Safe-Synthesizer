#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# gh-pr-helper: fetch PR comments and reply to inline review comments via GitHub API.
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

import os
import re
import subprocess
import sys
from pathlib import Path

import typer  # provided by PEP 723 deps when run via uv run --script
from github import Auth, Github
from pydantic import BaseModel, Field

app = typer.Typer(
    name="gh-pr-helper",
    help="Fetch PR comments and reply to inline review comments (GitHub API).",
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


class PRCommentsOutput(BaseModel):
    """Single JSON output for comments command: inline (review) + top-level (conversation)."""

    pr_number: int
    repo: str
    inline: list[ReviewComment] = Field(default_factory=list)
    top_level: list[IssueComment] = Field(default_factory=list)


class ReplyOutput(BaseModel):
    """Single JSON output for reply command after posting a reply."""

    comment_id: int
    body: str
    success: bool = True


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


@app.command()
def comments(
    pr_number: str | None = typer.Argument(None, help="PR number (default: from current branch)"),
    repo: str | None = typer.Option(None, "--repo", "-r", help="OWNER/REPO (default: from git remote)"),
    token: str | None = typer.Option(
        None, "--token", "-t", help="GitHub token (default: GITHUB_TOKEN or gh auth token)"
    ),
) -> None:
    """Fetch all PR comments: inline (code review) + top-level (conversation). Outputs JSON to stdout."""
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

    pr = repo_obj.get_pull(int(pr_number))
    inline = [ReviewComment.model_validate(c.raw_data) for c in pr.get_review_comments()]
    issue = repo_obj.get_issue(int(pr_number))
    top_level = [IssueComment.model_validate(c.raw_data) for c in issue.get_comments()]

    out = PRCommentsOutput(
        pr_number=int(pr_number),
        repo=rep,
        inline=inline,
        top_level=top_level,
    )
    # Single JSON object, written once (no sys.stdout / multiple writes).
    print(out.model_dump_json(indent=2))


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
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
