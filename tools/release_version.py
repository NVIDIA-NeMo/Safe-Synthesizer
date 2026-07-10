#!/usr/bin/env -S uv run --no-project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the next Safe-Synthesizer release-candidate tag name.

The tool inspects local Git tags only. It never creates, deletes, pushes, or
publishes anything.
"""

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

BUMP_CHOICES = ("major", "minor", "patch")
STABLE_TAG_RE = re.compile(r"^v(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)$")
RC_TAG_RE = re.compile(r"^v(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)rc(?P<rc>0|[1-9]\d*)$")
HISTORICAL_DEV_TAG_RE = re.compile(r"^v(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\+dev(?:0|[1-9]\d*)$")


class ReleaseVersionError(Exception):
    """Known release preparation error."""


@dataclass(frozen=True, order=True)
class StableVersion:
    """A stable major.minor.patch release version."""

    major: int
    minor: int
    patch: int

    @classmethod
    def from_tag(cls, tag: str) -> Self | None:
        """Parse a stable release tag, returning ``None`` for non-stable tags."""
        match = STABLE_TAG_RE.fullmatch(tag)
        if match is None:
            return None
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
        )

    def bump(self, part: str) -> Self:
        """Return the next stable base version for the requested bump part."""
        match part:
            case "major":
                return StableVersion(self.major + 1, 0, 0)
            case "minor":
                return StableVersion(self.major, self.minor + 1, 0)
            case "patch":
                return StableVersion(self.major, self.minor, self.patch + 1)
            case _:
                raise ReleaseVersionError(f"Unsupported bump part: {part!r}. Use one of: major, minor, patch.")

    @property
    def version(self) -> str:
        """The PEP 440 stable version without a leading ``v``."""
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def stable_tag(self) -> str:
        """The Git tag for this stable release version."""
        return f"v{self.version}"

    @property
    def rc0_version(self) -> str:
        """The PEP 440 rc0 version without a leading ``v``."""
        return f"{self.version}rc0"

    @property
    def rc0_tag(self) -> str:
        """The Git tag for this release-candidate version."""
        return f"v{self.rc0_version}"


@dataclass(frozen=True)
class ReleasePlan:
    """Computed release-candidate tag plan."""

    latest_stable_tag: str
    latest_stable_version: str
    bump: str
    candidate_ref: str
    candidate_commit: str
    next_version: str
    next_tag: str


def _run_git(args: Sequence[str], *, cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ReleaseVersionError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _is_valid_release_tag(tag: str) -> bool:
    return STABLE_TAG_RE.fullmatch(tag) is not None or RC_TAG_RE.fullmatch(tag) is not None


def _find_latest_stable(tags: Sequence[str]) -> tuple[str, StableVersion]:
    malformed = sorted(
        tag
        for tag in tags
        if tag.startswith("v") and not _is_valid_release_tag(tag) and HISTORICAL_DEV_TAG_RE.fullmatch(tag) is None
    )
    if malformed:
        formatted = ", ".join(malformed)
        raise ReleaseVersionError(
            f"Malformed release tag(s): {formatted}. Expected vMAJOR.MINOR.PATCH or vMAJOR.MINOR.PATCHrcN."
        )

    stable_tags: list[tuple[StableVersion, str]] = []
    for tag in tags:
        version = StableVersion.from_tag(tag)
        if version is not None:
            stable_tags.append((version, tag))

    if not stable_tags:
        raise ReleaseVersionError("No stable release tags found. Expected at least one vMAJOR.MINOR.PATCH tag.")

    latest_version, latest_tag = max(stable_tags, key=lambda item: item[0])
    return latest_tag, latest_version


def _list_local_tags(*, cwd: Path) -> list[str]:
    output = _run_git(["tag", "--list"], cwd=cwd)
    return [line for line in output.splitlines() if line]


def _resolve_commit(ref: str, *, cwd: Path) -> str:
    if not ref:
        raise ReleaseVersionError("Candidate ref must not be empty.")
    return _run_git(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], cwd=cwd).strip()


def plan_release(*, bump: str = "patch", ref: str = "HEAD", cwd: Path | None = None) -> ReleasePlan:
    """Compute the next rc0 release tag from local Git tags.

    Args:
        bump: Version part to increment: ``major``, ``minor``, or ``patch``.
        ref: Git ref or commit to use as the candidate tag target.
        cwd: Directory where Git commands should run.

    Returns:
        A release plan containing the latest stable tag, next rc0 tag, and
        resolved candidate commit.

    Raises:
        ReleaseVersionError: If Git inspection fails, release tags are
            malformed, no stable tag exists, or the candidate tag already
            exists locally.
    """
    repo = cwd or Path.cwd()
    tags = _list_local_tags(cwd=repo)
    latest_tag, latest_version = _find_latest_stable(tags)
    next_version = latest_version.bump(bump)
    next_tag = next_version.rc0_tag
    candidate_prefix = f"{next_version.stable_tag}rc"
    existing_candidates = sorted(
        tag for tag in tags if tag.startswith(candidate_prefix) and RC_TAG_RE.fullmatch(tag) is not None
    )
    if existing_candidates:
        formatted = ", ".join(existing_candidates)
        raise ReleaseVersionError(
            f"Candidate tag(s) already exist for {next_version.version}: {formatted}; refusing to prepare rc0."
        )

    return ReleasePlan(
        latest_stable_tag=latest_tag,
        latest_stable_version=latest_version.version,
        bump=bump,
        candidate_ref=ref,
        candidate_commit=_resolve_commit(ref, cwd=repo),
        next_version=next_version.rc0_version,
        next_tag=next_tag,
    )


def _format_human(plan: ReleasePlan) -> str:
    return "\n".join(
        [
            "Next Safe-Synthesizer release candidate",
            f"Latest stable tag: {plan.latest_stable_tag}",
            f"Latest stable version: {plan.latest_stable_version}",
            f"Bump: {plan.bump}",
            f"Candidate ref: {plan.candidate_ref}",
            f"Candidate commit: {plan.candidate_commit}",
            f"Next rc0 tag: {plan.next_tag}",
            f"Next PyPI version: {plan.next_version}",
            "",
            "No tag was created, deleted, or pushed.",
            "After review, create the tag manually with:",
            f"  git tag {plan.next_tag} {plan.candidate_commit}",
        ]
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the next rc0 release tag name from local Safe-Synthesizer Git tags."
    )
    parser.add_argument(
        "--bump",
        choices=BUMP_CHOICES,
        default="patch",
        help="Version part to increment from the latest stable tag. Defaults to patch.",
    )
    parser.add_argument(
        "--ref",
        default="HEAD",
        help="Candidate Git commit/ref that the prepared tag should target. Defaults to HEAD.",
    )
    parser.add_argument("--json", action="store_true", dest="json_output", help="Emit machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)

    try:
        plan = plan_release(bump=args.bump, ref=args.ref)
    except ReleaseVersionError as error:
        if args.json_output:
            sys.stdout.write(json.dumps({"error": str(error)}, sort_keys=True) + "\n")
        else:
            sys.stderr.write(f"release-version: error: {error}\n")
        return 2

    if args.json_output:
        sys.stdout.write(json.dumps(asdict(plan), sort_keys=True) + "\n")
    else:
        sys.stdout.write(_format_human(plan) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
