#!/usr/bin/env -S uv run --no-project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["packaging>=25.0.0"]
# ///
"""Prepare the next Safe-Synthesizer release tag name.

The tool inspects local Git tags only. It never creates, deletes, pushes, or
publishes anything.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Self

from packaging.version import InvalidVersion, Version

BUMP_CHOICES = ("major", "minor", "patch", "post")


class ReleaseVersionError(Exception):
    """Known release preparation error."""


@dataclass(frozen=True, order=True)
class StableVersion:
    """A stable major.minor.patch release version."""

    major: int
    minor: int
    patch: int

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

    def post_version(self, number: int) -> str:
        """Return a PEP 440 post-release version without a leading ``v``."""
        return f"{self.version}.post{number}"

    def post_tag(self, number: int) -> str:
        """Return the Git tag for a PEP 440 post-release version."""
        return f"v{self.post_version(number)}"


@dataclass(frozen=True)
class ParsedReleaseTag:
    """Canonical release tag classified according to project policy."""

    base: StableVersion
    kind: Literal["stable", "rc", "post", "historical-dev"]
    number: int | None = None


@dataclass(frozen=True)
class ReleasePlan:
    """Computed release tag plan."""

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


def _parse_release_tag(tag: str) -> ParsedReleaseTag | None:
    if not tag.startswith("v"):
        return None
    try:
        version = Version(tag.removeprefix("v"))
    except InvalidVersion:
        return None
    if version.epoch != 0 or len(version.release) != 3 or tag != f"v{version}":
        return None

    base = StableVersion(*version.release)
    if version.local is not None:
        local = version.local
        if (
            version.pre is None
            and version.post is None
            and version.dev is None
            and local.startswith("dev")
            and local.removeprefix("dev").isdigit()
        ):
            return ParsedReleaseTag(base=base, kind="historical-dev")
        return None
    if version.dev is not None:
        return None
    if version.pre is not None:
        pre_kind, pre_number = version.pre
        if pre_kind == "rc" and version.post is None:
            return ParsedReleaseTag(base=base, kind="rc", number=pre_number)
        return None
    if version.post is not None:
        return ParsedReleaseTag(base=base, kind="post", number=version.post)
    return ParsedReleaseTag(base=base, kind="stable")


def _find_latest_stable(tags: Sequence[str]) -> tuple[str, StableVersion]:
    parsed_tags = [(tag, _parse_release_tag(tag)) for tag in tags if tag.startswith("v")]
    malformed = sorted(tag for tag, parsed in parsed_tags if parsed is None)
    if malformed:
        formatted = ", ".join(malformed)
        raise ReleaseVersionError(
            "Malformed release tag(s): "
            f"{formatted}. Expected vMAJOR.MINOR.PATCH, vMAJOR.MINOR.PATCHrcN, or vMAJOR.MINOR.PATCH.postN."
        )

    stable_tags = [(parsed.base, tag) for tag, parsed in parsed_tags if parsed is not None and parsed.kind == "stable"]

    if not stable_tags:
        raise ReleaseVersionError("No stable release tags found. Expected at least one vMAJOR.MINOR.PATCH tag.")

    stable_versions = {version for version, _ in stable_tags}
    dangling_posts: list[tuple[str, StableVersion]] = []
    for tag, parsed in parsed_tags:
        if parsed is not None and parsed.kind == "post" and parsed.base not in stable_versions:
            dangling_posts.append((tag, parsed.base))
    if dangling_posts:
        tag, base = sorted(dangling_posts)[0]
        raise ReleaseVersionError(f"Post-release tag {tag} is missing stable tag {base.stable_tag}.")

    latest_version, latest_tag = max(stable_tags, key=lambda item: item[0])
    return latest_tag, latest_version


def _list_local_tags(*, cwd: Path) -> list[str]:
    output = _run_git(["tag", "--list"], cwd=cwd)
    return [line for line in output.splitlines() if line]


def _resolve_commit(ref: str, *, cwd: Path) -> str:
    if not ref:
        raise ReleaseVersionError("Candidate ref must not be empty.")
    return _run_git(["rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"], cwd=cwd).strip()


def _next_post_release(tags: Sequence[str], base: StableVersion) -> tuple[str, str]:
    post_numbers: list[int] = []
    for tag in tags:
        parsed = _parse_release_tag(tag)
        if parsed is not None and parsed.kind == "post" and parsed.base == base and parsed.number is not None:
            post_numbers.append(parsed.number)
    next_post_number = max(post_numbers, default=0) + 1
    return base.post_version(next_post_number), base.post_tag(next_post_number)


def _next_release_candidate(tags: Sequence[str], base: StableVersion, bump: str) -> tuple[str, str]:
    next_version = base.bump(bump)
    existing_candidates = sorted(
        tag
        for tag in tags
        if (parsed := _parse_release_tag(tag)) is not None and parsed.kind == "rc" and parsed.base == next_version
    )
    if existing_candidates:
        formatted = ", ".join(existing_candidates)
        raise ReleaseVersionError(
            f"Candidate tag(s) already exist for {next_version.version}: {formatted}; refusing to prepare rc0."
        )
    return next_version.rc0_version, next_version.rc0_tag


def plan_release(*, bump: str = "patch", ref: str = "HEAD", cwd: Path | None = None) -> ReleasePlan:
    """Compute the next release tag from local Git tags.

    Args:
        bump: Version part to increment: ``major``, ``minor``, ``patch``, or ``post``.
        ref: Git ref or commit to use as the candidate tag target.
        cwd: Directory where Git commands should run.

    Returns:
        A release plan containing the latest stable tag, next release tag, and
        resolved target commit.

    Raises:
        ReleaseVersionError: If Git inspection fails, release tags are
            malformed, no stable tag exists, or the candidate tag already
            exists locally.
    """
    repo = cwd or Path.cwd()
    tags = _list_local_tags(cwd=repo)
    latest_tag, latest_version = _find_latest_stable(tags)
    if bump == "post":
        next_version_text, next_tag = _next_post_release(tags, latest_version)
    else:
        next_version_text, next_tag = _next_release_candidate(tags, latest_version, bump)

    return ReleasePlan(
        latest_stable_tag=latest_tag,
        latest_stable_version=latest_version.version,
        bump=bump,
        candidate_ref=ref,
        candidate_commit=_resolve_commit(ref, cwd=repo),
        next_version=next_version_text,
        next_tag=next_tag,
    )


def _format_human(plan: ReleasePlan) -> str:
    is_post = plan.bump == "post"
    heading = "Next Safe-Synthesizer post-release" if is_post else "Next Safe-Synthesizer release candidate"
    tag_label = "Next post-release tag" if is_post else "Next rc0 tag"
    return "\n".join(
        [
            heading,
            f"Latest stable tag: {plan.latest_stable_tag}",
            f"Latest stable version: {plan.latest_stable_version}",
            f"Bump: {plan.bump}",
            f"Candidate ref: {plan.candidate_ref}",
            f"Candidate commit: {plan.candidate_commit}",
            f"{tag_label}: {plan.next_tag}",
            f"Next PyPI version: {plan.next_version}",
            "",
            "No tag was created, deleted, or pushed.",
            "After review, create the tag manually with:",
            f"  git tag {plan.next_tag} {plan.candidate_commit}",
        ]
    )


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the next release tag from local Safe-Synthesizer Git tags.")
    parser.add_argument(
        "--bump",
        choices=BUMP_CHOICES,
        default="patch",
        help="Version part to increment from the latest stable tag, or post for its next post-release.",
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
