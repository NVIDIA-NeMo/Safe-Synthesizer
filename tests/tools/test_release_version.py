# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal, Protocol, cast

import pytest

Bump = Literal["major", "minor", "patch", "post"]
StableBump = Literal["major", "minor", "patch"]


class ReleasePlan(Protocol):
    """Typed view of a plan returned by the dynamically loaded tool."""

    latest_stable_tag: str
    latest_stable_version: str
    bump: Bump
    candidate_ref: str
    candidate_commit: str
    next_version: str
    next_tag: str


class ReleaseVersionTool(Protocol):
    """Typed interface exposed by the dynamically loaded release tool."""

    ReleaseVersionError: type[Exception]

    def plan_release(self, *, bump: Bump = "patch", ref: str = "HEAD", cwd: Path | None = None) -> ReleasePlan: ...


def _load_tool(tool_path: Path) -> ReleaseVersionTool:
    spec = importlib.util.spec_from_file_location("release_version", tool_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {tool_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(ReleaseVersionTool, module)


@pytest.fixture(scope="module")
def release_tool(pytestconfig: pytest.Config) -> tuple[ReleaseVersionTool, Path]:
    """Load the release helper from pytest's discovered repository root."""
    tool_path = Path(pytestconfig.rootpath) / "tools" / "release_version.py"
    return _load_tool(tool_path), tool_path


class TestReleaseVersion:
    @pytest.fixture(autouse=True)
    def _configure_release_tool(self, release_tool: tuple[ReleaseVersionTool, Path]) -> None:
        self.release_version, self.tool_path = release_tool

    def setup_method(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmpdir.name)
        self._git("init")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "Test User")
        self._git("commit", "--allow-empty", "-m", "initial")

    def teardown_method(self) -> None:
        self.tmpdir.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=self.repo, check=False, capture_output=True, text=True)

    def _git(self, *args: str) -> str:
        result = self._run("git", *args)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def _tag(self, tag: str) -> None:
        self._git("tag", tag)

    def test_default_patch_uses_latest_stable_tag_and_ignores_rc_tags(self) -> None:
        self._tag("v1.2.0")
        self._tag("v1.10.0")
        self._tag("v1.11.0rc0")

        plan = self.release_version.plan_release(cwd=self.repo)

        assert plan.latest_stable_tag == "v1.10.0"
        assert plan.latest_stable_version == "1.10.0"
        assert plan.bump == "patch"
        assert plan.next_tag == "v1.10.1rc0"
        assert plan.next_version == "1.10.1rc0"

    def test_historical_development_tag_is_ignored(self) -> None:
        self._tag("v0.0.0+dev5")
        self._tag("v1.2.3")

        plan = self.release_version.plan_release(cwd=self.repo)

        assert plan.latest_stable_tag == "v1.2.3"
        assert plan.next_tag == "v1.2.4rc0"

    @pytest.mark.parametrize(
        ("bump", "expected"),
        [
            ("major", ("v3.0.0rc0", "3.0.0rc0")),
            ("minor", ("v2.4.0rc0", "2.4.0rc0")),
            ("patch", ("v2.3.5rc0", "2.3.5rc0")),
        ],
    )
    def test_major_minor_and_patch_bumps_prepare_rc0(self, bump: StableBump, expected: tuple[str, str]) -> None:
        self._tag("v2.3.4")

        plan = self.release_version.plan_release(bump=bump, cwd=self.repo)

        assert (plan.next_tag, plan.next_version) == expected

    def test_post_bump_prepares_first_post_release(self) -> None:
        self._tag("v1.5.0")
        self._tag("v1.6.0")

        plan = self.release_version.plan_release(bump="post", cwd=self.repo)

        assert plan.latest_stable_tag == "v1.6.0"
        assert plan.latest_stable_version == "1.6.0"
        assert plan.bump == "post"
        assert plan.next_tag == "v1.6.0.post1"
        assert plan.next_version == "1.6.0.post1"

    def test_post_bump_increments_highest_post_for_latest_stable(self) -> None:
        self._tag("v1.5.0")
        self._tag("v1.5.0.post7")
        self._tag("v1.6.0")
        self._tag("v1.6.0.post1")
        self._tag("v1.6.0.post10")

        plan = self.release_version.plan_release(bump="post", cwd=self.repo)

        assert plan.next_tag == "v1.6.0.post11"

    def test_regular_patch_accepts_post_tags_and_uses_stable_base(self) -> None:
        self._tag("v1.6.0")
        self._tag("v1.6.0.post1")

        plan = self.release_version.plan_release(cwd=self.repo)

        assert plan.latest_stable_tag == "v1.6.0"
        assert plan.next_tag == "v1.6.1rc0"

    def test_json_cli_resolves_candidate_ref(self) -> None:
        self._tag("v0.1.0")
        first_commit = self._git("rev-parse", "HEAD")
        self._git("commit", "--allow-empty", "-m", "second")

        result = self._run(sys.executable, str(self.tool_path), "--json", "--ref", first_commit)

        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["candidate_ref"] == first_commit
        assert payload["candidate_commit"] == first_commit
        assert payload["next_tag"] == "v0.1.1rc0"
        assert result.stderr == ""

    def test_human_cli_does_not_create_candidate_tag(self) -> None:
        self._tag("v3.4.5")
        tags_before = self._git("tag", "--list")

        result = self._run(sys.executable, str(self.tool_path))

        assert result.returncode == 0, result.stderr
        assert "Next rc0 tag: v3.4.6rc0" in result.stdout
        assert "No tag was created, deleted, or pushed." in result.stdout
        assert self._git("tag", "--list") == tags_before

    def test_human_cli_describes_post_release_without_creating_tag(self) -> None:
        self._tag("v1.6.0")
        tags_before = self._git("tag", "--list")

        result = self._run(sys.executable, str(self.tool_path), "--bump", "post")

        assert result.returncode == 0, result.stderr
        assert "Next Safe-Synthesizer post-release" in result.stdout
        assert "Next post-release tag: v1.6.0.post1" in result.stdout
        assert "No tag was created, deleted, or pushed." in result.stdout
        assert self._git("tag", "--list") == tags_before

    def test_post_release_without_matching_stable_tag_is_refused(self) -> None:
        self._tag("v1.5.0")
        self._tag("v1.6.0.post1")

        with pytest.raises(self.release_version.ReleaseVersionError, match="missing stable tag v1.6.0"):
            self.release_version.plan_release(bump="post", cwd=self.repo)

    def test_malformed_v_tag_is_an_explicit_error(self) -> None:
        self._tag("v0.1.0")
        self._tag("v0.2.0-rc0")

        with pytest.raises(self.release_version.ReleaseVersionError, match="Malformed release tag"):
            self.release_version.plan_release(cwd=self.repo)

    @pytest.mark.parametrize("tag", ["v1.2", "v01.2.3", "v1.2.3-post1", "v1.2.3a1"])
    def test_noncanonical_or_unsupported_pep440_tag_is_refused(self, tag: str) -> None:
        self._tag("v0.1.0")
        self._tag(tag)

        with pytest.raises(self.release_version.ReleaseVersionError, match="Malformed release tag"):
            self.release_version.plan_release(cwd=self.repo)

    def test_missing_stable_tag_is_an_explicit_json_error(self) -> None:
        self._tag("v0.1.0rc0")

        result = self._run(sys.executable, str(self.tool_path), "--json")

        assert result.returncode == 2
        assert "No stable release tags found" in json.loads(result.stdout)["error"]
        assert result.stderr == ""

    def test_existing_candidate_tag_is_refused(self) -> None:
        self._tag("v1.2.3")
        self._tag("v1.2.4rc0")

        with pytest.raises(self.release_version.ReleaseVersionError, match="already exist"):
            self.release_version.plan_release(cwd=self.repo)

    def test_existing_later_candidate_tag_is_refused(self) -> None:
        self._tag("v1.2.3")
        self._tag("v1.2.4rc1")

        with pytest.raises(self.release_version.ReleaseVersionError, match="v1.2.4rc1"):
            self.release_version.plan_release(cwd=self.repo)
