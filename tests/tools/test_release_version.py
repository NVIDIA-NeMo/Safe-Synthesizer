# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "release_version.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_version", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_version = _load_tool()


class ReleaseVersionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmpdir.name)
        self._git("init")
        self._git("config", "user.email", "test@example.com")
        self._git("config", "user.name", "Test User")
        self._git("commit", "--allow-empty", "-m", "initial")

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _run(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(args, cwd=self.repo, check=False, capture_output=True, text=True)

    def _git(self, *args: str) -> str:
        result = self._run("git", *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def _tag(self, tag: str) -> None:
        self._git("tag", tag)

    def test_default_patch_uses_latest_stable_tag_and_ignores_rc_tags(self) -> None:
        self._tag("v1.2.0")
        self._tag("v1.10.0")
        self._tag("v1.11.0rc0")

        plan = release_version.plan_release(cwd=self.repo)

        self.assertEqual(plan.latest_stable_tag, "v1.10.0")
        self.assertEqual(plan.latest_stable_version, "1.10.0")
        self.assertEqual(plan.bump, "patch")
        self.assertEqual(plan.next_tag, "v1.10.1rc0")
        self.assertEqual(plan.next_version, "1.10.1rc0")

    def test_historical_development_tag_is_ignored(self) -> None:
        self._tag("v0.0.0+dev5")
        self._tag("v1.2.3")

        plan = release_version.plan_release(cwd=self.repo)

        self.assertEqual(plan.latest_stable_tag, "v1.2.3")
        self.assertEqual(plan.next_tag, "v1.2.4rc0")

    def test_major_minor_and_patch_bumps_prepare_rc0(self) -> None:
        self._tag("v2.3.4")

        cases = {
            "major": ("v3.0.0rc0", "3.0.0rc0"),
            "minor": ("v2.4.0rc0", "2.4.0rc0"),
            "patch": ("v2.3.5rc0", "2.3.5rc0"),
        }
        for bump, expected in cases.items():
            with self.subTest(bump=bump):
                plan = release_version.plan_release(bump=bump, cwd=self.repo)
                self.assertEqual((plan.next_tag, plan.next_version), expected)

    def test_post_bump_prepares_first_post_release(self) -> None:
        self._tag("v1.5.0")
        self._tag("v1.6.0")

        plan = release_version.plan_release(bump="post", cwd=self.repo)

        self.assertEqual(plan.latest_stable_tag, "v1.6.0")
        self.assertEqual(plan.latest_stable_version, "1.6.0")
        self.assertEqual(plan.bump, "post")
        self.assertEqual(plan.next_tag, "v1.6.0.post1")
        self.assertEqual(plan.next_version, "1.6.0.post1")

    def test_post_bump_increments_highest_post_for_latest_stable(self) -> None:
        self._tag("v1.5.0")
        self._tag("v1.5.0.post7")
        self._tag("v1.6.0")
        self._tag("v1.6.0.post1")
        self._tag("v1.6.0.post10")

        plan = release_version.plan_release(bump="post", cwd=self.repo)

        self.assertEqual(plan.next_tag, "v1.6.0.post11")

    def test_regular_patch_accepts_post_tags_and_uses_stable_base(self) -> None:
        self._tag("v1.6.0")
        self._tag("v1.6.0.post1")

        plan = release_version.plan_release(cwd=self.repo)

        self.assertEqual(plan.latest_stable_tag, "v1.6.0")
        self.assertEqual(plan.next_tag, "v1.6.1rc0")

    def test_json_cli_resolves_candidate_ref(self) -> None:
        self._tag("v0.1.0")
        first_commit = self._git("rev-parse", "HEAD")
        self._git("commit", "--allow-empty", "-m", "second")

        result = self._run(sys.executable, str(TOOL_PATH), "--json", "--ref", first_commit)

        self.assertEqual(result.returncode, 0, result.stderr)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["candidate_ref"], first_commit)
        self.assertEqual(payload["candidate_commit"], first_commit)
        self.assertEqual(payload["next_tag"], "v0.1.1rc0")
        self.assertEqual(result.stderr, "")

    def test_human_cli_does_not_create_candidate_tag(self) -> None:
        self._tag("v3.4.5")
        tags_before = self._git("tag", "--list")

        result = self._run(sys.executable, str(TOOL_PATH))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Next rc0 tag: v3.4.6rc0", result.stdout)
        self.assertIn("No tag was created, deleted, or pushed.", result.stdout)
        self.assertEqual(self._git("tag", "--list"), tags_before)

    def test_human_cli_describes_post_release_without_creating_tag(self) -> None:
        self._tag("v1.6.0")
        tags_before = self._git("tag", "--list")

        result = self._run(sys.executable, str(TOOL_PATH), "--bump", "post")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Next Safe-Synthesizer post-release", result.stdout)
        self.assertIn("Next post-release tag: v1.6.0.post1", result.stdout)
        self.assertIn("No tag was created, deleted, or pushed.", result.stdout)
        self.assertEqual(self._git("tag", "--list"), tags_before)

    def test_post_release_without_matching_stable_tag_is_refused(self) -> None:
        self._tag("v1.5.0")
        self._tag("v1.6.0.post1")

        with self.assertRaisesRegex(release_version.ReleaseVersionError, "missing stable tag v1.6.0"):
            release_version.plan_release(bump="post", cwd=self.repo)

    def test_malformed_v_tag_is_an_explicit_error(self) -> None:
        self._tag("v0.1.0")
        self._tag("v0.2.0-rc0")

        with self.assertRaisesRegex(release_version.ReleaseVersionError, "Malformed release tag"):
            release_version.plan_release(cwd=self.repo)

    def test_missing_stable_tag_is_an_explicit_json_error(self) -> None:
        self._tag("v0.1.0rc0")

        result = self._run(sys.executable, str(TOOL_PATH), "--json")

        self.assertEqual(result.returncode, 2)
        self.assertIn("No stable release tags found", json.loads(result.stdout)["error"])
        self.assertEqual(result.stderr, "")

    def test_existing_candidate_tag_is_refused(self) -> None:
        self._tag("v1.2.3")
        self._tag("v1.2.4rc0")

        with self.assertRaisesRegex(release_version.ReleaseVersionError, "already exist"):
            release_version.plan_release(cwd=self.repo)

    def test_existing_later_candidate_tag_is_refused(self) -> None:
        self._tag("v1.2.3")
        self._tag("v1.2.4rc1")

        with self.assertRaisesRegex(release_version.ReleaseVersionError, "v1.2.4rc1"):
            release_version.plan_release(cwd=self.repo)


if __name__ == "__main__":
    unittest.main()
