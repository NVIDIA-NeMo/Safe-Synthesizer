# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# dependencies = ["pyyaml>=6.0,<7"]
# ///
"""Check trusted publishing boundaries without uploading a package."""

import unittest
from pathlib import Path

import yaml

WORKFLOW = yaml.safe_load((Path(__file__).resolve().parents[1] / "workflows/release.yml").read_text())


class ReleaseWorkflow(unittest.TestCase):
    """Keep verification before publishing and release side effects after it."""

    def test_upload_uses_verified_build_artifact(self) -> None:
        """Upload the artifact emitted only after the clean installation check."""
        build = WORKFLOW["jobs"]["build-wheel"]
        steps = build["steps"]
        verify = next(index for index, step in enumerate(steps) if step.get("run") == "mise run release:verify-wheel")
        artifact = next(
            index for index, step in enumerate(steps) if step.get("uses", "").startswith("actions/upload-artifact@")
        )
        self.assertLess(verify, artifact)
        upload = WORKFLOW["jobs"]["publish-pypi"]
        self.assertEqual(upload["needs"], "build-wheel")
        self.assertEqual(
            upload["steps"][0]["with"]["name"], "nemo-safe-synthesizer-${{ needs.build-wheel.outputs.version }}"
        )

    def test_only_upload_can_request_oidc(self) -> None:
        """Source execution cannot request the publishing identity."""
        self.assertNotIn("id-token", WORKFLOW["permissions"])
        jobs = WORKFLOW["jobs"]
        for name, job in jobs.items():
            self.assertEqual(job.get("permissions", {}).get("id-token") == "write", name == "publish-pypi")
        upload = jobs["publish-pypi"]
        self.assertEqual(upload["environment"]["name"], "pypi")
        self.assertEqual(upload["permissions"], {"id-token": "write"})
        self.assertEqual(len(upload["steps"]), 2)
        for step in upload["steps"]:
            self.assertNotIn("run", step)
            self.assertNotIn("checkout", step["uses"])
            self.assertNotIn("password", step.get("with", {}))

    def test_release_side_effects_require_successful_upload(self) -> None:
        """GitHub's default success guard blocks downstream jobs on upload failure."""
        jobs = WORKFLOW["jobs"]
        for name in ("create-gh-release", "publish-release-docs"):
            self.assertIn("publish-pypi", jobs[name]["needs"])
            self.assertNotIn("always()", jobs[name].get("if", ""))
            self.assertNotIn("!failure()", jobs[name].get("if", ""))

    def test_only_canonical_tags_publish_to_production(self) -> None:
        """Retain the release tag trigger and remove all TestPyPI uploads."""
        self.assertEqual(WORKFLOW[True], {"push": {"tags": ["v*"]}})
        upload = WORKFLOW["jobs"]["publish-pypi"]
        self.assertEqual(
            upload["if"],
            "${{ github.repository == 'NVIDIA-NeMo/Safe-Synthesizer' && github.ref_type == 'tag' && startsWith(github.ref_name, 'v') }}",
        )
        publish = upload["steps"][1]
        self.assertTrue(publish["uses"].startswith("pypa/gh-action-pypi-publish@"))
        self.assertNotIn("repository-url", publish.get("with", {}))
        self.assertNotIn("publish-testpypi", WORKFLOW["jobs"])


if __name__ == "__main__":
    unittest.main()
