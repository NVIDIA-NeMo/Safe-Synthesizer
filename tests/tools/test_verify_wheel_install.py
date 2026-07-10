# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the end-user built-wheel verification gate."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).parents[2]
TOOL_PATH = REPO_ROOT / "tools" / "verify_wheel_install.py"


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_wheel_install", TOOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {TOOL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_wheel_install = _load_tool()


class VerifyWheelInstallTest(unittest.TestCase):
    def test_sanitized_environment_removes_project_and_resolver_state(self) -> None:
        source = {
            "HOME": "/home/tester",
            "PATH": "/usr/bin",
            "PYTHONPATH": "/checkout/src",
            "VIRTUAL_ENV": "/checkout/.venv",
            "UV_CONFIG_FILE": "/checkout/uv.toml",
            "UV_CONSTRAINT": "/checkout/constraints.txt",
            "UV_CACHE_DIR": "/cache/uv",
            "UV_INDEX": "https://private.example/simple",
            "UV_OVERRIDE": "/checkout/overrides.txt",
            "UV_PROJECT": "/checkout",
            "UV_PROJECT_ENVIRONMENT": "/checkout/.venv",
            "UV_WORKING_DIR": "/checkout",
        }

        result = verify_wheel_install.sanitized_environment(source)

        self.assertEqual(result["HOME"], "/home/tester")
        self.assertEqual(result["PATH"], "/usr/bin")
        self.assertNotIn("PYTHONPATH", result)
        self.assertNotIn("VIRTUAL_ENV", result)
        self.assertEqual(
            {key: value for key, value in result.items() if key.startswith("UV_")},
            {
                "UV_CACHE_DIR": "/cache/uv",
                "UV_NO_CONFIG": "1",
                "UV_NO_SOURCES": "1",
            },
        )

    def test_cpu_plan_is_an_actual_install_with_documented_indexes(self) -> None:
        wheel = Path("/checkout/dist/nemo_safe_synthesizer-1.2.3-py3-none-any.whl")

        plan = verify_wheel_install.build_plan(wheel, verify_wheel_install.Variant.CPU)

        self.assertEqual(
            plan.requirement,
            "nemo-safe-synthesizer[cpu,engine] @ file:///checkout/dist/nemo_safe_synthesizer-1.2.3-py3-none-any.whl",
        )
        self.assertEqual(
            plan.index_args,
            (
                "--default-index",
                "https://pypi.org/simple",
                "--index",
                "https://download.pytorch.org/whl/cpu",
                "--index-strategy",
                "unsafe-best-match",
            ),
        )
        self.assertFalse(plan.resolve_only)

    def test_cuda_plan_resolves_with_all_documented_indexes(self) -> None:
        wheel = Path("/checkout/dist/nemo_safe_synthesizer-1.2.3-py3-none-any.whl")

        plan = verify_wheel_install.build_plan(wheel, verify_wheel_install.Variant.CU129)

        self.assertEqual(
            plan.requirement,
            "nemo-safe-synthesizer[cu129,engine] @ file:///checkout/dist/nemo_safe_synthesizer-1.2.3-py3-none-any.whl",
        )
        self.assertEqual(
            plan.index_args,
            (
                "--default-index",
                "https://pypi.org/simple",
                "--index",
                "https://flashinfer.ai/whl/cu129",
                "--index",
                "https://download.pytorch.org/whl/cu129",
                "--index",
                "https://wheels.vllm.ai/ee0da84ab9e04ac7610e28580af62c365e898389/cu129",
                "--index-strategy",
                "unsafe-best-match",
            ),
        )
        self.assertTrue(plan.resolve_only)

    def test_actual_install_commands_check_dependencies_version_import_and_cli(self) -> None:
        plan = verify_wheel_install.build_plan(
            Path("/checkout/dist/nemo_safe_synthesizer-1.2.3-py3-none-any.whl"),
            verify_wheel_install.Variant.CPU,
        )

        commands = verify_wheel_install.build_commands(plan, python="3.13", venv=Path("/tmp/gate/.venv"))

        self.assertEqual(commands[0][:2], ("uv", "--no-config"))
        install = commands[1]
        self.assertIn("pip", install)
        self.assertIn("install", install)
        self.assertIn("--no-sources", install)
        self.assertNotIn("--dry-run", install)
        self.assertEqual(commands[2][:4], ("uv", "--no-config", "pip", "check"))
        self.assertIn("import nemo_safe_synthesizer", commands[3][-1])
        self.assertEqual(commands[4], ("/tmp/gate/.venv/bin/safe-synthesizer", "--help"))

    def test_cuda_commands_resolve_without_duplicate_install(self) -> None:
        plan = verify_wheel_install.build_plan(
            Path("/checkout/dist/nemo_safe_synthesizer-1.2.3-py3-none-any.whl"),
            verify_wheel_install.Variant.CU129,
        )

        commands = verify_wheel_install.build_commands(plan, python="3.13", venv=Path("/tmp/gate/.venv"))

        self.assertEqual(len(commands), 2)
        self.assertIn("--dry-run", commands[1])

    def test_pr_and_release_workflows_run_the_reusable_gate(self) -> None:
        ci_workflow = (REPO_ROOT / ".github" / "workflows" / "ci-checks.yml").read_text()
        change_action = (REPO_ROOT / ".github" / "actions" / "detect-changes" / "action.yml").read_text()
        release_workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text()
        invocation = "mise run release:verify-wheel -- dist/*.whl"

        self.assertIn("wheel-install:", ci_workflow)
        self.assertIn(invocation, ci_workflow)
        wheel_job = ci_workflow[ci_workflow.index("wheel-install:") : ci_workflow.index("unit-test:")]
        self.assertIn("needs: changes", wheel_job)
        self.assertIn("github.event_name == 'workflow_dispatch'", wheel_job)
        self.assertIn("needs.changes.outputs.ci == 'true'", wheel_job)
        self.assertNotIn("needs.changes.outputs.tests", wheel_job)
        self.assertIn("tools/release_version.py", change_action)
        self.assertIn("tools/verify_wheel_install.py", change_action)
        self.assertIn("wheel-install", ci_workflow[ci_workflow.index("ci-status:") :])
        self.assertIn(invocation, release_workflow)
        self.assertLess(release_workflow.index(invocation), release_workflow.index("actions/upload-artifact"))
        self.assertLess(release_workflow.index(invocation), release_workflow.index("twine upload"))


if __name__ == "__main__":
    unittest.main()
