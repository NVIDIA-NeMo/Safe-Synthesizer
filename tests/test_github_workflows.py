# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import yaml

pytestmark = pytest.mark.unit


def test_gpu_e2e_optional_failure_is_reported(pytestconfig: pytest.Config) -> None:
    workflow_path = pytestconfig.rootpath / ".github" / "workflows" / "gpu-tests.yml"
    workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
    e2e_job = workflow["jobs"]["gpu-e2e-test"]

    assert e2e_job["outputs"] == {
        "cu129_result": "${{ steps.record_result.outputs.cu129_result }}",
        "cu130_result": "${{ steps.record_result.outputs.cu130_result }}",
    }

    e2e_step = next(step for step in e2e_job["steps"] if step.get("id") == "e2e_tests")
    assert e2e_step["continue-on-error"] == "${{ !matrix.required }}"

    record_step = next(step for step in e2e_job["steps"] if step.get("id") == "record_result")
    assert record_step["if"] == "always()"

    status_job = workflow["jobs"]["gpu-ci-status"]
    status_script = next(step["run"] for step in status_job["steps"] if step["name"] == "Check job results")
    for result in (
        "${{ needs.gpu-e2e-test.result }}",
        "${{ needs.gpu-e2e-test.outputs.cu129_result }}",
        "${{ needs.gpu-e2e-test.outputs.cu130_result }}",
    ):
        assert status_script.count(result) >= 2
