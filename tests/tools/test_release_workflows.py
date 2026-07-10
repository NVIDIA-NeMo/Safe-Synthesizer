# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import yaml

WORKFLOW = Path(__file__).parents[2] / ".github" / "workflows" / "container-build.yml"


def test_container_tags_use_pep440_versions() -> None:
    workflow = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)
    steps = workflow["jobs"]["build"]["steps"]
    metadata_step = next(step for step in steps if step.get("name") == "Extract image metadata")
    tags = metadata_step["with"]["tags"].splitlines()

    assert "type=pep440,pattern={{version}}-${{ matrix.variant }}" in tags
    assert "type=pep440,pattern={{major}}.{{minor}}-${{ matrix.variant }}" in tags
    assert not any(tag.startswith("type=semver") for tag in tags)
