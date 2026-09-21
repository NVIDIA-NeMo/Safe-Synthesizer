# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tomllib

import pytest
from packaging.requirements import Requirement

pytestmark = pytest.mark.unit


def test_engine_uses_python_314_compatible_outlines_core(pytestconfig: pytest.Config) -> None:
    pyproject_path = pytestconfig.rootpath / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    engine_dependencies = pyproject["project"]["optional-dependencies"]["engine"]
    dependency_names = {Requirement(dependency).name for dependency in engine_dependencies}

    assert "outlines-core==0.2.14" in engine_dependencies
    assert "outlines" not in dependency_names
