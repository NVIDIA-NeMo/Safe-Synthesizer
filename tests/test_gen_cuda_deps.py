# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from click.testing import CliRunner


def _load_generator() -> ModuleType:
    path = Path(__file__).parents[1] / "tools" / "gen_cuda_deps.py"
    spec = importlib.util.spec_from_file_location("gen_cuda_deps", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = _load_generator()


CUDA_DEPS = """
base_runtime_deps = [
  { name = "faker" },
]
torch_runtime_deps = [
  { name = "accelerate" },
  "vllm==0.20.0; sys_platform == 'linux'",
]
cuda_runtime_deps = [
  { name = "flashinfer-python", version = "0.6.6", sys_platform = "linux", source_kind = "flashinfer", source_marker = "sys_platform=='linux'" },
  { name = "flashinfer-jit-cache", version = "0.6.6", local = "{torch_local_version}", sys_platform = "linux", source_kind = "flashinfer", variants = ["cu129"] },
  { name = "nvidia-cublas", sys_platform = "linux" },
]
torch_wheel_deps = [
  { name = "torch", version = "3.0.0", local = "{torch_local_version}", sys_platform = "linux", source_kind = "pytorch" },
]
managed_extras = ["cpu", "gpu-old", "cu129", "cu132"]
nvidia_cuda_libraries = [
  { name = "cublas", nvidia_package_suffix = "", index = "nvidia-pypi-public" },
  "nvtx",
]
conflict_extras = ["cpu"]

[cpu]
dependencies = [
  { name = "torch", version = "2.10.0", local = "cpu", sys_platform = "linux", index = "pytorch-cpu", source_marker = "sys_platform=='linux'" },
]

[[indexes]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[indexes]]
name = "nvidia-pypi-public"
url = "https://pypi.nvidia.com"
explicit = true

[cuda_indexes.pytorch]
name = "pytorch-{extra}"
url = "https://download.pytorch.org/whl/{extra}"
explicit = true

[cuda_indexes.flashinfer]
name = "flashinfer-{extra}"
url = "https://flashinfer.ai/whl/{extra}"
explicit = true

[sources.cpu]
torch = [
  { index = "pytorch-cpu", extra = "cpu", marker = "sys_platform=='linux'" },
]

[variants.cu132]
cuda_package_suffix = "cu13"
nvidia_package_suffix = ""

[variants.cu129]
cuda_package_suffix = "cu12"
"""


PYPROJECT = """
[project]
name = "demo"

[project.optional-dependencies]
cpu = [
  "old-cpu",
]
gpu-old = [
  "old-cuda",
]

[tool.uv]
required-version = ">=0.9.30, <0.10.0"
conflicts = [
  [
    { extra = "cpu" },
    { extra = "gpu-old" },
  ],
]

[tool.uv.sources]
torch = [
  { index = "old", extra = "gpu-old" },
]

[[tool.uv.index]]
name = "old"
url = "https://example.invalid"
explicit = true
"""


def test_build_cuda_pyproject_fragment_renders_cuda_variant_and_sources(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")

    generated = GENERATOR.build_cuda_pyproject_fragment(GENERATOR.load_cuda_deps_config(config_path))
    parsed = tomllib.loads(generated.text)

    assert parsed["project"]["optional-dependencies"]["cu132"] == [
        "faker",
        "accelerate",
        "vllm==0.20.0; sys_platform == 'linux'",
        "flashinfer-python==0.6.6; sys_platform == 'linux'",
        "nvidia-cublas; sys_platform == 'linux'",
        "torch==3.0.0+cu132; sys_platform == 'linux'",
    ]
    assert parsed["project"]["optional-dependencies"]["cpu"] == [
        "faker",
        "accelerate",
        "vllm==0.20.0; sys_platform == 'linux'",
        "torch==2.10.0+cpu; sys_platform == 'linux'",
    ]
    assert parsed["tool"]["uv"]["conflicts"] == [[{"extra": "cpu"}, {"extra": "cu132"}, {"extra": "cu129"}]]
    assert parsed["tool"]["uv"]["sources"]["torch"] == [
        {"index": "pytorch-cpu", "extra": "cpu", "marker": "sys_platform=='linux'"},
        {"index": "pytorch-cu132", "extra": "cu132"},
        {"index": "pytorch-cu129", "extra": "cu129"},
    ]
    assert parsed["tool"]["uv"]["sources"]["flashinfer-python"] == [
        {"index": "flashinfer-cu132", "extra": "cu132", "marker": "sys_platform=='linux'"},
        {"index": "flashinfer-cu129", "extra": "cu129", "marker": "sys_platform=='linux'"},
    ]
    assert parsed["tool"]["uv"]["sources"]["flashinfer-jit-cache"] == [{"index": "flashinfer-cu129", "extra": "cu129"}]
    assert parsed["tool"]["uv"]["sources"]["nvidia-cublas"] == [
        {"index": "nvidia-pypi-public", "extra": "cu132"},
        {"index": "nvidia-pypi-public", "extra": "cu129"},
    ]
    assert parsed["tool"]["uv"]["sources"]["nvidia-nvtx"] == [{"index": "pytorch-cu132"}]
    assert parsed["tool"]["uv"]["sources"]["nvidia-nvtx-cu12"] == [{"index": "pytorch-cu129"}]
    assert [index["name"] for index in parsed["tool"]["uv"]["index"]] == [
        "pytorch-cu132",
        "flashinfer-cu132",
        "pytorch-cu129",
        "flashinfer-cu129",
        "pytorch-cpu",
        "nvidia-pypi-public",
    ]


def test_apply_cuda_fragment_to_pyproject_splices_generated_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = GENERATOR.build_cuda_pyproject_fragment(GENERATOR.load_cuda_deps_config(config_path))

    updated = GENERATOR.apply_cuda_fragment_to_pyproject(PYPROJECT, generated)
    parsed = tomllib.loads(updated)

    assert "# >>> BEGIN GENERATED CUDA RUNTIME EXTRAS - DO NOT EDIT <<<" in updated
    assert "# <<< END GENERATED CUDA RUNTIME EXTRAS - DO NOT EDIT >>>" in updated
    assert "# >>> BEGIN GENERATED CUDA UV CONFLICTS - DO NOT EDIT <<<" in updated
    assert "# <<< END GENERATED CUDA UV CONFLICTS - DO NOT EDIT >>>" in updated
    assert "# >>> BEGIN GENERATED CUDA UV SOURCES AND INDEXES - DO NOT EDIT <<<" in updated
    assert "# <<< END GENERATED CUDA UV SOURCES AND INDEXES - DO NOT EDIT >>>" in updated
    assert "gpu-old" not in parsed["project"]["optional-dependencies"]
    assert parsed["project"]["optional-dependencies"]["cu132"] == [
        "faker",
        "accelerate",
        "vllm==0.20.0; sys_platform == 'linux'",
        "flashinfer-python==0.6.6; sys_platform == 'linux'",
        "nvidia-cublas; sys_platform == 'linux'",
        "torch==3.0.0+cu132; sys_platform == 'linux'",
    ]
    assert parsed["tool"]["uv"]["required-version"] == ">=0.9.30, <0.10.0"
    assert parsed["tool"]["uv"]["sources"]["torch"] == [
        {"index": "pytorch-cpu", "extra": "cpu", "marker": "sys_platform=='linux'"},
        {"index": "pytorch-cu132", "extra": "cu132"},
        {"index": "pytorch-cu129", "extra": "cu129"},
    ]


def test_apply_cuda_fragment_to_pyproject_is_idempotent(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = GENERATOR.build_cuda_pyproject_fragment(GENERATOR.load_cuda_deps_config(config_path))

    updated = GENERATOR.apply_cuda_fragment_to_pyproject(PYPROJECT, generated)
    assert GENERATOR.apply_cuda_fragment_to_pyproject(updated, generated) == updated


def test_run_generation_command_check_reports_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    output_path = tmp_path / "generated.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    output_path.write_text("# stale\n", encoding="utf-8")

    result = GENERATOR.run_generation_command(config_path, output_path, check=True)

    assert result.status == GENERATOR.GenStatus.changed
    assert "differ" in result.message


def test_run_generation_command_check_reports_ok(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    output_path = tmp_path / "generated.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = GENERATOR.build_cuda_pyproject_fragment(GENERATOR.load_cuda_deps_config(config_path))
    output_path.write_text(generated.text, encoding="utf-8")

    result = GENERATOR.run_generation_command(config_path, output_path, check=True)

    assert result.status == GENERATOR.GenStatus.ok
    assert "up to date" in result.message


def test_run_generation_command_check_reports_missing_output(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    output_path = tmp_path / "generated.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")

    result = GENERATOR.run_generation_command(config_path, output_path, check=True)

    assert result.status == GENERATOR.GenStatus.changed
    assert "does not exist" in result.message


def test_run_generation_command_check_requires_output_or_pyproject(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")

    result = GENERATOR.run_generation_command(config_path, None, check=True)

    assert result.status == GENERATOR.GenStatus.error
    assert "requires --output" in result.message


def test_run_generation_command_pyproject_check_reports_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    pyproject_path = tmp_path / "pyproject.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    pyproject_path.write_text(PYPROJECT, encoding="utf-8")

    result = GENERATOR.run_generation_command(config_path, None, check=True, pyproject_path=pyproject_path)

    assert result.status == GENERATOR.GenStatus.changed
    assert "pyproject.toml" in result.message


def test_run_generation_command_pyproject_check_reports_ok(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    pyproject_path = tmp_path / "pyproject.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = GENERATOR.build_cuda_pyproject_fragment(GENERATOR.load_cuda_deps_config(config_path))
    pyproject_path.write_text(GENERATOR.apply_cuda_fragment_to_pyproject(PYPROJECT, generated), encoding="utf-8")

    result = GENERATOR.run_generation_command(config_path, None, check=True, pyproject_path=pyproject_path)

    assert result.status == GENERATOR.GenStatus.ok
    assert "up to date" in result.message


def test_click_cli_writes_parseable_output_and_checks_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    output_path = tmp_path / "generated.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")

    stale_result = CliRunner().invoke(GENERATOR._cli, [str(config_path), "--output", str(output_path), "--check"])

    assert stale_result.exit_code == 1
    assert not output_path.exists()

    result = CliRunner().invoke(GENERATOR._cli, [str(config_path), "--output", str(output_path)])

    assert result.exit_code == 0
    parsed = tomllib.loads(output_path.read_text(encoding="utf-8"))
    assert parsed["project"]["optional-dependencies"]["cu132"] == [
        "faker",
        "accelerate",
        "vllm==0.20.0; sys_platform == 'linux'",
        "flashinfer-python==0.6.6; sys_platform == 'linux'",
        "nvidia-cublas; sys_platform == 'linux'",
        "torch==3.0.0+cu132; sys_platform == 'linux'",
    ]
    assert parsed["tool"]["uv"]["sources"]["torch"] == [
        {"index": "pytorch-cpu", "extra": "cpu", "marker": "sys_platform=='linux'"},
        {"index": "pytorch-cu132", "extra": "cu132"},
        {"index": "pytorch-cu129", "extra": "cu129"},
    ]

    current_result = CliRunner().invoke(GENERATOR._cli, [str(config_path), "--output", str(output_path), "--check"])

    assert current_result.exit_code == 0


def test_load_cuda_deps_config_rejects_missing_managed_extra(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS.replace('"cpu", "gpu-old", "cu129", "cu132"', '"cpu"'), encoding="utf-8")

    with pytest.raises(Exception, match="managed_extras"):
        GENERATOR.load_cuda_deps_config(config_path)


def test_load_cuda_deps_config_rejects_invalid_structured_specifier(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(
        CUDA_DEPS.replace('{ name = "accelerate" }', '{ name = "accelerate", specifier = "=>1" }'), encoding="utf-8"
    )

    with pytest.raises(Exception, match="Invalid specifier"):
        GENERATOR.load_cuda_deps_config(config_path)


def test_build_cuda_pyproject_fragment_rejects_invalid_raw_requirement(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS.replace('"faker"', '"not @@@ invalid"'), encoding="utf-8")

    with pytest.raises(Exception):
        GENERATOR.build_cuda_pyproject_fragment(GENERATOR.load_cuda_deps_config(config_path))


def test_collect_uv_indexes_rejects_conflicting_duplicate_index(tmp_path: Path) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(
        CUDA_DEPS
        + """
[[indexes]]
name = "pytorch-cu132"
url = "https://example.invalid/conflict"
explicit = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Conflicting uv index definition"):
        GENERATOR.build_cuda_pyproject_fragment(GENERATOR.load_cuda_deps_config(config_path))
