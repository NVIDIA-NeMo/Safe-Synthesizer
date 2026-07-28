# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest
from click.testing import CliRunner
from packaging.requirements import InvalidRequirement

pytestmark = pytest.mark.unit


def _load_generator(root_path: Path) -> ModuleType:
    path = root_path / "tools" / "gen_cuda_deps.py"
    spec = importlib.util.spec_from_file_location("gen_cuda_deps", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator(pytestconfig: pytest.Config) -> ModuleType:
    return _load_generator(pytestconfig.rootpath)


CUDA_DEPS = """
base_runtime_deps = [
  { name = "faker" },
]
torch_runtime_deps = [
  { name = "accelerate" },
  "vllm==0.20.0; sys_platform == 'linux'",
]
cuda_runtime_deps = [
  { name = "flashinfer-python", version = "0.6.6", sys_platform = "linux", source_kind = "flashinfer" },
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
nvidia_index = "nvidia-pypi-public"

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

# The cpu torch entry keeps an explicit source_marker (tests the override path in
# DependencySpec.effective_source_marker); every other entry below derives its uv source
# marker from sys_platform/arch (tests the default-derivation path).
EXPECTED_CU132_DEPS = [
    "faker",
    "accelerate",
    "vllm==0.20.0; sys_platform == 'linux'",
    "flashinfer-python==0.6.6; sys_platform == 'linux'",
    "nvidia-cublas; sys_platform == 'linux'",
    "torch==3.0.0+cu132; sys_platform == 'linux'",
]

EXPECTED_CU132_TORCH_SOURCES = [
    {"index": "pytorch-cpu", "extra": "cpu", "marker": "sys_platform=='linux'"},
    {"index": "pytorch-cu132", "extra": "cu132", "marker": "sys_platform == 'linux'"},
    {"index": "pytorch-cu129", "extra": "cu129", "marker": "sys_platform == 'linux'"},
]


def _cuda_deps_dict() -> dict:
    """Parse CUDA_DEPS into a fresh, mutable dict for tests that need a targeted edit."""
    return tomllib.loads(CUDA_DEPS)


def test_build_cuda_pyproject_fragment_renders_cuda_variant_and_sources(tmp_path: Path, generator: ModuleType) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")

    generated = generator.build_cuda_pyproject_fragment(generator.load_cuda_deps_config(config_path))
    parsed = tomllib.loads(generated.text)

    assert parsed["project"]["optional-dependencies"]["cu132"] == EXPECTED_CU132_DEPS
    assert parsed["project"]["optional-dependencies"]["cpu"] == [
        "faker",
        "accelerate",
        "vllm==0.20.0; sys_platform == 'linux'",
        "torch==2.10.0+cpu; sys_platform == 'linux'",
    ]
    assert parsed["tool"]["uv"]["conflicts"] == [[{"extra": "cpu"}, {"extra": "cu132"}, {"extra": "cu129"}]]
    assert parsed["tool"]["uv"]["sources"]["torch"] == EXPECTED_CU132_TORCH_SOURCES
    assert parsed["tool"]["uv"]["sources"]["flashinfer-python"] == [
        {"index": "flashinfer-cu132", "extra": "cu132", "marker": "sys_platform == 'linux'"},
        {"index": "flashinfer-cu129", "extra": "cu129", "marker": "sys_platform == 'linux'"},
    ]
    assert parsed["tool"]["uv"]["sources"]["flashinfer-jit-cache"] == [
        {"index": "flashinfer-cu129", "extra": "cu129", "marker": "sys_platform == 'linux'"},
    ]
    assert parsed["tool"]["uv"]["sources"]["nvidia-cublas"] == [{"index": "nvidia-pypi-public"}]
    assert parsed["tool"]["uv"]["sources"]["nvidia-nvtx"] == [{"index": "nvidia-pypi-public"}]
    assert parsed["tool"]["uv"]["sources"]["nvidia-nvtx-cu12"] == [{"index": "pytorch-cu129"}]
    assert [index["name"] for index in parsed["tool"]["uv"]["index"]] == [
        "pytorch-cu132",
        "flashinfer-cu132",
        "pytorch-cu129",
        "flashinfer-cu129",
        "pytorch-cpu",
        "nvidia-pypi-public",
    ]


def test_apply_cuda_fragment_to_pyproject_splices_generated_sections(tmp_path: Path, generator: ModuleType) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = generator.build_cuda_pyproject_fragment(generator.load_cuda_deps_config(config_path))

    updated = generator.apply_cuda_fragment_to_pyproject(PYPROJECT, generated)
    parsed = tomllib.loads(updated)

    assert "# >>> BEGIN GENERATED CUDA RUNTIME EXTRAS - DO NOT EDIT <<<" in updated
    assert "# <<< END GENERATED CUDA RUNTIME EXTRAS - DO NOT EDIT >>>" in updated
    assert "# >>> BEGIN GENERATED CUDA UV CONFLICTS - DO NOT EDIT <<<" in updated
    assert "# <<< END GENERATED CUDA UV CONFLICTS - DO NOT EDIT >>>" in updated
    assert "# >>> BEGIN GENERATED CUDA UV SOURCES AND INDEXES - DO NOT EDIT <<<" in updated
    assert "# <<< END GENERATED CUDA UV SOURCES AND INDEXES - DO NOT EDIT >>>" in updated
    assert "gpu-old" not in parsed["project"]["optional-dependencies"]
    assert parsed["project"]["optional-dependencies"]["cu132"] == EXPECTED_CU132_DEPS
    assert parsed["tool"]["uv"]["required-version"] == ">=0.9.30, <0.10.0"
    assert parsed["tool"]["uv"]["sources"]["torch"] == EXPECTED_CU132_TORCH_SOURCES


def test_apply_cuda_fragment_to_pyproject_is_idempotent(tmp_path: Path, generator: ModuleType) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = generator.build_cuda_pyproject_fragment(generator.load_cuda_deps_config(config_path))

    updated = generator.apply_cuda_fragment_to_pyproject(PYPROJECT, generated)
    assert generator.apply_cuda_fragment_to_pyproject(updated, generated) == updated


def test_apply_cuda_fragment_to_pyproject_replaces_stale_marker_text(tmp_path: Path, generator: ModuleType) -> None:
    """Regenerating over a header written by an older script version must replace it, not leave a stray copy."""
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = generator.build_cuda_pyproject_fragment(generator.load_cuda_deps_config(config_path))
    updated = generator.apply_cuda_fragment_to_pyproject(PYPROJECT, generated)

    stale = updated.replace(
        "# Regenerate with: uv run --frozen tools/gen_cuda_deps.py cuda_deps.toml --pyproject pyproject.toml",
        "# Regenerate with: uv run --script tools/gen_cuda_deps.py cuda_deps.toml --pyproject pyproject.toml",
    )
    assert stale != updated

    reapplied = generator.apply_cuda_fragment_to_pyproject(stale, generated)

    assert reapplied == updated
    assert reapplied.count("Regenerate with") == 3


def test_run_generation_command_pyproject_check_reports_drift(tmp_path: Path, generator: ModuleType) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    pyproject_path = tmp_path / "pyproject.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    pyproject_path.write_text(PYPROJECT, encoding="utf-8")

    result = generator.run_generation_command(config_path, pyproject_path, check=True)

    assert result.status == generator.GenStatus.changed
    assert "pyproject.toml" in result.message


def test_run_generation_command_pyproject_check_reports_ok(tmp_path: Path, generator: ModuleType) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    pyproject_path = tmp_path / "pyproject.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = generator.build_cuda_pyproject_fragment(generator.load_cuda_deps_config(config_path))
    pyproject_path.write_text(generator.apply_cuda_fragment_to_pyproject(PYPROJECT, generated), encoding="utf-8")

    result = generator.run_generation_command(config_path, pyproject_path, check=True)

    assert result.status == generator.GenStatus.ok
    assert "up to date" in result.message


def test_run_generation_command_skips_writing_current_pyproject(
    tmp_path: Path, generator: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    pyproject_path = tmp_path / "pyproject.toml"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    generated = generator.build_cuda_pyproject_fragment(generator.load_cuda_deps_config(config_path))
    pyproject_path.write_text(generator.apply_cuda_fragment_to_pyproject(PYPROJECT, generated), encoding="utf-8")

    def fail_write(*args: object, **kwargs: object) -> int:
        pytest.fail(f"Unexpected write to current pyproject.toml: {args!r}, {kwargs!r}")

    monkeypatch.setattr(Path, "write_text", fail_write)

    result = generator.run_generation_command(config_path, pyproject_path, check=False)

    assert result.status is generator.GenStatus.ok
    assert "up to date" in result.message


def test_cpu_pytorch_wheel_sources_are_linux_only(pytestconfig: pytest.Config, generator: ModuleType) -> None:
    config_path = pytestconfig.rootpath / "cuda_deps.toml"
    generated = generator.build_cuda_pyproject_fragment(generator.load_cuda_deps_config(config_path))
    sources = tomllib.loads(generated.text)["tool"]["uv"]["sources"]

    for package in ("torch", "torchaudio", "torchvision"):
        cpu_source = next(source for source in sources[package] if source["extra"] == "cpu")
        assert cpu_source == {"index": "pytorch-cpu", "extra": "cpu", "marker": "sys_platform == 'linux'"}


def test_repository_cuda_variant_dependencies_and_sources(pytestconfig: pytest.Config, generator: ModuleType) -> None:
    config_path = pytestconfig.rootpath / "cuda_deps.toml"
    generated = generator.build_cuda_pyproject_fragment(generator.load_cuda_deps_config(config_path))
    parsed = tomllib.loads(generated.text)

    assert "vllm==0.26.0+cu129; sys_platform == 'linux'" in parsed["project"]["optional-dependencies"]["cu129"]
    assert "vllm==0.26.0; sys_platform == 'linux'" in parsed["project"]["optional-dependencies"]["cu130"]
    assert parsed["tool"]["uv"]["sources"]["vllm"] == [
        {"index": "vllm-v0-26-0-cu129", "marker": "sys_platform == 'linux'", "extra": "cu129"}
    ]
    assert parsed["tool"]["uv"]["sources"]["flashinfer-jit-cache"] == [
        {"index": "flashinfer-jit-cache-cu129", "marker": "sys_platform == 'linux'", "extra": "cu129"},
        {"index": "flashinfer-jit-cache-cu130", "marker": "sys_platform == 'linux'", "extra": "cu130"},
    ]
    assert parsed["tool"]["uv"]["sources"]["nvidia-cublas"] == [{"index": "nvidia-pypi-public"}]
    indexes = {index["name"]: index["url"] for index in parsed["tool"]["uv"]["index"]}
    assert indexes["flashinfer-jit-cache-cu129"] == "https://flashinfer.ai/whl/cu129"
    assert indexes["flashinfer-jit-cache-cu130"] == "https://flashinfer.ai/whl/cu130"


def test_build_cuda_installer_fragment_renders_runtime_index_arrays(
    pytestconfig: pytest.Config, generator: ModuleType
) -> None:
    config = generator.load_cuda_deps_config(pytestconfig.rootpath / "cuda_deps.toml")
    vllm_url = next(index.url for index in config.indexes if index.name.startswith("vllm-"))

    generated = generator.build_cuda_installer_fragment(config)

    assert "readonly -a CUDA_INDEXES_CPU=(https://download.pytorch.org/whl/cpu)" in generated.text
    assert "readonly -a CUDA_INDEXES_CU129=(" in generated.text
    assert vllm_url in generated.text
    assert "readonly -a CUDA_INDEXES_CU130=(" in generated.text
    assert "https://pypi.nvidia.com" in generated.text
    flashinfer_cu129 = config.cuda_indexes.flashinfer.url.format(extra="cu129")
    pytorch_cu129 = config.cuda_indexes.pytorch.url.format(extra="cu129")
    assert generated.text.index(flashinfer_cu129) < generated.text.index(pytorch_cu129) < generated.text.index(vllm_url)


def test_apply_cuda_fragment_to_installer_replaces_generated_block(
    pytestconfig: pytest.Config, generator: ModuleType
) -> None:
    config = generator.load_cuda_deps_config(pytestconfig.rootpath / "cuda_deps.toml")
    vllm_url = next(index.url for index in config.indexes if index.name.startswith("vllm-"))
    generated = generator.build_cuda_installer_fragment(config)
    current = (pytestconfig.rootpath / "install_nss.sh").read_text(encoding="utf-8")
    stale = current.replace(vllm_url, f"{vllm_url}-stale")

    updated = generator.apply_cuda_fragment_to_installer(stale, generated)

    assert updated == current
    assert generator.apply_cuda_fragment_to_installer(updated, generated) == updated


def test_apply_cuda_fragment_to_installer_rejects_duplicate_markers(
    pytestconfig: pytest.Config, generator: ModuleType
) -> None:
    config = generator.load_cuda_deps_config(pytestconfig.rootpath / "cuda_deps.toml")
    generated = generator.build_cuda_installer_fragment(config)
    current = (pytestconfig.rootpath / "install_nss.sh").read_text(encoding="utf-8")
    duplicated = current.replace(
        generator.INSTALLER_INDEXES_END,
        f"{generator.INSTALLER_INDEXES_END}\n{generator.INSTALLER_INDEXES_BEGIN}\n"
        f"stale\n{generator.INSTALLER_INDEXES_END}",
    )

    with pytest.raises(ValueError, match="exactly one generated CUDA installer index block"):
        generator.apply_cuda_fragment_to_installer(duplicated, generated)


def test_run_generation_command_installer_check_reports_drift(
    tmp_path: Path, pytestconfig: pytest.Config, generator: ModuleType
) -> None:
    config_text = (pytestconfig.rootpath / "cuda_deps.toml").read_text(encoding="utf-8")
    config_data = tomllib.loads(config_text)
    vllm_index = next(index for index in config_data["indexes"] if index["name"].startswith("vllm-"))
    old_url = vllm_index["url"]
    config_path = tmp_path / "cuda_deps.toml"
    config_path.write_text(config_text.replace(old_url, f"{old_url}-next"), encoding="utf-8")
    config = generator.load_cuda_deps_config(config_path)
    generated_pyproject = generator.build_cuda_pyproject_fragment(config)
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        generator.apply_cuda_fragment_to_pyproject(
            (pytestconfig.rootpath / "pyproject.toml").read_text(encoding="utf-8"),
            generated_pyproject,
        ),
        encoding="utf-8",
    )

    result = generator.run_generation_command(
        config_path,
        pyproject_path,
        check=True,
        installer_path=pytestconfig.rootpath / "install_nss.sh",
    )

    assert result.status == generator.GenStatus.changed
    assert "install_nss.sh" in result.message


def test_run_generation_command_validates_both_outputs_before_writing(
    tmp_path: Path, pytestconfig: pytest.Config, generator: ModuleType
) -> None:
    config_path = pytestconfig.rootpath / "cuda_deps.toml"
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(PYPROJECT, encoding="utf-8")
    installer_path = tmp_path / "install_nss.sh"
    installer_path.write_text("#!/usr/bin/env bash\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing generated CUDA installer index markers"):
        generator.run_generation_command(
            config_path,
            pyproject_path,
            check=False,
            installer_path=installer_path,
        )

    assert pyproject_path.read_text(encoding="utf-8") == PYPROJECT


def test_click_cli_updates_pyproject_and_checks_drift(
    tmp_path: Path, pytestconfig: pytest.Config, generator: ModuleType
) -> None:
    config_path = tmp_path / "cuda_deps.toml"
    pyproject_path = tmp_path / "pyproject.toml"
    installer_path = tmp_path / "install_nss.sh"
    config_path.write_text(CUDA_DEPS, encoding="utf-8")
    pyproject_path.write_text(PYPROJECT, encoding="utf-8")
    installer_path.write_text(
        (pytestconfig.rootpath / "install_nss.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    cli_paths = [
        str(config_path),
        "--pyproject",
        str(pyproject_path),
        "--installer",
        str(installer_path),
    ]

    stale_result = CliRunner().invoke(generator._cli, [*cli_paths, "--check"])

    assert stale_result.exit_code == 1

    result = CliRunner().invoke(generator._cli, cli_paths)

    assert result.exit_code == 0
    parsed = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    assert parsed["project"]["optional-dependencies"]["cu132"] == EXPECTED_CU132_DEPS
    assert parsed["tool"]["uv"]["sources"]["torch"] == EXPECTED_CU132_TORCH_SOURCES

    current_result = CliRunner().invoke(generator._cli, [*cli_paths, "--check"])

    assert current_result.exit_code == 0


def test_click_cli_checks_installer_in_working_directory_by_default(
    tmp_path: Path,
    pytestconfig: pytest.Config,
    generator: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pyproject_path = tmp_path / "pyproject.toml"
    pyproject_path.write_text(
        (pytestconfig.rootpath / "pyproject.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (tmp_path / "install_nss.sh").write_text(
        (pytestconfig.rootpath / "install_nss.sh").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(
        generator._cli,
        [
            str(pytestconfig.rootpath / "cuda_deps.toml"),
            "--pyproject",
            str(pyproject_path),
            "--check",
        ],
    )

    assert result.exit_code == 0
    assert "Generated CUDA installer indexes in install_nss.sh are up to date" in result.output


def test_load_cuda_deps_config_rejects_missing_managed_extra(generator: ModuleType) -> None:
    data = _cuda_deps_dict()
    data["managed_extras"] = ["cpu"]

    with pytest.raises(generator.ValidationError, match="managed_extras"):
        generator.CudaDepsConfig.model_validate(data)


def test_load_cuda_deps_config_rejects_mismatched_static_source_extra(generator: ModuleType) -> None:
    data = _cuda_deps_dict()
    data["sources"]["cpu"]["torch"][0]["extra"] = "cu129"

    with pytest.raises(generator.ValidationError, match="does not match"):
        generator.CudaDepsConfig.model_validate(data)


def test_static_source_group_sets_its_extra(generator: ModuleType) -> None:
    data = _cuda_deps_dict()
    del data["sources"]["cpu"]["torch"][0]["extra"]

    generated = generator.build_cuda_pyproject_fragment(generator.CudaDepsConfig.model_validate(data))
    parsed = tomllib.loads(generated.text)

    assert parsed["tool"]["uv"]["sources"]["torch"][0]["extra"] == "cpu"


def test_load_cuda_deps_config_rejects_invalid_structured_specifier(generator: ModuleType) -> None:
    data = _cuda_deps_dict()
    data["torch_runtime_deps"][0]["specifier"] = "=>1"

    with pytest.raises(generator.ValidationError, match="Invalid specifier"):
        generator.CudaDepsConfig.model_validate(data)


def test_build_cuda_pyproject_fragment_rejects_invalid_raw_requirement(generator: ModuleType) -> None:
    data = _cuda_deps_dict()
    data["base_runtime_deps"][0]["name"] = "not @@@ invalid"

    with pytest.raises(InvalidRequirement):
        generator.build_cuda_pyproject_fragment(generator.CudaDepsConfig.model_validate(data))


def test_collect_uv_indexes_rejects_conflicting_duplicate_index(generator: ModuleType) -> None:
    data = _cuda_deps_dict()
    data["indexes"].append({"name": "pytorch-cu132", "url": "https://example.invalid/conflict", "explicit": True})

    with pytest.raises(ValueError, match="Conflicting uv index definition"):
        generator.build_cuda_pyproject_fragment(generator.CudaDepsConfig.model_validate(data))


def test_build_cuda_pyproject_fragment_rejects_unknown_source_index(generator: ModuleType) -> None:
    data = _cuda_deps_dict()
    data["cpu"]["dependencies"][0]["index"] = "missing-index"

    with pytest.raises(ValueError, match="unknown indexes"):
        generator.build_cuda_pyproject_fragment(generator.CudaDepsConfig.model_validate(data))
