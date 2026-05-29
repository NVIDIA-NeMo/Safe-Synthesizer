# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark vLLM generation across structured-output methods.

This benchmark intentionally invokes the CLI in subprocesses. It is meant for
comparing real generation runs against an already-trained adapter, not for unit
testing the private backend helpers.

Run with the defaults used during local SmolLM3 DP experiments:

    uv run --frozen pytest tests/benchmarks/test_generation_structured_methods.py -m benchmark -n0 -s

Override inputs with:

    NSS_GENERATION_BENCHMARK_DATA_SOURCE=cleaned/amazon_reviews_25k.csv
    NSS_GENERATION_BENCHMARK_CONFIG=script/slurm/configs/smollm3-dp.yaml
    NSS_GENERATION_BENCHMARK_RUN_PATH=local_runs/smollm3-dp_amazon_reviews_25k_1_5609622_1
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import pytest

DEFAULT_DATA_SOURCE = "cleaned/amazon_reviews_25k.csv"
DEFAULT_CONFIG = "script/slurm/configs/smollm3-dp.yaml"
DEFAULT_RUN_PATH = "local_runs/smollm3-dp_amazon_reviews_25k_1_5609622_1"
DEFAULT_METHODS = ("unstructured", "regex", "json_schema", "structural_tag")
DEFAULT_TIMEOUT_SECONDS = 1800


def _timeout_seconds() -> float:
    return float(os.environ.get("NSS_GENERATION_BENCHMARK_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)))


@dataclass(frozen=True)
class GenerationMethod:
    name: str
    use_structured_generation: bool
    schema_method: str | None = None


@dataclass(frozen=True)
class GenerationBenchmarkResult:
    method: str
    command: list[str]
    duration_seconds: float
    output_file: str
    output_records: int
    log_file: str


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def _configured_path(env_name: str, default: str, root: Path) -> Path:
    path = Path(os.environ.get(env_name, default))
    return path if path.is_absolute() else root / path


def _selected_methods() -> list[GenerationMethod]:
    methods = {
        "unstructured": GenerationMethod("unstructured", use_structured_generation=False),
        "regex": GenerationMethod("regex", use_structured_generation=True, schema_method="regex"),
        "json_schema": GenerationMethod("json_schema", use_structured_generation=True, schema_method="json_schema"),
        "structural_tag": GenerationMethod(
            "structural_tag",
            use_structured_generation=True,
            schema_method="structural_tag",
        ),
    }
    requested = os.environ.get("NSS_GENERATION_BENCHMARK_METHODS")
    names = (
        DEFAULT_METHODS if requested is None else tuple(name.strip() for name in requested.split(",") if name.strip())
    )
    unknown = sorted(set(names).difference(methods))
    if unknown:
        raise ValueError(f"Unknown generation benchmark methods: {', '.join(unknown)}")
    return [methods[name] for name in names]


def _count_csv_records(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        line_count = sum(1 for _ in handle)
    return max(line_count - 1, 0)


def _tail(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _build_command(method: GenerationMethod, output_file: Path, root: Path) -> list[str]:
    data_source = _configured_path("NSS_GENERATION_BENCHMARK_DATA_SOURCE", DEFAULT_DATA_SOURCE, root)
    config = _configured_path("NSS_GENERATION_BENCHMARK_CONFIG", DEFAULT_CONFIG, root)
    run_path = _configured_path("NSS_GENERATION_BENCHMARK_RUN_PATH", DEFAULT_RUN_PATH, root)
    num_records = os.environ.get("NSS_GENERATION_BENCHMARK_NUM_RECORDS", "100")

    command = [
        sys.executable,
        "-m",
        "nemo_safe_synthesizer.cli.cli",
        "run",
        "generate",
        "--data-source",
        str(data_source),
        "--config",
        str(config),
        "--run-path",
        str(run_path),
        "--output-file",
        str(output_file),
        "--generation__num_records",
        num_records,
        "--generation__use_structured_generation",
        str(method.use_structured_generation).lower(),
    ]
    if method.use_structured_generation:
        command.extend(
            [
                "--generation__structured_generation_backend",
                "xgrammar",
                "--generation__structured_generation_schema_method",
                method.schema_method or "regex",
            ]
        )
    return command


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.requires_gpu
@pytest.mark.vllm
@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available")
@pytest.mark.timeout(_timeout_seconds() + 120)
@pytest.mark.parametrize("method", _selected_methods(), ids=lambda method: method.name)
def test_generation_structured_method_benchmark(
    method: GenerationMethod,
    tmp_path: Path,
    pytestconfig: pytest.Config,
) -> None:
    """Benchmark one generation structured-output method."""
    root = Path(pytestconfig.rootpath)
    for env_name, default in (
        ("NSS_GENERATION_BENCHMARK_DATA_SOURCE", DEFAULT_DATA_SOURCE),
        ("NSS_GENERATION_BENCHMARK_CONFIG", DEFAULT_CONFIG),
        ("NSS_GENERATION_BENCHMARK_RUN_PATH", DEFAULT_RUN_PATH),
    ):
        path = _configured_path(env_name, default, root)
        if not path.exists():
            pytest.skip(f"{env_name} path does not exist: {path}")

    timeout = _timeout_seconds()
    output_file = tmp_path / f"{method.name}.csv"
    log_file = tmp_path / f"{method.name}.log"
    command = _build_command(method, output_file, root)

    print(f"\nRunning {method.name} generation benchmark")
    print(f"vLLM version: {_package_version('vllm')}")
    print(f"XGrammar version: {_package_version('xgrammar')}")
    print(f"Command: {' '.join(command)}")
    print(f"Log file: {log_file}")

    start = time.perf_counter()
    try:
        with log_file.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"{method.name} generation benchmark timed out after {exc.timeout:.0f}s\n"
            f"Command: {' '.join(command)}\n\n{_tail(log_file)}"
        )
    duration = time.perf_counter() - start
    if completed.returncode != 0:
        pytest.fail(
            f"{method.name} generation benchmark failed with exit code {completed.returncode}\n"
            f"Command: {' '.join(command)}\n\n{_tail(log_file)}"
        )

    result = GenerationBenchmarkResult(
        method=method.name,
        command=command,
        duration_seconds=duration,
        output_file=str(output_file),
        output_records=_count_csv_records(output_file),
        log_file=str(log_file),
    )
    records_per_second = result.output_records / result.duration_seconds if result.duration_seconds > 0 else 0.0
    print(
        f"\nGeneration benchmark result: {result.method}: "
        f"{result.duration_seconds:.2f}s, {result.output_records} records, "
        f"{records_per_second:.3f} records/s"
    )

    summary_file = tmp_path / f"generation_structured_method_{method.name}_benchmark.json"
    summary_file.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    print(f"Benchmark summary JSON: {summary_file}")

    assert result.output_records > 0
