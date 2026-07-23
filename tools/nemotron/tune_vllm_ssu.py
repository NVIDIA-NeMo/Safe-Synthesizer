#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tune vLLM 0.24.0 selective-state-update for Nemotron 3 Nano on A100."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

VLLM_VERSION = "0.24.0"
VLLM_REVISION = "ee0da84ab9e04ac7610e28580af62c365e898389"  # pragma: allowlist secret
VLLM_TAG = "v0.24.0"
VLLM_REMOTE = "https://github.com/vllm-project/vllm.git"
DEVICE_NAME = "NVIDIA_A100-SXM4-80GB"
CONFIG_FILENAME = f"headdim=80,dstate=128,device_name={DEVICE_NAME},cache_dtype=float32.json"
UPSTREAM_FILES = (
    "benchmarks/kernels/benchmark_selective_state_update.py",
    "tests/kernels/mamba/utils.py",
)
BLOCK_SIZES = {4, 8, 16, 32, 64, 128}
NUM_WARPS = {1, 2, 4, 8}
TUNING_NHEADS = 128
TUNING_BATCH_SIZES = (1, 2, 8, 16, 32, 64, 128, 256, 512)
EXPECTED_EFFECTIVE_BATCHES = frozenset(batch * TUNING_NHEADS for batch in TUNING_BATCH_SIZES)


def _run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def _git_output(repository: Path, *arguments: str) -> str:
    result = _run(["git", "-C", str(repository), *arguments], capture=True)
    return result.stdout.strip()


def _git_file(repository: Path, relative_path: str) -> str:
    result = _run(
        ["git", "-C", str(repository), "show", f"{VLLM_REVISION}:{relative_path}"],
        capture=True,
    )
    return result.stdout


def _default_cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "nemo-safe-synthesizer"


def expected_config_path(folder: Path) -> Path:
    """Return the exact path vLLM uses for this kernel target."""
    return folder.expanduser().resolve() / CONFIG_FILENAME


def shell_export(folder: Path) -> str:
    """Return a shell-safe vLLM tuning-folder export."""
    return f"export VLLM_TUNED_CONFIG_FOLDER={shlex.quote(str(folder.expanduser().resolve()))}"


def validate_config(path: Path) -> dict[str, object]:
    """Validate the filename and launch-config schema accepted by vLLM 0.24.0."""
    if path.name != CONFIG_FILENAME:
        raise ValueError(f"expected filename {CONFIG_FILENAME!r}, found {path.name!r}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or not isinstance(payload.get("triton_version"), str):
        raise ValueError("config must contain a string triton_version")
    entries = {key: value for key, value in payload.items() if key != "triton_version"}
    expected_keys = {str(batch) for batch in EXPECTED_EFFECTIVE_BATCHES}
    if set(entries) != expected_keys:
        missing = sorted(expected_keys - set(entries), key=int)
        unexpected = sorted(set(entries) - expected_keys)
        raise ValueError(f"config effective-batch grid mismatch; missing={missing}, unexpected={unexpected}")
    for effective_batch, config in entries.items():
        _validate_entry(effective_batch, config)
    return payload


def _validate_entry(effective_batch: str, config: object) -> None:
    if not effective_batch.isdigit() or int(effective_batch) <= 0:
        raise ValueError(f"invalid effective-batch key: {effective_batch!r}")
    if not isinstance(config, dict) or set(config) != {"BLOCK_SIZE_M", "num_warps"}:
        raise ValueError(f"invalid launch config for effective batch {effective_batch}")
    launch_config = cast(dict[str, object], config)
    if launch_config["BLOCK_SIZE_M"] not in BLOCK_SIZES:
        raise ValueError(f"invalid BLOCK_SIZE_M for effective batch {effective_batch}")
    if launch_config["num_warps"] not in NUM_WARPS:
        raise ValueError(f"invalid num_warps for effective batch {effective_batch}")


def _prepare_source(cache: Path, source: Path | None) -> tuple[Path, str]:
    repository = source.expanduser().resolve() if source else cache / f"vllm-{VLLM_TAG}"
    if source is None and not (repository / ".git").exists():
        repository.mkdir(parents=True, exist_ok=True)
        _run(["git", "-C", str(repository), "init"])
        _run(["git", "-C", str(repository), "remote", "add", "origin", VLLM_REMOTE])
    if not _has_revision(repository):
        if source is not None:
            raise RuntimeError(f"{repository} does not contain pinned vLLM revision {VLLM_REVISION}")
        _run(["git", "-C", str(repository), "fetch", "--depth=1", "origin", f"refs/tags/{VLLM_TAG}"])
    revision = _git_output(repository, "rev-parse", f"{VLLM_REVISION}^{{commit}}")
    if revision != VLLM_REVISION:
        raise RuntimeError(f"unexpected vLLM revision: {revision}")
    return repository, revision


def _has_revision(repository: Path) -> bool:
    if not (repository / ".git").exists():
        return False
    result = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{VLLM_REVISION}^{{commit}}"],
        capture_output=True,
    )
    return result.returncode == 0


def _stage_upstream_benchmark(repository: Path, stage: Path) -> Path:
    for relative_path in UPSTREAM_FILES:
        destination = stage / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_git_file(repository, relative_path))
        _add_package_markers(destination.parent, stage)
    return stage / UPSTREAM_FILES[0]


def _add_package_markers(folder: Path, root: Path) -> None:
    while folder != root:
        (folder / "__init__.py").touch()
        folder = folder.parent


def _check_runtime(allow_active_gpu: bool) -> None:
    import torch
    import vllm

    if vllm.__version__ != VLLM_VERSION:
        raise RuntimeError(f"expected vLLM {VLLM_VERSION}, found {vllm.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_name = torch.cuda.get_device_name(0).replace(" ", "_")
    if device_name != DEVICE_NAME:
        raise RuntimeError(f"expected {DEVICE_NAME}, found {device_name}")
    if not allow_active_gpu:
        _refuse_active_compute_processes()


def _refuse_active_compute_processes() -> None:
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", maxsplit=1)[0]
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
        "-i",
        selector,
    ]
    processes = _run(command, capture=True).stdout.strip()
    if processes:
        raise RuntimeError(
            "refusing to tune while the selected GPU has active compute processes; "
            "use --allow-active-gpu only after confirming they may share the GPU:\n" + processes
        )


def _benchmark_command(benchmark: Path, output: Path, num_iters: int) -> list[str]:
    return [
        sys.executable,
        str(benchmark),
        "--dstate",
        "128",
        "--dtype",
        "bfloat16",
        "--mamba-ssm-cache-dtype",
        "float32",
        "--headdim",
        "80",
        "--ngroups",
        "8",
        "--batch-sizes",
        *(str(batch) for batch in TUNING_BATCH_SIZES),
        "--nheads",
        str(TUNING_NHEADS),
        "--num-iters",
        str(num_iters),
        "--save-configs",
        "--save-dir",
        str(output),
        "--results-file",
        str(output / "selective_state_update_A100-SXM4-80GB.txt"),
        "--compare",
        "--validate",
    ]


def tune(args: argparse.Namespace) -> None:
    output = args.output_dir.expanduser().resolve()
    _check_runtime(args.allow_active_gpu)
    output.mkdir(parents=True, exist_ok=True)
    repository, revision = _prepare_source(args.source_cache.expanduser(), args.source_dir)
    with tempfile.TemporaryDirectory(prefix="nss-vllm-ssu-") as temporary:
        stage = Path(temporary)
        benchmark = _stage_upstream_benchmark(repository, stage)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(stage)
        subprocess.run(_benchmark_command(benchmark, output, args.num_iters), check=True, env=environment)
    config = expected_config_path(output)
    validate_config(config)
    _emit(f"Validated {config}\nUpstream vLLM revision: {revision}\n{shell_export(output)}")


def _emit(message: str) -> None:
    sys.stdout.write(message + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    tune_parser = subparsers.add_parser("tune", help="run the pinned upstream GPU tuner")
    tune_parser.add_argument("--output-dir", type=Path, default=_default_cache_root() / "vllm-tuned-configs")
    tune_parser.add_argument("--source-cache", type=Path, default=_default_cache_root() / "sources")
    tune_parser.add_argument("--source-dir", type=Path, help="offline vLLM checkout containing the pinned revision")
    tune_parser.add_argument("--num-iters", type=int, default=100)
    tune_parser.add_argument("--allow-active-gpu", action="store_true")
    tune_parser.set_defaults(handler=tune)
    validate_parser = subparsers.add_parser("validate", help="validate an existing tuning folder")
    validate_parser.add_argument("folder", type=Path)
    validate_parser.set_defaults(handler=_validate_command)
    env_parser = subparsers.add_parser("env", help="print the runtime environment export")
    env_parser.add_argument("folder", type=Path)
    env_parser.set_defaults(handler=lambda args: _emit(shell_export(args.folder)))
    return parser


def _validate_command(args: argparse.Namespace) -> None:
    path = expected_config_path(args.folder)
    payload = validate_config(path)
    _emit(f"Validated {len(payload) - 1} effective-batch configs in {path}\n{shell_export(args.folder)}")


def main() -> None:
    args = _parser().parse_args()
    if getattr(args, "num_iters", 1) <= 0:
        raise SystemExit("--num-iters must be positive")
    try:
        args.handler(args)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        sys.stderr.write(f"Error: {error}\n")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
