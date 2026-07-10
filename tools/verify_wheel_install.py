#!/usr/bin/env -S uv run --no-project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify a built wheel from a clean end-user environment."""

import os
import shlex
import subprocess
import sys
import tempfile
from argparse import ArgumentParser
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DEFAULT_INDEX = "https://pypi.org/simple"
PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PYTORCH_CU129_INDEX = "https://download.pytorch.org/whl/cu129"
FLASHINFER_CU129_INDEX = "https://flashinfer.ai/whl/cu129"
VLLM_CU129_INDEX = "https://wheels.vllm.ai/ee0da84ab9e04ac7610e28580af62c365e898389/cu129"
UV_BASE = ("uv", "--no-config")
VERSION_IMPORT_CHECK = (
    "from importlib.metadata import version; "
    "import nemo_safe_synthesizer; "
    "from nemo_safe_synthesizer.package_info import __version__; "
    "installed = version('nemo-safe-synthesizer'); "
    "assert __version__ == installed, f'{__version__} != {installed}'"
)


class Variant(StrEnum):
    """Supported end-user installation variants."""

    CPU = "cpu"
    CU129 = "cu129"


@dataclass(frozen=True)
class VerificationPlan:
    """Commands required to verify one built-wheel installation variant."""

    variant: Variant
    requirement: str
    index_args: tuple[str, ...]
    resolve_only: bool


def sanitized_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the subprocess environment used by the isolated resolver."""
    original = os.environ if source is None else source
    clean = {
        key: value
        for key, value in original.items()
        if (not key.startswith("UV_") or key == "UV_CACHE_DIR") and key not in {"PYTHONPATH", "VIRTUAL_ENV"}
    }
    clean.update({"UV_NO_CONFIG": "1", "UV_NO_SOURCES": "1"})
    return clean


def build_plan(wheel: Path, variant: Variant) -> VerificationPlan:
    """Build the verification plan for one installation variant."""
    wheel_url = wheel.resolve().as_uri()
    if variant is Variant.CPU:
        return VerificationPlan(
            variant=variant,
            requirement=f"nemo-safe-synthesizer[cpu,engine] @ {wheel_url}",
            index_args=(
                "--default-index",
                DEFAULT_INDEX,
                "--index",
                PYTORCH_CPU_INDEX,
                "--index-strategy",
                "unsafe-best-match",
            ),
            resolve_only=False,
        )
    return VerificationPlan(
        variant=variant,
        requirement=f"nemo-safe-synthesizer[cu129,engine] @ {wheel_url}",
        index_args=(
            "--default-index",
            DEFAULT_INDEX,
            "--index",
            FLASHINFER_CU129_INDEX,
            "--index",
            PYTORCH_CU129_INDEX,
            "--index",
            VLLM_CU129_INDEX,
            "--index-strategy",
            "unsafe-best-match",
        ),
        resolve_only=True,
    )


def build_commands(plan: VerificationPlan, *, python: str, venv: Path) -> tuple[tuple[str, ...], ...]:
    """Build the uv and installed-environment commands for a plan."""
    venv_python = venv / "bin" / "python"
    create = (*UV_BASE, "venv", "--clear", "--python", python, str(venv))
    install = (
        *UV_BASE,
        "pip",
        "install",
        "--python",
        str(venv_python),
        "--no-sources",
        *plan.index_args,
        *(("--dry-run",) if plan.resolve_only else ()),
        plan.requirement,
    )
    if plan.resolve_only:
        return create, install
    return (
        create,
        install,
        (*UV_BASE, "pip", "check", "--python", str(venv_python)),
        (str(venv_python), "-c", VERSION_IMPORT_CHECK),
        (str(venv / "bin" / "safe-synthesizer"), "--help"),
    )


def _run(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> None:
    sys.stdout.write(f"+ {shlex.join(command)}\n")
    sys.stdout.flush()
    subprocess.run(command, cwd=cwd, env=env, check=True)


def verify_plan(plan: VerificationPlan, *, python: str, env: Mapping[str, str]) -> None:
    """Execute one verification plan in a temporary directory."""
    with tempfile.TemporaryDirectory(prefix=f"nss-wheel-{plan.variant}-") as temporary:
        root = Path(temporary)
        commands = build_commands(plan, python=python, venv=root / ".venv")
        sys.stdout.write(f"Verifying {plan.variant} wheel dependencies in {root}\n")
        for command in commands:
            _run(command, cwd=root, env=env)


def verify_wheel(wheel: Path, *, python: str, variants: Sequence[Variant]) -> None:
    """Verify a wheel against the requested end-user installation variants."""
    resolved_wheel = wheel.resolve()
    if not resolved_wheel.is_file() or resolved_wheel.suffix != ".whl":
        raise ValueError(f"Expected one built wheel file, got: {wheel}")
    env = sanitized_environment()
    for variant in variants:
        verify_plan(build_plan(resolved_wheel, variant), python=python, env=env)


def _parse_args(argv: Sequence[str] | None = None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Exact built wheel to verify")
    parser.add_argument("--python", default=os.environ.get("PYTHON_VERSION", "3.13"))
    parser.add_argument("--variant", action="append", choices=tuple(Variant))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the built-wheel verification gate."""
    args = _parse_args(argv)
    variants = tuple(Variant(value) for value in (args.variant or tuple(Variant)))
    try:
        verify_wheel(args.wheel, python=args.python, variants=variants)
    except ValueError as error:
        sys.stderr.write(f"verify-wheel-install: error: {error}\n")
        return 2
    except subprocess.CalledProcessError as error:
        return error.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
