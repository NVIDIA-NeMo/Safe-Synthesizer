#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Login-node CLI for NSS Slurm work: submit experiment arrays, build the shared image, download models.

Run it through the shell entry points (``submit_slurm_jobs.sh``, ``build_image.sh``,
``download_models.sh``), which go through ``nss_slurm.sh``. That launcher sources
``env_variables.sh`` and keeps uv's cache and Python on login-node local disk. The
compute-node steps (``build_image.sh --job/--container``, ``download_models.sh
--job/--container``, ``slurm_srun.sh``, ``slurm_nss_matrix.sh``) stay in bash because
some of them run before the image has Python.

This file uses only the standard library so it runs anywhere ``uv`` can start Python 3.11+.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ACCOUNT = "nemotron_data_dev"

# Lets jobs sit idle on the GPU for 15 minutes while loading data before the
# cluster's idle-GPU reaper may cancel them.
IDLE_EXEMPT_COMMENT = (
    '{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"15","reason":"data_loading",'
    '"description":"Data loading and preprocessing grace period for this job"}}'
)

DATASET_GROUPS: dict[str, tuple[str, ...]] = {
    "long": (
        "ai_generated_essays",
        "call_transcripts",
        "cover_type",
        "online_news_popularity",
        "car_accident",
        "diabetes",
    ),
    "short": (
        "adult",
        "amazon_reviews_25k",
        "aids_clinical_trials",
        "beijing",
        "bike_sales",
        "clinc_oos",
        "default",
        "dialogs",
        "EHR",
        "ecommerce_reviews",
        "magic",
        "ontonotes5_reduced",
        "patient_events",
        "project_management_sequences",
        "shoppers",
        "stack_exchange_data_dump",
    ),
}

_PRETRAINED_MODEL_RE = re.compile(r"^\s*pretrained_model:\s*[\"']?([^\"'\s#]+)", re.MULTILINE)


class CliError(Exception):
    """A user-facing error; ``main`` prints the message and exits with status 1."""


@dataclass(frozen=True)
class Settings:
    """Paths and defaults exported by ``env_variables.sh``."""

    user_name: str
    lustre_dir: Path
    shared_dir: Path
    nss_dir: Path
    slurm_dir: Path
    config_dir: Path
    base_log_dir: Path
    image: Path
    base_image: Path
    models_dir: Path
    account: str = DEFAULT_ACCOUNT
    default_configs: tuple[str, ...] = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Settings:
        """Read settings from the environment prepared by ``nss_slurm.sh``."""
        user_name = env.get("USER_NAME", "")
        if not user_name:
            raise CliError("USER_NAME is not set. Export your cluster user name first, e.g. export USER_NAME=jdoe")

        def path(key: str) -> Path:
            value = env.get(key, "")
            if not value:
                raise CliError(f"{key} is not set; run this through the script/slurm shell entry points")
            return Path(value)

        return cls(
            user_name=user_name,
            lustre_dir=path("LUSTRE_DIR"),
            shared_dir=path("NSS_SHARED_DIR"),
            nss_dir=path("NSS_DIR"),
            slurm_dir=path("NSS_SLURM_DIR"),
            config_dir=path("CONFIG_DIR"),
            base_log_dir=path("BASE_LOG_DIR"),
            image=path("NSS_IMAGE"),
            base_image=path("NSS_BASE_IMAGE"),
            models_dir=path("NSS_MODELS_DIR"),
            account=env.get("ACCOUNT") or DEFAULT_ACCOUNT,
            default_configs=split_csv(env.get("NSS_DEFAULT_CONFIGS", "")),
        )


@dataclass(frozen=True)
class SbatchCall:
    """One ``sbatch`` invocation: its argv and the extra environment it exports with ``--export=ALL``."""

    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)


# ── Pure helpers (unit tested) ──────────────────────────────────────────────


def split_csv(value: str) -> tuple[str, ...]:
    """Split a comma-separated list, dropping blanks and surrounding spaces."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


def resolve_datasets(dataset_urls: str | None, dataset_group: str | None) -> tuple[str, ...]:
    """Return the datasets from exactly one of ``--dataset-urls`` and ``--dataset-group``."""
    if bool(dataset_urls) == bool(dataset_group):
        raise CliError("Provide exactly one of --dataset-urls and --dataset-group.")
    if dataset_group:
        if dataset_group not in DATASET_GROUPS:
            raise CliError(f"Unknown dataset group {dataset_group!r}; choose from {', '.join(DATASET_GROUPS)}")
        return DATASET_GROUPS[dataset_group]
    datasets = split_csv(dataset_urls or "")
    if not datasets:
        raise CliError("--dataset-urls is empty.")
    return datasets


def pack_matrix(configs: Sequence[str], datasets: Sequence[str], runs: int) -> tuple[str, str]:
    """Return ``(PACKED_CONFIGS, PACKED_DATASETS)``: comma-separated lists with one entry per array task.

    Task order is config, then dataset, then run. ``slurm_nss_matrix.sh`` reads entry
    ``SLURM_ARRAY_TASK_ID`` from each list, and ``job_idx`` in run paths is that index.
    """
    if runs < 1:
        raise CliError(f"--runs must be at least 1, got {runs}")
    if not configs:
        raise CliError("No configs; pass --configs or set CONFIGS in env_variables.sh.")
    for name in (*configs, *datasets):
        if "," in name:
            raise CliError(f"Config and dataset names must not contain commas: {name!r}")
    packed_configs: list[str] = []
    packed_datasets: list[str] = []
    for config in configs:
        for dataset in datasets:
            for _ in range(runs):
                packed_configs.append(config)
                packed_datasets.append(dataset)
    return ",".join(packed_configs), ",".join(packed_datasets)


def array_spec(total_tasks: int, max_concurrent: int | None = None) -> str:
    """Return the ``sbatch --array`` value, with an optional ``%N`` limit on concurrently running tasks."""
    if total_tasks < 1:
        raise CliError("Nothing to submit.")
    spec = f"0-{total_tasks - 1}"
    if max_concurrent is not None:
        if max_concurrent < 1:
            raise CliError(f"--max-concurrent-slurm-jobs must be at least 1, got {max_concurrent}")
        spec += f"%{max_concurrent}"
    return spec


def image_name(python_version: str, lock_bytes: bytes) -> str:
    """Name an NSS image after its Python version and the first 12 hex digits of ``sha256(uv.lock)``."""
    return f"nss-slurm-py{python_version.strip()}-{hashlib.sha256(lock_bytes).hexdigest()[:12]}.sqsh"


def pinned_uv_release(mise_lock_text: str, platform: str = "linux-x64") -> tuple[str, str]:
    """Return ``(url, sha256)`` of the uv release that ``mise.lock`` pins for ``platform``."""
    data = tomllib.loads(mise_lock_text)
    for entry in data.get("tools", {}).get("uv", []):
        release = entry.get(f"platforms.{platform}")
        if release and release.get("url") and release.get("checksum"):
            return release["url"], str(release["checksum"]).removeprefix("sha256:")
    raise CliError(f"mise.lock has no uv release for {platform}")


def configured_models(config_texts: Sequence[str]) -> list[str]:
    """Return the sorted, unique ``training.pretrained_model`` values found in config YAML texts."""
    models = {match for text in config_texts for match in _PRETRAINED_MODEL_RE.findall(text)}
    return sorted(models)


def compare_image_metadata(image_python: str, image_lock: bytes, branch_python: str, branch_lock: bytes) -> str | None:
    """Check an image against this checkout; return a note when only ``uv.lock`` differs.

    A Python version mismatch cannot be fixed at job start, so it is an error. A
    ``uv.lock`` difference only means each job installs the difference.
    """
    if image_python.strip() != branch_python.strip():
        raise CliError(
            f"This checkout needs Python {branch_python.strip()}, but the image has {image_python.strip()}. "
            "Build a new image with build_image.sh."
        )
    if image_lock != branch_lock:
        return (
            "uv.lock differs from the image; each job will install the difference at start. "
            "For large dependency changes, build a new image with build_image.sh."
        )
    return None


def submit_common_args(settings: Settings, partition: str, exp_log_dir: Path, array: str, dry_run: bool) -> list[str]:
    """Return ``sbatch`` arguments shared by every experiment array."""
    args = [
        "sbatch",
        "--parsable",
        "--partition",
        partition,
        "--account",
        settings.account,
        "--comment",
        IDLE_EXEMPT_COMMENT,
        "--output",
        str(exp_log_dir / "slurm_%A_%a.out"),
        "--error",
        str(exp_log_dir / "slurm_%A_%a.err"),
        "--nodes",
        "1",
        "--ntasks-per-node",
        "1",
        "--cpus-per-task",
        "16",
        "--mem",
        "32G",
        "--gres",
        "gpu:1",
        "--array",
        array,
    ]
    if dry_run:
        args.append("--test-only")
    return args


def phase_args(
    common: Sequence[str],
    settings: Settings,
    phase: str,
    time_limit: str,
    dependency: str | None = None,
) -> list[str]:
    """Return the full ``sbatch`` argv for one phase (``train``, ``generate``, or ``end_to_end``)."""
    args = [*common, f"--time={time_limit}", "--job-name", f"nss_{phase}"]
    if dependency:
        # aftercorr: generate task N starts only after train task N succeeds.
        args.append(f"--dependency=aftercorr:{dependency}")
    args += [f"--export=ALL,NSS_PHASE={phase}", str(settings.slurm_dir / "slurm_srun.sh")]
    return args


def build_image_call(
    settings: Settings,
    name: str,
    uv_release: tuple[str, str],
    *,
    partition: str,
    time_limit: str,
    force: bool,
    wait: bool,
) -> SbatchCall:
    """Return the ``sbatch`` call that runs ``build_image.sh --job``."""
    log_dir = settings.base_log_dir / "image_builds"
    argv = [
        "sbatch",
        "--parsable",
        "--job-name",
        "nss_build_image",
        "--account",
        settings.account,
        "--partition",
        partition,
        "--time",
        time_limit,
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--cpus-per-task",
        "16",
        "--mem",
        "64G",
        "--output",
        str(log_dir / "build_%j.out"),
        "--error",
        str(log_dir / "build_%j.err"),
        "--export=ALL",
    ]
    if wait:
        argv.append("--wait")
    argv += [str(settings.slurm_dir / "build_image.sh"), "--job"]
    env = {
        "REPO_ROOT": str(settings.nss_dir),
        "NSS_IMAGES_DIR": str(settings.shared_dir / "images"),
        "NSS_IMAGE_NAME": name,
        "NSS_UV_URL": uv_release[0],
        "NSS_UV_SHA256": uv_release[1],
        "NSS_FORCE": "1" if force else "0",
    }
    return SbatchCall(argv=argv, env=env)


def download_models_call(settings: Settings, models: Sequence[str], *, partition: str, wait: bool) -> SbatchCall:
    """Return the ``sbatch`` call that runs ``download_models.sh --job``."""
    log_dir = settings.base_log_dir / "model_downloads"
    argv = [
        "sbatch",
        "--parsable",
        "--job-name",
        "nss_download_models",
        "--account",
        settings.account,
        "--partition",
        partition,
        "--time",
        "01:00:00",
        "--nodes",
        "1",
        "--ntasks",
        "1",
        "--cpus-per-task",
        "8",
        "--mem",
        "32G",
        "--output",
        str(log_dir / "download_%j.out"),
        "--error",
        str(log_dir / "download_%j.err"),
        "--export=ALL",
    ]
    if wait:
        argv.append("--wait")
    argv += [str(settings.slurm_dir / "download_models.sh"), "--job"]
    env = {"REPO_ROOT": str(settings.nss_dir), "NSS_DOWNLOAD_MODELS": " ".join(models)}
    return SbatchCall(argv=argv, env=env)


# ── Side effects ────────────────────────────────────────────────────────────


def read_image_metadata(image: Path) -> tuple[str, bytes]:
    """Return the ``.python-version`` text and ``uv.lock`` bytes recorded in an NSS image.

    Extracts only ``/opt/nss-image`` to a temporary directory on login-node local disk;
    the cluster's unsquashfs 4.4 has no ``-cat``.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "root"
        result = subprocess.run(
            ["unsquashfs", "-n", "-d", str(root), str(image), "/opt/nss-image"],
            capture_output=True,
            text=True,
            check=False,
        )
        meta = root / "opt" / "nss-image"
        if result.returncode != 0 or not (meta / ".python-version").is_file():
            raise CliError(f"Could not read /opt/nss-image from {image}; is it an image from build_image.sh?")
        return (meta / ".python-version").read_text(), (meta / "uv.lock").read_bytes()


def run_sbatch(call: SbatchCall, base_env: Mapping[str, str]) -> str:
    """Run ``sbatch`` and return its stdout (the job ID with ``--parsable``)."""
    result = subprocess.run(call.argv, env={**base_env, **call.env}, capture_output=True, text=True, check=False)
    if result.stderr:
        print(result.stderr.strip(), file=sys.stderr)
    if result.returncode != 0:
        raise CliError(f"sbatch failed with exit code {result.returncode}")
    return result.stdout.strip()


def run_sbatch_wait(call: SbatchCall, base_env: Mapping[str, str], log_hint: str) -> int:
    """Run ``sbatch --wait``: print the job ID as soon as it is known, then return the job's exit code."""
    proc = subprocess.Popen(call.argv, env={**base_env, **call.env}, stdout=subprocess.PIPE, text=True)
    assert proc.stdout is not None
    job_id = proc.stdout.readline().strip()
    print(f"Submitted job {job_id}; log: {log_hint.replace('%j', job_id)}", flush=True)
    proc.stdout.read()
    return proc.wait()


def cmd_submit(args: argparse.Namespace, settings: Settings, env: Mapping[str, str]) -> int:
    """Submit an experiment array (or a train array plus a dependent generate array)."""
    tokens = settings.lustre_dir / ".api_tokens.sh"
    if not tokens.is_file():
        raise CliError(f"{tokens} not found. Create it and define NSS_INFERENCE_KEY.")

    configs = split_csv(args.configs) if args.configs else settings.default_configs
    for config in configs:
        if not (settings.config_dir / f"{config}.yaml").is_file():
            raise CliError(f"Config not found: {settings.config_dir / f'{config}.yaml'}")
    datasets = resolve_datasets(args.dataset_urls, args.dataset_group)
    packed_configs, packed_datasets = pack_matrix(configs, datasets, args.runs)
    total_tasks = len(configs) * len(datasets) * args.runs
    array = array_spec(total_tasks, args.max_concurrent_slurm_jobs)

    if not settings.image.is_file():
        raise CliError(
            f"NSS image not found: {settings.image}. Build it with: bash {settings.slurm_dir}/build_image.sh"
        )
    print(f"Using container image: {settings.image}")
    image_python, image_lock = read_image_metadata(settings.image)
    note = compare_image_metadata(
        image_python,
        image_lock,
        (settings.nss_dir / ".python-version").read_text(),
        (settings.nss_dir / "uv.lock").read_bytes(),
    )
    if note:
        print(f"NOTE: {note}")

    wandb_project = args.wandb_project or args.exp_name
    exp_log_dir = settings.base_log_dir / args.exp_name
    job_env = {
        "ACCOUNT": settings.account,
        "EXP_NAME": args.exp_name,
        "WANDB_PROJECT": wandb_project,
        "PACKED_CONFIGS": packed_configs,
        "PACKED_DATASETS": packed_datasets,
    }
    print(
        f"Submitting {total_tasks} task(s): pipeline={args.pipeline_mode} configs={','.join(configs)} "
        f"datasets={','.join(datasets)} runs={args.runs} partition={args.partition} "
        f"exp_name={args.exp_name} wandb_project={wandb_project} array={array}"
    )
    if not args.dry_run:
        exp_log_dir.mkdir(parents=True, exist_ok=True)

    common = submit_common_args(settings, args.partition, exp_log_dir, array, args.dry_run)
    if args.pipeline_mode == "two_stage":
        train_id = run_sbatch(
            SbatchCall(phase_args(common, settings, "train", args.train_time_limit or args.time_limit), job_env), env
        )
        # With --test-only there is no train job ID to depend on, so the dry run checks generate on its own.
        generate_id = run_sbatch(
            SbatchCall(
                phase_args(
                    common,
                    settings,
                    "generate",
                    args.generate_time_limit or args.time_limit,
                    dependency=None if args.dry_run else train_id,
                ),
                job_env,
            ),
            env,
        )
        recorded_id = generate_id
        if not args.dry_run:
            print(f"train array: {train_id}  generate array: {generate_id}")
    else:
        recorded_id = run_sbatch(SbatchCall(phase_args(common, settings, "end_to_end", args.time_limit), job_env), env)

    if args.dry_run:
        print("Dry run completed without errors")
        return 0
    print(recorded_id)
    if args.job_id_file:
        job_id_file = Path(args.job_id_file)
        job_id_file.parent.mkdir(parents=True, exist_ok=True)
        job_id_file.write_text(f"{recorded_id}\n")
    print(f"Done submitting jobs to slurm cluster. Monitor with: squeue -u {settings.user_name}")
    return 0


def cmd_build_image(args: argparse.Namespace, settings: Settings, env: Mapping[str, str]) -> int:
    """Submit a CPU job that builds the shared NSS image for this checkout."""
    name = image_name((settings.nss_dir / ".python-version").read_text(), (settings.nss_dir / "uv.lock").read_bytes())
    final_image = settings.shared_dir / "images" / name
    if final_image.exists() and not args.force:
        print(f"Image already exists: {final_image} (use --force to rebuild)")
        return 0
    if not settings.base_image.is_file():
        raise CliError(f"Base image not found: {settings.base_image}")
    uv_release = pinned_uv_release((settings.nss_dir / "mise.lock").read_text())
    call = build_image_call(
        settings,
        name,
        uv_release,
        partition=args.partition,
        time_limit=args.time,
        force=args.force,
        wait=args.wait,
    )
    print(f"Building {final_image}\n  from checkout {settings.nss_dir}\n  base image   {settings.base_image}")
    if args.dry_run:
        print("--dry-run:", " ".join(call.argv))
        return 0
    log_hint = str(settings.base_log_dir / "image_builds" / "build_%j.out")
    (settings.base_log_dir / "image_builds").mkdir(parents=True, exist_ok=True)
    if args.wait:
        if run_sbatch_wait(call, env, log_hint) != 0 or not final_image.is_file():
            raise CliError(f"Image build failed; see {log_hint}")
        print(f"Image ready: {final_image}")
    else:
        job_id = run_sbatch(call, env)
        print(f"Submitted build job {job_id}; log: {log_hint.replace('%j', job_id)}")
    print("To use it, set NSS_IMAGE in env_variables.sh to that path.")
    return 0


def cmd_download_models(args: argparse.Namespace, settings: Settings, env: Mapping[str, str]) -> int:
    """Submit a CPU job that downloads models into the shared, read-only models directory."""
    models = list(args.models) or configured_models(
        [path.read_text() for path in sorted(settings.config_dir.glob("*.yaml"))]
    )
    if not models:
        raise CliError("No models given and none found in the configs.")
    try:
        settings.models_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CliError(f"Cannot create NSS_MODELS_DIR={settings.models_dir}: {exc}") from exc
    if not os.access(settings.models_dir, os.W_OK):
        raise CliError(f"Cannot write to NSS_MODELS_DIR={settings.models_dir}")
    if not settings.image.is_file():
        raise CliError(f"NSS image not found: {settings.image}; build it with build_image.sh")
    call = download_models_call(settings, models, partition=args.partition, wait=args.wait)
    print(f"Downloading to {settings.models_dir}: {' '.join(models)}")
    log_hint = str(settings.base_log_dir / "model_downloads" / "download_%j.out")
    (settings.base_log_dir / "model_downloads").mkdir(parents=True, exist_ok=True)
    if args.wait:
        if run_sbatch_wait(call, env, log_hint) != 0:
            raise CliError(f"Model download failed; see {log_hint}")
        print("Models ready.")
    else:
        job_id = run_sbatch(call, env)
        print(f"Submitted download job {job_id}; log: {log_hint.replace('%j', job_id)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for all subcommands."""
    parser = argparse.ArgumentParser(
        prog="nss_slurm",
        description="Submit NSS experiments, build the shared image, and download models on a Slurm cluster.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser(
        "submit",
        help="Submit Slurm array jobs for each config and dataset",
        description=(
            "Submit Slurm array jobs. Provide exactly one of --dataset-urls and --dataset-group. "
            "--time-limit applies to end_to_end mode; two_stage mode uses --train-time-limit and "
            "--generate-time-limit, which default to --time-limit."
        ),
    )
    submit.add_argument("-c", "--configs", help="comma-separated config names (default: CONFIGS in env_variables.sh)")
    submit.add_argument("-d", "--dataset-urls", help="comma-separated dataset names, URLs, or paths")
    submit.add_argument("-g", "--dataset-group", choices=sorted(DATASET_GROUPS), help="predefined dataset set")
    submit.add_argument("-r", "--runs", type=int, default=1, help="runs per dataset-config pair (default: 1)")
    submit.add_argument("-p", "--partition", default="polar4", help="Slurm partition(s) (default: polar4)")
    submit.add_argument("-e", "--exp-name", default="nss_exp", help="experiment name for logs and outputs")
    submit.add_argument(
        "-m", "--pipeline-mode", choices=["two_stage", "end_to_end"], default="end_to_end", help="pipeline mode"
    )
    submit.add_argument("-w", "--wandb-project", help="W&B project (default: --exp-name)")
    submit.add_argument("--max-concurrent-slurm-jobs", type=int, help="limit running tasks per array")
    submit.add_argument("-t", "--time-limit", default="04:00:00", help="time limit (default: 04:00:00)")
    submit.add_argument("--train-time-limit", help="two_stage train time limit")
    submit.add_argument("--generate-time-limit", help="two_stage generate time limit")
    submit.add_argument("--job-id-file", help="write the array ID to this file (generate array in two_stage mode)")
    submit.add_argument("--dry-run", action="store_true", help="validate with sbatch --test-only; submit nothing")
    submit.set_defaults(handler=cmd_submit)

    build = sub.add_parser("build-image", help="Build the shared NSS image for this checkout on a CPU node")
    build.add_argument("--partition", default="cpu_short", help="Slurm partition (default: cpu_short)")
    build.add_argument("--time", default="01:00:00", help="time limit (default: 01:00:00)")
    build.add_argument("--wait", action="store_true", help="wait for the build to finish")
    build.add_argument("--force", action="store_true", help="rebuild and replace an existing image of the same name")
    build.add_argument("--dry-run", action="store_true", help="print the sbatch command without submitting")
    build.set_defaults(handler=cmd_build_image)

    download = sub.add_parser("download-models", help="Download models into the shared models directory")
    download.add_argument("models", nargs="*", help="Hugging Face model IDs (default: models in configs/*.yaml)")
    download.add_argument("--partition", default="cpu_short", help="Slurm partition (default: cpu_short)")
    download.add_argument("--wait", action="store_true", help="wait for the download to finish")
    download.set_defaults(handler=cmd_download_models)
    return parser


def main(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Parse arguments, load settings from the environment, and run a subcommand."""
    environ = dict(os.environ if env is None else env)
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.from_env(environ)
        return args.handler(args, settings, environ)
    except CliError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
