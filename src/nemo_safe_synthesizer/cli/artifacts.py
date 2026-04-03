# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry points for artifact management.

Provides CLI commands for inspecting and cleaning up artifact directories
produced by Safe Synthesizer runs.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import click

from .artifact_structure import BoundDir, PathT, Workdir


@click.group(invoke_without_command=True)
@click.pass_context
def artifacts(ctx: click.Context):
    """Artifact management commands."""
    pass


@artifacts.command()
@click.option(
    "--artifact-path",
    type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True),
    help="Path to the artifact directory.",
)
@click.option("--dry-run", is_flag=True, help="Dry run the command.")
@click.option("--caches-only", is_flag=True, help="Only clean caches.")
@click.option("--force", is_flag=True, help="Force clean.")
def clean(ctx: click.Context, artifact_path: PathT | None, dry_run: bool, caches_only: bool, force: bool):
    """Clean artifacts in a Workdir structure."""
    if artifact_path is None:
        artifact_path = Path("safe-synthesizer-artifacts")

    try:
        workdir = Workdir.from_path(Path(artifact_path))
    except ValueError as e:
        raise click.ClickException(str(e))

    # Determine what to clean
    if caches_only:
        cache_dir = workdir.train.cache
        if not isinstance(cache_dir, BoundDir):
            raise TypeError(f"Expected BoundDir, got {type(cache_dir)}")
        target = cache_dir.path
        item_name = "cache"
    else:
        target = workdir.run_dir
        item_name = "all artifacts (run directory)"

    if not target.exists():
        click.echo(f"Path does not exist, nothing to clean: {target}")
        return

    # Confirmation
    if not force and not dry_run:
        if not click.confirm(f"Are you sure you want to delete {item_name} at {target}?"):
            click.echo("Aborted.")
            return

    if dry_run:
        click.secho(f"[DRY RUN] Would delete: {target}", fg="yellow")
    else:
        click.echo(f"Cleaning {item_name}...")
        try:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
            click.secho(f"Successfully deleted: {target}", fg="green")
        except Exception as e:
            raise click.ClickException(f"Error deleting {target}: {e}")
