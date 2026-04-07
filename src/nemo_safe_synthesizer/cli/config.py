# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI entry points for configuration management.

Each command loads or creates a ``SafeSynthesizerParameters`` model, optionally applies
CLI overrides, and either validates, prints, or writes the result.
"""

from __future__ import annotations

from pathlib import Path

import click
import rich

from ..config import SafeSynthesizerParameters
from ..configurator.pydantic_click_options import parse_overrides, pydantic_options
from .utils import CLI_NESTED_FIELD_SEPARATOR, merge_overrides


@click.group()
def config() -> None:
    """Manage Safe Synthesizer configurations."""
    pass


@config.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=str,
    help="path to a yaml config file",
)
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def validate(config_path: str | Path, **kwargs) -> None:
    """Validate a Safe Synthesizer configuration."""
    overrides = parse_overrides(kwargs)
    my_config = merge_overrides(config_path, overrides)

    click.echo(my_config.model_dump_json(indent=2))
    click.echo(f"Config {config_path} is valid!", err=True)


@config.command()
@click.option(
    "--config",
    "config_path",
    required=False,
    type=str,
    help="path to a yaml config file",
)
@click.option("--output", required=False, default=None, help="write modified config to this path")
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def modify(config_path: str | Path, output: str, **kwargs) -> None:
    """Modify a Safe Synthesizer configuration."""
    overrides = parse_overrides(kwargs)
    my_config = merge_overrides(config_path, overrides)

    if output:
        my_config.to_yaml(output)
        click.echo(f"Modified config written to {output}")
    else:
        rich.print(f"{my_config.model_dump_json(indent=2, exclude_unset=False)}")


@config.command()
@click.option(
    "--output",
    "-o",
    required=False,
    default=None,
    type=click.Path(exists=False, dir_okay=False, file_okay=True, resolve_path=True),
    help="path to the output yaml config file",
)
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def create(output: str, **kwargs) -> None:
    """Create a new Safe Synthesizer configuration."""
    overrides = parse_overrides(kwargs)
    my_config = merge_overrides(None, overrides)

    if output:
        my_config.to_yaml(output, exclude_unset=False)
        click.echo(f"config written to {output}")
    else:
        rich.print(f"{my_config.model_dump_json(indent=2, exclude_unset=False)}")
