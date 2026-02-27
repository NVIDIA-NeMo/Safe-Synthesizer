# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration management subcommands (validate, modify, create).

Each command loads a ``SafeSynthesizerParameters`` model, optionally applies
CLI overrides, and either validates, prints, or writes the result.
"""

from __future__ import annotations

import click
import rich

from ..config import SafeSynthesizerParameters
from ..configurator.pydantic_click_options import parse_overrides, pydantic_options
from .utils import CLI_NESTED_FIELD_SEPARATOR, PathT, merge_overrides


@click.group()
def config():
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
def validate(config_path: PathT, **kwargs):
    """Validate a Safe Synthesizer configuration."""
    msg = f"Config {config_path}"
    overrides = parse_overrides(kwargs, field_sep=".")
    my_config = merge_overrides(config_path, overrides)

    click.echo(f"{msg} \n{my_config.model_dump_json(indent=2)} \n is valid!")
    return


@config.command()
@click.option(
    "--config",
    "config_path",
    required=False,
    type=str,
    help="path to a yaml config file",
)
@click.option("--output", required=False, default=None, help="validate config and exit")
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def modify(config_path: PathT, output: str, **kwargs):
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
def create(output: str, **kwargs):
    """Create a new Safe Synthesizer configuration."""
    overrides = parse_overrides(kwargs)
    my_config = merge_overrides(None, overrides)

    if output:
        my_config.to_yaml(output, exclude_unset=False)
        click.echo(f"config written to {output}")
    else:
        rich.print(f"{my_config.model_dump_json(indent=2, exclude_unset=False)}")
