# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import click

from .artifacts import artifacts
from .config import config
from .run import run


@click.group()
def cli():
    """NeMo Safe Synthesizer command-line interface.
    This application is used to run the Safe Synthesizer pipeline.
    It can be used to train a model, generate synthetic data, and evaluate the synthetic data.
    It can also be used to modify a config file.
    """
    pass


cli.add_command(config)
cli.add_command(run)
cli.add_command(artifacts)
if __name__ == "__main__":
    cli()
