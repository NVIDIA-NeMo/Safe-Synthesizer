<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Installation

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) -- Python package manager (>=0.9.14, <0.10.0)
- Git

## Quick Start

### 1. Bootstrap development tools

Install `uv`, `ruff`, `ty`, `yq`, and other development tools:

```bash
make bootstrap-tools
```

### 2. Install the package

Bootstrap the project with your desired dependency set:

=== "CPU (macOS / Linux without GPU)"

    ```bash
    make bootstrap-nss cpu
    ```

=== "CUDA 12.8 (Linux with NVIDIA GPU)"

    ```bash
    make bootstrap-nss cuda
    ```

=== "Engine only (no torch/training)"

    ```bash
    make bootstrap-nss engine
    ```

=== "Dev only (minimal)"

    ```bash
    make bootstrap-nss dev
    ```

## Verifying the Installation

After bootstrapping, verify the CLI is available:

```bash
safe-synthesizer --help
```

You should see:

```text
Usage: safe-synthesizer [OPTIONS] COMMAND [ARGS]...

  NeMo Safe Synthesizer command-line interface. This application is used to
  run the Safe Synthesizer pipeline. It can be used to train a model, generate
  synthetic data, and evaluate the synthetic data. It can also be used to
  modify a config file.

Options:
  --help  Show this message and exit.

Commands:
  artifacts  Artifacts management commands.
  config     Manage Safe Synthesizer configurations.
  run        Run the Safe Synthesizer end-to-end pipeline.
```

## Next Steps

Head to the [Quick Start](quickstart.md) guide to run your first pipeline.
