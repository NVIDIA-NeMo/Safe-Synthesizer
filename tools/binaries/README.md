<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Developer Tools and Setup

This folder holds setup scripts for binary tools and meta-configuration information needed
to develop with this repo. Services can keep tools for the service installed elsewhere, but
general purpose tools should be configured here.

The `bootstrap_tools.sh` file reads tool definitions from `tools.yaml` and installs them.
From the root of the repo, run `make bootstrap-tools` to install all tools.

## Quick Start

```bash
# From repo root
make bootstrap-tools

# Or directly
bash tools/binaries/bootstrap_tools.sh
```

## Architecture

Tools are defined in `tools.yaml` with two categories:

1. Simple binary tools - Defined entirely in YAML with download URLs
2. Custom scripts - Complex tools that need custom installation logic

### tools.yaml Structure

```yaml
tools:
  tool-name:
    version: "1.0.0"
    check_command: "tool-name --version"  # Command to verify installation
    url: "https://example.com/{version}/tool_{os}_{arch}"
    # Optional fields:
    os_map:           # Remap OS names in URL (e.g., darwin -> macos)
      darwin: "macos"
    arch_map:         # Remap arch names in URL
      amd64: "x86_64"
    extract: "path/to/binary"  # For tarballs: path to binary inside archive
    darwin_brew: "package"     # Use Homebrew on macOS instead of direct download
    bootstrap: true            # Install this tool first (before others)
    post_install: |            # Shell commands to run after install
      echo "Setup instructions..."

custom_scripts:
  - name: tool-name
    script: install_tool.sh
    bootstrap: true  # Run after all other tools (e.g., for uv)
```

### URL Placeholders

- `{version}` - Tool version from YAML
- `{os}` - Operating system: `darwin` or `linux` (can be remapped via `os_map`)
- `{arch}` - Architecture: `amd64` or `arm64` (can be remapped via `arch_map`)

## Supported Platforms

- macOS (darwin): arm64, amd64
- Linux: arm64, amd64

## Adding a New Tool

### Simple Binary Tool

Add to `tools.yaml`:

```yaml
tools:
  my-tool:
    version: "2.0.0"
    check_command: "my-tool --version"
    url: "https://github.com/org/my-tool/releases/download/v{version}/my-tool_{os}_{arch}"
```

### Tool with Custom Logic

For tools that need complex installation (shell hooks, version detection, etc.):

1. Create `install_my-tool.sh` in this directory
2. Add to `tools.yaml`:

```yaml
custom_scripts:
  - name: my-tool
    script: install_my-tool.sh
```

## Guidelines

- Pin the version explicitly in `tools.yaml`
- Ensure your tool works on both macOS (arm64) and Linux (amd64)
- Tools install to `$HOME/.local/bin` by default
- For Homebrew-only tools on macOS, use the `darwin_brew` field

## Files

| File | Purpose |
|------|---------|
| `tools.yaml` | Tool definitions (versions, URLs, options) |
| `bootstrap_tools.sh` | Main installer script |
| `install_yq.sh` | Bootstrap yq (needed to parse YAML) |
| `install_direnv.sh` | Custom script for direnv (shell hooks) |
| `install_uv.sh` | Custom script for uv (version from pyproject.toml) |
| `install_gnutar.sh` | macOS-only gnu-tar install |
| `defs.sh` | Common environment variables (OS, ARCH) |
| `common_functions.sh` | Shared shell functions |
| `test_tool_install.sh` | Docker-based installation test |

## Testing

Test installation in a Linux container:

```bash
bash test_tool_install.sh
```

This runs the bootstrap in a Docker container to verify Linux compatibility.
