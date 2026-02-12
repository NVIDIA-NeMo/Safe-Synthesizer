# Developer Tools and Setup

This folder holds setup scripts for tools that require custom installation logic beyond what
[mise](https://mise.jdx.dev/) handles. General-purpose tools are defined in the repo-root
`mise.toml` and installed via `mise install` (or `make bootstrap-tools`).

## Quick Start

```bash
# From repo root — installs all tools defined in mise.toml
make bootstrap-tools

# Or directly
mise install
```

## Tool Versions

Tool versions are pinned in `mise.toml` at the repo root. To update a tool version, edit that
file and run `mise install`.

## Custom Scripts

Some tools require platform-specific or post-install logic that mise does not cover:

| File | Purpose |
|------|---------|
| `install_direnv.sh` | Creates `.envrc` and configures shell hooks for direnv |
| `install_gnutar.sh` | macOS-only: installs GNU tar via Homebrew |

## Guidelines

- Pin tool versions in `mise.toml`, not in individual scripts
- Ensure tools work on both macOS (arm64) and Linux (amd64)
- Only add a custom script here if mise cannot handle the installation
