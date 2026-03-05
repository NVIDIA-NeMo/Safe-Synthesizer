<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# CLI Reference

NeMo Safe Synthesizer provides a command-line interface built with [Click](https://click.palletsprojects.com/).

## Top-Level Commands

```bash
safe-synthesizer --help
```

```text
Usage: safe-synthesizer [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  artifacts  Artifacts management commands.
  config     Manage Safe Synthesizer configurations.
  run        Run the Safe Synthesizer end-to-end pipeline.
```

## `run` -- Execute the Pipeline

The `run` command executes the Safe Synthesizer pipeline. Without a subcommand, it runs the full end-to-end pipeline.

### Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--config TEXT` | Path to a YAML config file | -- |
| `--url TEXT` | Dataset name, URL, or path to CSV | -- |
| `--artifact-path DIR` | Base directory for all runs | `./safe-synthesizer-artifacts` |
| `--run-path DIR` | Explicit path for this run's output | -- |
| `--output-file PATH` | Path to output CSV file | -- |
| `--log-format [json\|plain]` | Console log format | `plain` |
| `--wandb-mode [online\|offline\|disabled]` | WandB logging mode | `disabled` |
| `-v` / `-vv` | Verbose logging | -- |

### Subcommands

- `run train` -- Run only the training stage
- `run generate` -- Run only the generation stage (requires a trained adapter)

## `config` -- Manage Configurations

```bash
safe-synthesizer config --help
```

| Subcommand | Description |
|------------|-------------|
| `config validate` | Validate a configuration file |
| `config modify` | Modify a configuration file |

## `artifacts` -- Artifacts Management

```bash
safe-synthesizer artifacts --help
```

<!-- TODO: Document artifacts subcommands -->

## Environment Variables

| Variable | Description |
|----------|-------------|
| `NSS_ARTIFACTS_PATH` | Default artifact path |
| `NSS_LOG_FORMAT` | Default log format (`json` or `plain`) |
| `NSS_LOG_FILE` | Default log file path |
| `NSS_DATASET_REGISTRY` | Dataset registry YAML path/URL |
| `HF_HOME` | Hugging Face cache directory (default `~/.cache/huggingface`) |
| `HF_HUB_OFFLINE` | Set to `1` to error instead of downloading models |
| `LOCAL_FILES_ONLY` | Set to `true` to skip network downloads (Unsloth, GLiNER only) |
| `VLLM_ATTENTION_BACKEND` | Override vLLM attention backend |
| `VLLM_CACHE_ROOT` | vLLM model cache directory |
| `WANDB_MODE` | WandB mode |
| `WANDB_PROJECT` | WandB project name |
| `WANDB_API_KEY` | WandB API key |
| `SAFE_SYNTHESIZER_CPU_COUNT` | Number of CPU processes for NER (default: `max(1, cpu_count - 1)`, further limited by record count) |
| `NIM_ENDPOINT_URL` | NIM endpoint for column classification / external PII detection |
| `NIM_API_KEY` | NIM API key |
| `NIM_MODEL_ID` | Model ID for column classification (default: `qwen/qwen2.5-coder-32b-instruct`) |
