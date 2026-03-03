<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Reference

CLI commands, environment variables, logging, experiment tracking, and
parameter precedence. For synthesis parameter tables, see
[Configuration](configuration.md). For running the pipeline, see
[Pipeline Stages](pipeline-stages.md).

---

## CLI Commands

NeMo Safe Synthesizer provides a command-line interface built with
[Click](https://click.palletsprojects.com/). All commands are accessed
through `safe-synthesizer`.

```bash
safe-synthesizer --help
```

### `run` -- Execute the Pipeline

Without a subcommand, runs the full end-to-end pipeline (data processing,
optional PII replacement, training, generation, evaluation).

```bash
safe-synthesizer run --config config.yaml --url data.csv
```

#### Common Options

These options apply to `run`, `run train`, and `run generate`. Options
with `--` as default are required (no built-in default).

| Option | Env var | Default | Description |
|--------|---------|---------|-------------|
| `--config` | `NSS_CONFIG` | -- | Path to YAML config file |
| `--url` | -- | -- | Dataset path, URL, or registry name |
| `--artifact-path` | `NSS_ARTIFACTS_PATH` | `./safe-synthesizer-artifacts` | Base directory for all runs |
| `--run-path` | -- | -- | Explicit run directory (for `run generate`, must point to an existing trained run) |
| `--output-file` | -- | -- | Path to output CSV file |
| `--log-format` | `NSS_LOG_FORMAT` | auto | Console log format -- auto-detects from TTY (`plain` if interactive, `json` otherwise) |
| `--log-file` | `NSS_LOG_FILE` | -- | Log file path (defaults to run directory) |
| `--log-color` / `--no-log-color` | `NSS_LOG_COLOR` | auto | Colorize console output (auto-detected from TTY) |
| `--wandb-mode` | `NSS_WANDB_MODE` | `disabled` | WandB mode (`online`, `offline`, `disabled`) |
| `--wandb-project` | `NSS_WANDB_PROJECT` | -- | WandB project name |
| `--dataset-registry` | `NSS_DATASET_REGISTRY` | -- | Dataset registry YAML path/URL |
| `-v` / `-vv` | -- | -- | Verbose logging (`-v` debug, `-vv` debug + dependencies) |

#### Synthesis Parameter Overrides

Any synthesis parameter can be overridden on the command line using
`--section__field` syntax:

```bash
safe-synthesizer run \
  --config config.yaml \
  --url data.csv \
  --training__learning_rate 0.001 \
  --generation__num_records 5000 \
  --training__batch_size 2
```

These overrides take precedence over YAML config values. Use double
underscores (`__`) to separate nested sections. Run
`safe-synthesizer run --help` to see all available overrides.

!!! note
    Parameters that accept `"auto"` (like `training.rope_scaling_factor` or
    `training.use_unsloth`) cannot be set to `"auto"` via CLI flags --
    Click parses them as their typed value (float, bool, etc.) and rejects
    the string `"auto"`. To use `"auto"`, omit the flag entirely (the
    default kicks in) or set it in your YAML config.

### `run train`

Train only -- saves the adapter without generating or evaluating.

```bash
safe-synthesizer run train --config config.yaml --url data.csv
```

Accepts the same common options as `run`.

### `run generate`

Generate only -- requires a previously trained adapter.

```bash
safe-synthesizer run generate \
  --config config.yaml \
  --url data.csv \
  --auto-discover-adapter

# Or specify an explicit run path
safe-synthesizer run generate \
  --config config.yaml \
  --url data.csv \
  --run-path ./safe-synthesizer-artifacts/myconfig---mydata/2026-01-15T12:00:00
```

| Option | Description |
|--------|-------------|
| `--auto-discover-adapter` | Find the latest trained adapter in the artifact directory |
| `--run-path` | Explicit path to a previous run's output directory |
| `--wandb-resume-job-id` | WandB run ID to resume (or path to file containing the ID) |

Accepts the same common options and synthesis parameter overrides as `run`.

---

### `config` -- Manage Configurations

#### `config validate`

Validate a configuration file and display the merged parameters.
Fields set to `"auto"` remain as `"auto"` -- auto-resolution happens
at runtime during `process_data()`, not at validation time. Several auto parameters require the dataset for resolution.

```bash
safe-synthesizer config validate --config config.yaml
```

| Option | Description |
|--------|-------------|
| `--config` | Path to YAML config file (required) |

Also accepts synthesis parameter overrides (`--training__learning_rate`, etc.)
to test how overrides merge with the config file.

#### `config modify`

Modify a configuration and optionally save the result:

```bash
safe-synthesizer config modify --config config.yaml --training__learning_rate 0.001 --output modified.yaml
```

| Option | Description |
|--------|-------------|
| `--config` | Path to YAML config file (optional -- omit to build from overrides only) |
| `--output` | Path to write modified YAML config (prints JSON to stdout if omitted) |

#### `config create`

Create a new configuration from defaults:

```bash
safe-synthesizer config create --output config.yaml
safe-synthesizer config create --training__pretrained_model "TinyLlama/TinyLlama-1.1B-Chat-v1.0" --output config.yaml
```

| Option | Description |
|--------|-------------|
| `--output` / `-o` | Path to write YAML config (prints JSON to stdout if omitted) |

---

### `artifacts` -- Artifact Management

#### `artifacts clean`

Delete artifacts from a previous run:

```bash
safe-synthesizer artifacts clean --artifact-path ./safe-synthesizer-artifacts
safe-synthesizer artifacts clean --caches-only   # training cache only
safe-synthesizer artifacts clean --dry-run        # preview what would be deleted
```

| Option | Description |
|--------|-------------|
| `--artifact-path` | Path to artifact directory (defaults to `./safe-synthesizer-artifacts`) |
| `--dry-run` | Preview deletions without actually deleting |
| `--caches-only` | Only delete the training cache, keep everything else |
| `--force` | Skip confirmation prompt |

---

## Environment Variables

Environment variables control infrastructure settings -- artifact paths,
logging, model caching, and service endpoints. They do not set synthesis
parameters (use YAML, CLI flags, or the SDK for those).

### NSS env vars

| Variable | CLI flag | Purpose |
|----------|----------|---------|
| `NSS_CONFIG` | `--config` | Path to YAML config file |
| `NSS_ARTIFACTS_PATH` | `--artifact-path` | Default artifact path |
| `NSS_LOG_FORMAT` | `--log-format` | Log format (`json` or `plain`) |
| `NSS_LOG_FILE` | `--log-file` | Log file path |
| `NSS_LOG_COLOR` | `--log-color` / `--no-log-color` | Colorize console output (auto-detected from TTY) |
| `NSS_DATASET_REGISTRY` | `--dataset-registry` | Dataset registry YAML path/URL |
| `NSS_WANDB_MODE` | `--wandb-mode` | WandB mode (alias for `WANDB_MODE`) |
| `NSS_WANDB_PROJECT` | `--wandb-project` | WandB project name (alias for `WANDB_PROJECT`) |
| `NIM_ENDPOINT_URL` | -- | LLM endpoint for PII column classification |
| `NIM_API_KEY` | -- | API key (optional -- only for direct endpoints) |
| `NIM_MODEL_ID` | -- | Column classification model ID |
| `LOCAL_FILES_ONLY` | -- | Set to `true` for offline mode (Unsloth, GLiNER) |
| `SAFE_SYNTHESIZER_CPU_COUNT` | -- | NER CPU processes |

### Third-party env vars

| Variable | Read by | Purpose |
|----------|---------|---------|
| `HF_HOME` | Hugging Face Hub | Cache directory for model downloads |
| `HF_HUB_OFFLINE` | Hugging Face Hub | Set to `1` to error instead of downloading |
| `VLLM_ATTENTION_BACKEND` | vLLM | Override attention backend |
| `VLLM_CACHE_ROOT` | vLLM | vLLM internal cache directory (defaults to `~/.cache/vllm`) |
| `WANDB_MODE` | WandB | Mode (`online`, `offline`, `disabled`) |
| `WANDB_PROJECT` | WandB | Project name |
| `WANDB_API_KEY` | WandB | API key for authentication |

---

## Logging and Experiment Tracking

### Log Format

| Method | Setting |
|--------|---------|
| CLI | `--log-format json` or `--log-format plain` |
| Environment | `NSS_LOG_FORMAT=json` |

Verbosity: `-v` for debug, `-vv` for debug + dependencies.

### WandB Integration

WandB is configured via CLI flags or environment variables -- not in the YAML
config file.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --config config.yaml \
      --url data.csv \
      --wandb-mode online \
      --wandb-project my-experiments
    ```

=== "SDK"

    ```python
    import os
    import wandb

    os.environ["WANDB_API_KEY"] = "your-api-key"  # pragma: allowlist secret
    wandb.init(project="my-experiments", mode="online")

    synthesizer = SafeSynthesizer(config).with_data_source("data.csv")
    synthesizer.run()
    ```

    Unlike the CLI, the SDK does not auto-initialize WandB. You must call
    `wandb.init(...)` before `synthesizer.run()`.

=== "Environment"

    ```bash
    export WANDB_API_KEY="your-api-key"  # pragma: allowlist secret
    export WANDB_PROJECT="my-experiments"
    export NSS_WANDB_MODE="online"
    ```

    These environment variables are read by the CLI only. SDK users must
    call `wandb.init(...)` explicitly.

---

## Precedence

Synthesis parameters (`training.learning_rate`, `generation.num_records`, etc.)
and infrastructure settings follow separate resolution paths.

### Synthesis parameters

1. CLI flags (`--training__learning_rate 0.001`)
2. Dataset registry overrides
3. YAML config file
4. Model defaults

### Infrastructure settings

Artifact path, logging, WandB:

1. CLI flags (`--artifact-path`, `--log-format`, etc.)
2. Environment variables (`NSS_ARTIFACTS_PATH`, `NSS_LOG_FORMAT`, etc.)
3. Built-in defaults

---

## SDK Methods

| Method | Purpose |
|--------|---------|
| `SafeSynthesizer(config)` | Create builder |
| `.with_data_source(df_or_url)` | Set data source |
| `.with_train(**kwargs)` | Configure training |
| `.with_generate(**kwargs)` | Configure generation |
| `.with_evaluate(**kwargs)` | Configure evaluation |
| `.with_replace_pii(**kwargs)` | Configure PII replacement |
| `.with_differential_privacy(**kwargs)` | Configure DP |
| `.with_time_series(**kwargs)` | Configure time series |
| `.with_data(**kwargs)` | Configure data params |
| `.synthesize()` | Enable synthesis (set automatically by `with_train`/`with_generate`) |
| `.run()` | Execute full pipeline |
| `.process_data()` | Data processing only |
| `.train()` | Training only |
| `.generate()` | Generation only |
| `.evaluate()` | Evaluation only |
| `.load_from_save_path()` | Resume from saved run |
| `.save_results()` | Write outputs |
