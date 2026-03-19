<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# cli

Click CLI entry point with Pydantic-to-Click option mapping. Maps `SafeSynthesizerParameters` fields to typed CLI options and unifies config sources.

## Command structure

- cli (root group) → `config`, `run`, `artifacts`
- run (subgroup, `invoke_without_command=True`) → default = full pipeline; subcommands: `train`, `generate`
- config → `validate`, `modify`, `create`
- artifacts → `clean`

## pydantic_options()

From `configurator.pydantic_click_options`: decorator that recursively generates Click options from a Pydantic model. Nested `BaseModel` fields become `--parent__child` options. Uses `field_separator` (default `__`) to flatten dotted paths.

Critical: `parse_overrides(kwargs)` must use the same `field_sep` as `pydantic_options` uses for `field_separator` — otherwise `--data__holdout=0.1` won't become `{"data": {"holdout": 0.1}}`.

## common_run_options()

Shared decorator in `run.py` that adds `--config`, `--data-source`, `--artifact-path`, `--run-path`, `--output-file`, `--log-format`, `--log-file`, `--log-color`, `--wandb-mode`, `--wandb-project`, `--dataset-registry`, and `-v`. Applied to `run`, `run train`, and `run generate`. Env var handling lives in `CLISettings`, not in Click's `envvar=` — keeps precedence logic in one place.

## Settings precedence

1. CLI overrides — passed to `CLISettings.from_cli_kwargs(**kwargs)`
2. Env vars — pydantic-settings via `AliasChoices` (e.g. `NSS_ARTIFACTS_PATH`, `NSS_LOG_FORMAT`)
3. Config YAML — loaded via `merge_overrides(config_path, overrides)`
4. Defaults — model defaults

`CLISettings.from_cli_kwargs()` filters out `None` values so env vars can fill in. For run commands, `synthesis_overrides` from `**kwargs` (via `parse_overrides`) are merged over config: CLI > dataset registry overrides > config file.

## Gotchas

- Decorator order: `@common_run_options` then `@pydantic_options(...)`; options are applied bottom-up, so list them in reverse in the decorator chain.
- run vs run train/generate: `run` gets `pydantic_options` on the group; `run train` does not (no per-param overrides for train-only); `run generate` gets `pydantic_options` again for generation overrides.
- NSS_PHASE env: Set before `common_setup()` to `train`, `generate`, or `end_to_end`; used for artifact layout.
- VLLM_CONFIGURE_LOGGING: Set to `0` in `_initialize_logging_for_cli_from_settings` before vLLM import so vLLM uses our handlers.

## Extension recipe

New top-level command: Define `@click.group()` or `@click.command()` in a new module; call `cli.add_command(mycommand)` in `cli.py`.

New run subcommand: In `run.py`, add:

```python
@run.command("mystage")
@common_run_options
@pydantic_options(SafeSynthesizerParameters, field_separator=CLI_NESTED_FIELD_SEPARATOR)
def run_mystage(config_path, data_source, ..., **kwargs):
    settings = CLISettings.from_cli_kwargs(..., synthesis_overrides=parse_overrides(kwargs))
    run_logger, config, df, workdir = common_setup(settings=settings, phase="mystage")
    # ...
```


New common option: Add to `common_run_options` list, pass through to handler, include in `CLISettings.from_cli_kwargs()`.

## Read first

- `cli.py` — root group, command registration
- `run.py` — run commands, `common_run_options`, `common_setup` flow
- `settings.py` — `CLISettings`, `from_cli_kwargs`, precedence
- `utils.py` — `common_setup`, `merge_overrides`, `CLI_NESTED_FIELD_SEPARATOR`
- `configurator/pydantic_click_options.py` — `pydantic_options`, `parse_overrides`
