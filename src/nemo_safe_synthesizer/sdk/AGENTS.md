<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# sdk

Programmatic API for Safe-Synthesizer using a builder pattern. Provides fluent configuration and stepwise execution; the root `AGENTS.md` documents public usage. This guide covers internal patterns for agents.

## Purpose

The SDK is the programmatic entry point for running Safe-Synthesizer pipelines outside the CLI. It supports: config from YAML + builder overrides, DataFrame or URL data sources, PII replacement (on by default), and train/generate/evaluate.

## Builder pattern

Base: `ConfigBuilder` in `config_builder.py` holds `_*_config` attributes (e.g. `_training_config`, `_generation_config`) and `_data_source`. Each `with_*` method returns `Self` for chaining.

Extension: `SafeSynthesizer` in `library_builder.py` extends `ConfigBuilder`, adds `Workdir`, and implements the executable pipeline (`process_data`, `train`, `generate`, `evaluate`).

Important: `with_replace_pii(enable=False)` sets `_enable_replace_pii = False` and clears `_replace_pii_config`; PII is enabled by default (`_enable_replace_pii = True`). Use `Self` type hints for method chaining.

## Dynamic backend selection

Training: `get_training_backend_class(config)` chooses between `HuggingFaceBackend` and `UnslothTrainer`. The factory uses `config.training.use_unsloth`; `"unsloth"` → `_get_unsloth_backend_class()`.

Lazy import for Unsloth: `_get_unsloth_backend_class()` does `from ..training.unsloth_backend import UnslothTrainer` *inside* the function. This keeps Unsloth optional so the package loads without it; import happens only when Unsloth is selected.

Generation: VLLM vs Timeseries is chosen in `generate()`: `time_series.is_timeseries` → `TimeseriesBackend`, else `VllmBackend`.

## Config resolution

`_resolve_config(values, cls, **kwargs)` in `ConfigBuilder`:
- `values=None` → `cls(**kwargs)` (defaults + overrides)
- `values` = Pydantic model → `model.model_copy(update=kwargs)`
- `values` = dict → `cls.model_validate(d).model_copy(update=kwargs)`

Precedence: `kwargs` override `values`; `values` override model defaults. Each `with_*` stores a resolved config; `_resolve_nss_config()` builds `SafeSynthesizerParameters` from `_nss_inputs` (`_data_config`, `_training_config`, etc.).

`with_replace_pii`: Special case: when `config=None`, uses `PiiReplacerConfig.get_default_config()`; `_resolve_config` is not used.

## Stepwise execution

`process_data()` → `train()` → `generate()` → `evaluate()`. Each returns `self` and can be chained or called independently.

- process_data(): Holdout split, `AutoConfigResolver`, optional PII replacement, `ModelMetadata` creation, writes train/test CSV to workdir.
- train(): Calls `get_training_backend_class()`, loads model, trains. Updates `self._nss_config` from `trainer.params` (e.g. inferred timestamp_format).
- generate(): Chooses `TimeseriesBackend` or `VllmBackend`, initializes, generates.
- evaluate(): Builds `Evaluator`, compiles `results` via `make_nss_results`.

`run()` calls `process_data().train().generate().evaluate()`.

## Gotchas

- Unsloth lazy import: Do not import `UnslothTrainer` at module level; use `_get_unsloth_backend_class()` so the package works without the Unsloth extra.
- Self type hints: All fluent methods return `Self`; subclass overrides must match so chaining preserves the concrete type.
- Internal config state: `_nss_config` is assembled by `_resolve_nss_config()`. When `config` is passed to `__init__`, it seeds the per-section `_*_config`; later `with_*` calls overwrite those sections.
- NSS_PHASE env: Set during each stage (`process_data`, `train`, `generate`, `evaluate`) for artifact layout; check `Workdir` / `artifact_structure` if paths change.
- evaluate() dependencies: Expects `trainer`, `generator`, `_train_df`, `_test_df`, `_column_statistics`, `_pii_replacer_time`, `_total_start` to exist. Call `process_data().train().generate()` before `evaluate()`.

## Extension points

New training backend: Add to `class_map` in `get_training_backend_class()`. Use lazy import if the backend is optional.

New config section: Add a `_*_config` attribute and a `with_*()` method in `ConfigBuilder`; add the attribute name to `_nss_inputs` (and `params_map` mapping if the param name differs). Extend `SafeSynthesizerParameters` in `config/` accordingly.

New pipeline stage: Add a method on `SafeSynthesizer` that returns `self`, update `run()` to call it in sequence, and set `NSS_PHASE` for the stage.

## Read first

- `config_builder.py` — `ConfigBuilder`, `_resolve_config`, `_resolve_nss_config`, `_nss_inputs`
- `library_builder.py` — `SafeSynthesizer`, `get_training_backend_class`, `_get_unsloth_backend_class`, stepwise methods, `run()`
- `cli/artifact_structure.py` — `Workdir`, paths used by SDK
- `config/` — `SafeSynthesizerParameters` and section models
