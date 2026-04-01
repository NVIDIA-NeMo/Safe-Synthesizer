<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Testing Guide

Comprehensive testing reference for Safe-Synthesizer developers. Covers commands, markers, test data, fixtures, and gotchas.

## Read First

1. `tests/conftest.py` -- auto-marking, `load_test_dataset`/`load_test_dataframe`, `fixture_mock_processor` pattern
2. `pytest.ini` -- markers, asyncio, timeout
3. `tests/evaluation/conftest.py` -- most complex: Faker-based `make_df`, nullable dtype conversion
4. `tests/generation/conftest.py` -- JSONL/schema fixtures, `fixture_valid_iris_dataset_jsonl_and_schema`

## Running Tests

All `make` targets, grouped by scope:

```bash
make test                  # Unit (excludes slow, e2e, and smoke)
make test-unit-slow        # Unit tests including slow (excludes e2e and smoke)
make test-smoke            # CPU smoke tests (~few min, no GPU required)
make test-smoke-gpu        # GPU smoke tests (requires CUDA)
make test-e2e              # All e2e (requires CUDA) -- runs default + dp
make test-e2e-default      # e2e default (unsloth) tests only
make test-e2e-dp           # e2e DP tests only
make test-ci               # CI unit tests with coverage (excludes slow, e2e, gpu, smoke)
make test-ci-slow          # CI slow tests with coverage
make test-ci-container     # CI tests in a Linux container (Docker/Podman)
```

Run a single test:

```bash
uv run --frozen pytest tests/path/test_file.py::test_name -vvs -n0
```

Test runner: `uv run --frozen pytest -n auto --dist loadscope -vv`

## Config-Dataset Combo Tests

Two test functions (`test_clinc_oos_dataset`, `test_dow_jones_index_dataset`) each parametrized over 6 model configs = 12 combinations. Each has a dedicated Makefile target:

```bash
make test-nss-{CONFIG}-{DATASET}-ci
```

Configs: `tinyllama_unsloth`, `tinyllama_dp`, `smollm3_unsloth`, `smollm3_dp`, `mistral_nodp`, `mistral_dp`

Datasets: `clinc_oos`, `dow_jones_index`

Example:

```bash
make test-nss-tinyllama_unsloth-clinc_oos-ci
make test-nss-mistral_dp-dow_jones_index-ci
```

Details:

- Driven by `tests/e2e/test_dataset_config.py` with YAML configs under `tests/e2e/required_configs/`
- Each target bootstraps `cu128`, runs single-process (`-n 0`) with coverage
- These are not part of `make test-e2e` -- they are standalone CI targets

## Pytest Markers

Defined in `pytest.ini` (`--strict-markers` is enabled):


| Marker         | Meaning                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `unit`         | Unit tests (default, no marker needed)                                                           |
| `slow`         | Long-running tests                                                                               |
| `smoke`        | Quick smoke tests (training/generation hot paths, tiny models)                                   |
| `e2e`          | End-to-end pipeline tests (requires CUDA)                                                        |
| `requires_gpu` | Test needs CUDA hardware (modifier, stacks on `smoke`/`e2e`)                                     |
| `vllm`         | Tests using vLLM generation backend (each file runs in its own process for GPU memory isolation) |
| `smollm2`      | SmolLM2 Hub download tests (Makefile uses for process isolation)                                 |
| `unsloth`      | Unsloth backend tests (process-isolated from DP tests)                                           |
| `noautouse`    | Skip autouse fixtures for specific tests                                                         |

Every test should have exactly one of the category markers: `unit, smoke, e2e`.
The other markers modify the 3 categories, indicating when they should be run (`slow, requires_gpu`), or when separate pytest invocations are required (`vllm, unsloth`).

## Auto-marking

`pytest_collection_modifyitems` in root `conftest.py` assigns markers based on test path:

- `/e2e/` -> `e2e`
- `/smoke/` -> `smoke`
- No match -> `unit`

Markers are only added if none of the 3 category markers (`unit`, `smoke`, `e2e`) are already present on the test item.

## Test Data Locations


| Location                                     | Contents                                                                                                                                                                                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/stub_datasets/`                       | Sample datasets including: `iris.csv`, `chickweight.csv`, `dow_jones_index_group_size_8.csv`, `clinc_oos.csv`, `sample-patient-events-12groups-200-records.csv`, `pems_sf_sample.csv`, `lmsys_chat_non_english_sample.jsonl`, `doc_summaries.csv` (+ `licenses.md`) |
| `tests/stub_tokenizer/`                      | Minimal tokenizer config                                                                                                                                                                                                                                            |
| `tests/test_data/tokenizers/`                | Full tokenizers: `tinyllama/`, `mistral7b/`, `smollm3b/`                                                                                                                                                                                                            |
| `tests/pii_replacer/fake_people_dataset.csv` | PII test data for NER/replacement                                                                                                                                                                                                                                   |
| `tests/e2e/required_configs/`                | 6 YAML configs: `tinyllama-unsloth`, `tinyllama-dp`, `smollm3-unsloth`, `smollm3-dp`, `mistral-nodp`, `mistral-dp`                                                                                                                                                  |


Load helpers in root `conftest.py`:

- `load_test_dataset(filename)` -- returns HuggingFace `Dataset`
- `load_test_dataframe(filename)` -- returns `pd.DataFrame`

## Fixture Discovery

9 `conftest.py` files: `tests/`, `tests/training/`, `tests/generation/`, `tests/evaluation/`, `tests/cli/`, `tests/data_processing/`, `tests/config/`, `tests/e2e/` (currently empty), `tests/smoke/`.

Dataset/tokenizer fixtures use the `fixture_` prefix; config/CLI use descriptive names (`mock_workdir`, `training_hyperparams`).

Key fixtures in root `conftest.py`:

- `yaml_config_str`, `fixture_session_cache_dir`, `fixture_stub_tokenizer_path`
- Dataset loaders: `fixture_iris_dataset`, `fixture_dow_jones_index_dataset` (loads `dow_jones_index_group_size_8.csv`), `fixture_clinc_oos_dataset` (loads `clinc_oos.csv`), `fixture_chickweight_dataset`, etc.
- `fixture_mock_processor`, `fixture_mock_processor_without_valid_records`

Per-module fixtures:

- Generation/eval/data_processing: shared tokenizer and JSONL fixtures
- CLI: `mock_workdir(tmp_path)` for tmp_path-based Workdir
- Config: `basic_parameter`, `training_hyperparams`, `simple_safe_synthesizer_parameters`
- Smoke: session-scoped `tiny_llama_config`, `stub_tokenizer`, `local_tinyllama_dir`, `iris_df`, `base_smoke_config`, `_patch_attn_eager`; function-scoped `tiny_model`; helpers `train_with_sdk()`, `assert_adapter_saved()`

## Fixture Scoping

Tokenizers are function-scoped (expensive to load). Most fixtures are function-scoped. `fixture_session_cache_dir` is session-scoped.

## Mocking Conventions

`ParsedResponse`: `valid_records=[...]`, `invalid_records=[...]`, `errors=[...]`, `prompt_number=int`. Use `fixture_mock_processor` or `fixture_mock_processor_without_valid_records`.

Optional dependencies: use `pytest.importorskip` to gate on packages that require specific extras. E2e tests use this for `sentence_transformers` and `vllm` (require `cu128` extra).

Mock Workdir via `mock_workdir(tmp_path)` in `cli/conftest.py`.

## GPU Isolation Gotcha

Two GPU isolation hazards require per-file process isolation (`-n 0`):

1. vLLM pre-allocates all GPU memory and never releases it within a process. Tests that call `.generate()` must run in separate processes or later tests OOM.
2. Unsloth patches transformers at import time, poisoning Opacus/DP if they share a process.

GPU smoke tests use markers to express isolation requirements:

- `requires_gpu`: all GPU tests
- `vllm`: tests using vLLM generation (each file gets its own process)
- `smollm2`, `unsloth`: marker-isolated groups (auto-discovered)

`make test-smoke-gpu` uses marker algebra for train-only tests (auto-discovering via `requires_gpu and not vllm and not smollm2 and not unsloth`), explicit file paths for vLLM tests (per-file isolation), and marker selection for SmolLM2/Unsloth. When adding a new vLLM test file, add `pytest.mark.vllm` and also add the file to the Makefile's explicit list.

`make test-e2e` splits into `test-e2e-default` + `test-e2e-dp`, each single-process over `tests/e2e/`.

See [tests/smoke/README.md](smoke/README.md) for additional smoke-specific gotchas.

## Other Gotchas

- Nullable dtype before NaN: convert to `pd.Int64Dtype()`/`pd.BooleanDtype()` before assigning `np.nan`; see `evaluation/conftest.py` `make_df`.
- Faker: seed with `fake.seed_instance(seed)` and `random.seed(seed)` for reproducibility.
- Tests mirror source structure: `tests/training/`, `tests/generation/`, etc.
- Naming: fixture names use `fixture_` prefix consistently (e.g., `fixture_iris_dataset`).
- `print()` is allowed in tests (ruff `T201` is suppressed for `tests/`). Use it freely for debug output in test functions.
