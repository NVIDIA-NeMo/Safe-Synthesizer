<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# tests/AGENTS.md

Per-module guide for AI agents working with the Safe-Synthesizer test suite. Root `AGENTS.md` documents test commands and markers; this file covers conventions, fixture discovery, data locations, and isolation patterns.

## Read First

1. **tests/conftest.py** — Auto-marking, `load_test_dataset`/`load_test_dataframe`, `fixture_mock_processor` pattern
2. **pytest.ini** — Markers, asyncio, timeout
3. **tests/evaluation/conftest.py** — Most complex: Faker-based `make_df`, nullable dtype conversion
4. **tests/generation/conftest.py** — JSONL/schema fixtures, `fixture_valid_iris_dataset_jsonl_and_schema`

## Fixture Discovery

**7 conftest.py files:** `tests/`, `tests/training/`, `tests/generation/`, `tests/evaluation/`, `tests/cli/`, `tests/data_processing/`, `tests/config/`.

Dataset/tokenizer fixtures use the `fixture_` prefix; config/CLI use descriptive names (`mock_workdir`, `basic_parameter`). Root conftest: `yaml_config_str`, `fixture_session_cache_dir`, `fixture_stub_tokenizer_path`, dataset loaders (`fixture_iris_dataset`, etc.), `fixture_mock_processor`. Generation/eval/data_processing share tokenizer and JSONL fixtures. CLI: `mock_workdir(tmp_path)` for tmp_path-based Workdir. Config: `basic_parameter`, `training_hyperparams`, `simple_safe_synthesizer_parameters`.

## Auto-marking

`pytest_collection_modifyitems` in root conftest checks path for `/gpu_integration/`, `/e2e/`, `/integration/`; assigns marker only if not already present. Otherwise falls back to `unit`.

## Test Data Locations

| Location | Contents |
|----------|----------|
| `tests/stub_datasets/` | 8 files: `iris.csv`, `chickweight.csv`, `dow_jones_index.csv`, `sample-patient-events-12groups-200-records.csv`, `pems_sf_sample.csv`, `embedded_carriage_return.parquet`, `lmsys_chat_non_english_sample.jsonl`, `adobe-sampled.csv` |
| `tests/stub_tokenizer/` | Minimal tokenizer config |
| `tests/test_data/tokenizers/` | Full tokenizers: `tinyllama/`, `mistral7b/`, `smollm3b/` |
| `tests/pii_replacer/fake_people_dataset.csv` | PII test data for NER/replacement |

Load via `load_test_dataset(filename)` (HuggingFace `Dataset`) or `load_test_dataframe(filename)` (`pd.DataFrame`).

## Fixture Scoping

Tokenizers are function-scoped (expensive to load). Most fixtures are function-scoped. `fixture_session_cache_dir` is session-scoped.

## Mocking Conventions

**ParsedResponse:** `valid_records=[...]`, `invalid_records=[...]`, `errors=[...]`, `prompt_number=int`. Use `fixture_mock_processor` or `fixture_mock_processor_without_valid_records`. `pytest.importorskip("transformers")` for optional deps. Mock Workdir via `mock_workdir(tmp_path)` in `cli/conftest.py`.

## GPU Isolation Gotcha

Unsloth patches transformers invasively. Running unsloth tests before DP tests causes failures. Use separate invocations:

```bash
pytest tests/e2e/ -k default
pytest tests/e2e/ -k dp
```

## Other Gotchas

- **Nullable dtype before NaN:** Convert to `pd.Int64Dtype()`/`pd.BooleanDtype()` before assigning `np.nan`; see `evaluation/conftest.py` `make_df`.
- **Faker:** Seed with `fake.seed_instance(seed)` and `random.seed(seed)` for reproducibility.
- **Tests mirror source structure:** `tests/training/`, `tests/generation/`, etc.
- **Naming:** fixture names use `fixture_` prefix consistently (e.g., `fixture_embedded_carriage_return_dataframe`).
