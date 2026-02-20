<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# generation

Synthetic data generation via vLLM with structured output, LoRA adapters, and batch management.

## Purpose

Generate tabular or time-series records using fine-tuned LLMs. Supports structured output (regex or JSON schema), LoRA adapter loading, adaptive batching, and stopping conditions based on invalid-record fraction.

## Backend Hierarchy

- **GeneratorBackend** (ABC) — defines `initialize`, `prepare_params`, `generate`, `teardown`. Subclasshook checks for these methods.
- **VllmBackend** — concrete implementation for tabular data. Loads vLLM, builds prompts from schema, runs generation in batches.
- **TimeseriesBackend** — extends VllmBackend (not a separate hierarchy). Adds sliding-window prefill, parallel group generation, chronological validation, and time-range-based output.

## Processor ABC

Three implementations (selected by `create_processor()` in `processors.py`):

- **TabularDataProcessor** — `extract_and_validate_records()`. Standard JSONL parsing.
- **TimeSeriesDataProcessor** — `extract_and_validate_timeseries_records()`. Requires `time_column`, `interval_seconds`, `time_format`.
- **GroupedDataProcessor** — `extract_groups_from_jsonl_string()` with BOS/EOS delimiters. Supports `group_by_accept_no_delineator`, `group_by_ignore_invalid_records`, `group_by_fix_non_unique_value`, `group_by_fix_unordered_records`.

## Structured Generation

Two methods (config: `structured_generation_schema_method`):

- **regex** — `build_json_based_regex()` in `regex_manager.py`. Produces a regex from JSON schema; used for TabFT-style schemas. Passed to `StructuredOutputsParams(regex=...)`.
- **json_schema** — native vLLM `StructuredOutputsParams(json=...)` with the schema dict.

**Regex limitations** (not supported): `additionalProperties`, `oneOf`/`anyOf`/`allOf`, `$ref`. Regex handles `properties`, `required`, `enum`, `type` (string, integer, number, boolean, null, array, object), `minItems`/`maxItems`, `minProperties`/`maxProperties`, `minLength`/`maxLength`, `pattern`, `format` (date-time, date, time, uuid).

## LoRA Loading

`LoRARequest("lora", 1, str(adapter_path))` passed to `llm.generate(..., lora_request=self.lora_req)`. Rank from `config.training.lora_r` (used in vLLM init via `max_lora_rank`). Adapter path from `workdir.adapter_path` or `model_metadata.adapter_path`.

## Batch Adaptation

`GenerationBatches.get_next_num_prompts()` (in `results.py`): estimates prompts from `valid_records_per_prompt = num_valid_records / num_prompts`; computes `num_prompts_needed = num_records_remaining / (valid_records_per_prompt + EPS)`; caps at `max_num_prompts_per_batch` and adds `NUM_PROMPT_BUFFER`. First batch uses `max_num_prompts_per_batch`; subsequent batches adapt.

## Stopping Conditions

In `GenerationBatches.add_batch()`:

- **STOP_NO_RECORDS** — first batch (batch 0) has zero valid records. Immediate stop.
- **STOP_METRIC_REACHED** — invalid fraction (1 − valid_record_fraction) ≥ `invalid_fraction_threshold` for `patience` consecutive batches. Uses `GenerationStopCondition.has_been_reached(running_stopping_metric.mean)`.
- With `stop_condition=None`, any batch with zero valid records stops.

## Memory Cleanup

`teardown()` → `_clear_llm_state()` → `cleanup_dist_env_and_memory()` + `self.llm = None` + `cleanup_memory()`. `__del__` also calls `_clear_llm_state()` with exception suppression to avoid masking during GC.

## Gotchas

- **`_resolve_temperature()`** — forces `temperature=0.0` when `do_sample=False`; raises if `do_sample=False` and `temperature > 0`.
- **`need_special_token_outputs`** — `True` when processor is not `TabularDataProcessor` (VllmBackend) or not `TimeSeriesDataProcessor` (TimeseriesBackend). When True: `skip_special_tokens=False`, `include_stop_str_in_output=True`, `ignore_eos=True`. Needed for GroupedDataProcessor BOS/EOS.
- **Timeseries single valid response** — `_retain_single_valid_response()` keeps only the response with the most valid records per batch; others trimmed for sliding-window continuity.
- **`typical_p`** — wrapped as `TypicalLogitsWarperWrapper` (logits processor) because vLLM lacks native typical sampling.
- **`num_beams`** — mapped to `beam_width` only when `x > 1`; otherwise `(None, None)` (excluded from SamplingParams).

## Extension Points

- **New backend** — implement `GeneratorBackend` ABC (`initialize`, `prepare_params`, `generate`, `teardown`).
- **New processor** — subclass `Processor`, implement `_process_text_generation(text) -> ParsedResponse`, register in `create_processor()`.
- **New regex builders** — extend `_build_regex()` / `_type_regex` in `regex_manager.py` for new schema constructs.

## Read First

- `backend.py` — GeneratorBackend ABC
- `vllm_backend.py` — VllmBackend, prepare_params, _generate, LoRA, teardown
- `timeseries_backend.py` — TimeseriesBackend, GroupState, _generate_parallel_groups, _retain_single_valid_response
- `regex_manager.py` — build_json_based_regex, _build_regex
- `processors.py` — Processor ABC, TabularDataProcessor, TimeSeriesDataProcessor, GroupedDataProcessor, create_processor
- `batch.py` — Batch (single-batch results), process(), to_dataframe()
- `results.py` — GenerationBatches, get_next_num_prompts, add_batch, stopping logic
- `stopping.py` — GenerationStopCondition
