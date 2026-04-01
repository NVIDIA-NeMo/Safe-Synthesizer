# Time Series Support for Safe Synthesizer

This document describes the time series synthesis capabilities added to NeMo Safe Synthesizer. It covers the design decisions, configuration parameters, data preprocessing, generation logic, validation, and evaluation components.

## Table of Contents

1. [Overview](#overview)
2. [Configuration Parameters](#configuration-parameters)
   - [Timestamp Validation Mode (Planned)](#timestamp-validation-mode-planned)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Training Data Assembly](#training-data-assembly)
5. [Generation Logic](#generation-logic)
6. [Record Validation](#record-validation)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Usage Examples](#usage-examples)
9. [Design Decisions & Rationale](#design-decisions--rationale)

---

## Overview

Time series support enables Safe Synthesizer to generate synthetic tabular data that preserves temporal ordering and sequential dependencies. This is crucial for datasets where records represent measurements over time (e.g., sensor readings, financial data, IoT telemetry).

### Key Features

- **Unified grouped architecture**: All time series are processed through a unified grouped architecture. Single-sequence data is automatically treated as a single group via an internal pseudo-group column, enabling consistent processing paths.
- **Sliding window generation**: Uses a sliding window approach where recently generated records are fed back as context for generating the next batch.
- **Parallel group generation**: Multiple groups are processed in parallel batches for efficiency, even single-sequence data uses this optimized path.
- **Time-range based generation**: The number of records generated is determined by the configured time range and interval `(stop_timestamp - start_timestamp) / interval_seconds`, not by a target count.
- **Chronological constraint enforcement**: Validates that generated timestamps follow the expected interval pattern.
- **Autocorrelation-based evaluation**: Measures how well the synthetic data preserves temporal patterns from the original data. (ToDo)

---

## Configuration Parameters

### TimeSeriesParameters

Located in `src/nemo_safe_synthesizer/config/time_series.py`, this configuration class controls time series behavior:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `is_timeseries` | `bool` | `False` | Master switch to enable time series mode. When enabled, `timestamp_column` or `timestamp_interval_seconds` must be provided. |
| `timestamp_column` | `str \| None` | `None` | Name of the column containing timestamps. Required when `is_timeseries=True` unless `timestamp_interval_seconds` is provided. |
| `timestamp_interval_seconds` | `int \| None` | `None` | Expected interval in seconds between consecutive timestamps. If not provided, will be inferred from the data. |
| `timestamp_format` | `str \| None` | `None` | Format string for parsing timestamps (e.g., `"%Y-%m-%d %H:%M:%S"`) or `"elapsed_seconds"` for numeric timestamps. If not provided, will be inferred from the data. |
| `start_timestamp` | `str \| int \| None` | `None` | Start timestamp for generation. Defaults to the first timestamp in the training data (validated to be consistent across groups). |
| `stop_timestamp` | `str \| int \| None` | `None` | Stop timestamp for generation. Defaults to the last timestamp in the training data (validated to be consistent across groups). |

**Validation Rules:**
- When `is_timeseries=True`, at least one of `timestamp_column` or `timestamp_interval_seconds` must be provided.
- When `is_timeseries=False`, `timestamp_column` cannot be set.
- For grouped time series, `group_training_examples_by` (in data config) should also be set.
- All groups must have the same start and stop timestamps (enforced during preprocessing).

### Generation Parameters

Additional generation parameters in `src/nemo_safe_synthesizer/config/generate.py`:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enforce_timeseries_fidelity` | `bool` | `False` | When enabled, enforces strict time series ordering, intervals, and start/end times during generation for consistent time intervals. |
| `prefill_context_ratio` | `float` | `0.0` | Ratio of average records per training example to use as the sliding window prefill context size. Effective size is `max(int(ratio * avg_records_per_example), 3)`, capped by context length. `0.0` uses the default of 3 records. |
| `invalid_fraction_threshold` | `float` | `0.95` | Threshold for invalid record fraction. Groups fail after `patience` consecutive batches exceeding this threshold. |
| `patience` | `int` | `1` | Number of consecutive batches with high invalid fraction before a group fails. |
| `num_records` | `int` | - | Target number of records (used for progress tracking, not to limit generation in time-series mode). |



### Evaluation Parameters (ToDo)

Time series evaluation is configured through `src/nemo_safe_synthesizer/config/evaluate.py`:

```python
class AutocorrelationSimilarityParameters:
    enabled: bool = False  # Enable the autocorrelation similarity metric
    value_columns: list[str] | None = None  # Columns to evaluate (defaults to all numeric)
    timestamp_column: str | None = None  # Timestamp column for ordering
    group_column: str | None = None  # Column for independent time series groups
    max_lag: int = 20  # Number of lags for autocorrelation comparison
    distance_metric: Literal["euclidean", "mae"] = "euclidean"  # Distance metric
```

---

## Preprocessing Pipeline

Time series preprocessing occurs during training data preparation in `src/nemo_safe_synthesizer/training/timeseries_preprocessing.py`.

### Processing Steps

1. **Pseudo-Group Column Addition**
   - If no `group_training_examples_by` column is specified, a `__pseudo_group__` column is added with all rows set to 0.
   - This unifies the processing of grouped and ungrouped time series through a single code path.

2. **Timestamp Column Creation (if needed)**
   - If `timestamp_column` is not provided but `timestamp_interval_seconds` is, a synthetic `elapsed_seconds` column is created.
   - Records are indexed sequentially with the specified interval per group.

3. **Validation**
   - Verifies the timestamp column exists in the data.
   - Checks for missing values in the timestamp column.

4. **Sorting**
   - Data is always sorted by group column first, then by timestamp within each group.
   - Even ungrouped data follows this pattern (using the pseudo-group column).

5. **Format Inference and Conversion**
   - If `timestamp_format` is not provided, attempts to infer the datetime format using `guess_datetime_format()`.
   - Special format `"elapsed_seconds"` indicates integer elapsed time (not datetime).
   - Validates user-provided format matches the actual data.
   - Converts timestamp column to datetime for processing (then back to string format).

6. **Interval Inference and Validation**
   - Collects timestamp statistics for each group.
   - If `timestamp_interval_seconds` is provided, validates that actual intervals match (with 0.1s tolerance).
   - If not provided, infers the interval only if all groups have consistent intervals.

7. **Start/Stop Consistency Validation**
   - Validates that all groups have the same start and stop timestamps.
   - If timestamps differ across groups, raises a `DataError`.
   - Sets `start_timestamp` and `stop_timestamp` in config based on validated values.

### Code Flow

```
HuggingFaceBackend._process_timeseries()
    └── process_timeseries_data(df, config)
            ├── _add_pseudo_group_if_needed()     # Unify grouped/ungrouped
            ├── _create_elapsed_time_column()     # If no timestamp column
            ├── _validate_timestamp_column()      # Check existence and nulls
            ├── _sort_by_group_and_timestamp()    # Sort data
            ├── _infer_and_convert_timestamp_format()  # Handle datetime conversion
            ├── _process_grouped_timestamps()     # Validate all groups
            │       ├── _collect_group_timestamp_stats()
            │       ├── _validate_interval_consistency()
            │       └── _validate_start_stop_consistency()
            └── Return (processed_df, updated_config)
```

---

## Training Data Assembly

The `SequentialExampleAssembler` in `src/nemo_safe_synthesizer/data_processing/assembler.py` handles time series training data.

### Key Differences from Tabular Assembler

1. **No Shuffling**: Records maintain their original order to preserve temporal relationships.

2. **Single-Group Examples**: Each training example contains records from exactly one group. This ensures the model learns patterns within a group's sequence without cross-group contamination.

3. **Sequential Packing**: Records are packed into training examples sequentially, respecting:
   - Token budget (70-100% of max context length for training, 100% for validation)
   - Group boundaries (one group per example)
   - Dataset boundaries (restart detection when row index decreases for data_fraction > 1)

4. **Pseudo-Group Handling**: When no group column is specified, preprocessing adds a `__pseudo_group__` column so ungrouped time series is treated as a single group. This unifies the grouped and ungrouped code paths.

5. **Initial Prefill Extraction**
   - Dictionary mapping each group to its first 3 decoded samples (including pseudo-group for single sequences).
   - Stored in `model_metadata.initial_prefill` for use during generation.
   - Used by `TimeseriesBackend` to seed each group's context.

6. **Train/Test Split**
   - Split is done by group boundaries using `grouped_train_test_split`.
   - Entire groups go to train OR validation, never split across.
   - Re-sort after split since GroupShuffleSplit shuffles indices.
   - Row indices are added to detect dataset restart boundaries.

### Example Generation Algorithm

```python
def _fill_context_with_records_generator(dataset):
    # Iteration indices for slicing
    example_start = 0
    current_idx = 0
    
    while current_idx < len(dataset):
        record = dataset[current_idx]
        row_idx = record["__row_idx"]  # Original position
        
        # Set group value at start of new example
        if token_total == 0:
            current_group_value = record_group
        
        # Check flush conditions
        restart_boundary = prev_row_idx is not None and row_idx < prev_row_idx
        group_boundary = current_group_value is not None and record_group != current_group_value
        would_exceed_tokens = token_total + record_len > token_budget
        
        if _should_flush_example(...):
            yield _flush_example(dataset, example_start, current_idx)
            # Reset state for next example
            example_start = current_idx
            token_budget = _next_token_budget()  # Random 70-100% for train
            continue
        
        # Add record to current example
        token_total += record_len
        current_idx += 1
```

### Class Hierarchy

```
TrainingExampleAssembler (ABC)
    ├── TabularDataExampleAssembler       # Standard tabular (shuffled)
    ├── SequentialExampleAssembler        # Time series (ordered, single-group examples)
    └── GroupedDataExampleAssembler       # Grouped tabular (multiple groups per example)
```

---

## Generation Logic

Time series generation is handled by `TimeseriesBackend` in `src/nemo_safe_synthesizer/generation/timeseries_backend.py`.

### Architecture

All time series (including single-sequence) use parallel group generation. Single-sequence data is treated as 1 group via the pseudo-group column added during preprocessing.

```
TimeseriesBackend(VllmBackend)
    └── Parallel group generation (_generate_parallel_groups)
```

### Key Concepts

- **Time-Range Based Generation**: The number of records generated is determined by `(stop_timestamp - start_timestamp) / interval_seconds`, not by a target count. The `config.generation.num_records` parameter is used only for progress tracking.
- **Sliding Window**: Maintains a window of recent records (controlled by `_prefill_context_size`) included in each prompt for context continuity. The window size is computed as `max(int(prefill_context_ratio * avg_records_per_example), 3)`, capped by the model's context length. When the ratio is 0 or training stats are unavailable, it defaults to 3.
- **Groups from Training**: Groups are the same as those seen during training (from `model_metadata.initial_prefill`).

### Sliding Window Approach

1. **Prefill Initialization**: Start with initial prefill from training data (first 3 records per group).
2. **Batch Generation**: Generate multiple samples (default 5) per prompt for each active group.
3. **Response Selection**: Keep the response with the most valid records per group.
4. **Context Update**: Update sliding window with new valid records.
5. **Repeat**: Continue until stop timestamp is reached for each group.

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `_samples_per_prompt` | 5 | Number of samples generated per prompt |
| `_max_prompts_per_batch` | 100 | Max prompts per batch in parallel generation |
| `_prefill_context_size` | dynamic | Number of recent records in sliding window. Computed as `max(int(prefill_context_ratio * avg_records_per_example), 3)`, capped by context length. Defaults to 3 when ratio is 0 or stats are unavailable. |

### Parallel Group Generation Flow

All time series use parallel group generation (single-sequence is just 1 group):

```
1. Initialize GroupState for each group with prefill from training
2. Compute expected records per group: (stop - start) / interval + 1
3. While groups remain pending or active:
   a. Fill active slots with pending groups (up to max_groups_per_batch)
   b. Build prompts for all active groups using current prefill
   c. Generate completions for all prompts in single LLM batch call
   d. Process LLM outputs into per-group Batch objects
   e. For each group:
      - Validate chronological order against group's last timestamp
      - Retain response with most valid records (discard others)
      - Update group state (prefill, last_timestamp)
      - Check if stop timestamp reached (marks group complete)
      - Track low valid fraction; fail group after max retries
   f. Remove completed/failed groups from active list
   g. Save progress snapshots if thresholds are met
   h. Log per-group progress summary
```

### GroupState Tracking

Each group maintains independent state:

```python
@dataclass
class GroupState:
    group_id: str
    initial_prefill: str           # Original prefill (first few records)
    current_prefill: str           # Updated as generation progresses
    recent_records: list[dict]     # Sliding window context
    expected_records: int          # Based on (stop - start) / interval
    last_timestamp_seconds: int | None
    low_valid_fraction_count: int  # Counter for consecutive bad batches
    completed: bool
    failed: bool
    total_valid_records: int
    total_invalid_records: int
```

### Stopping Conditions

**Per-Group Stopping:**
- **Completion (success)**: A group completes when any generated record has timestamp >= `stop_timestamp`.
- **Failure (low valid fraction)**: A group fails after `config.generation.patience` consecutive batches where invalid fraction >= `config.generation.invalid_fraction_threshold`. Failed groups produce no synthetic data.

**Global Stopping:**
- **Natural completion**: All groups processed (pending and active lists empty).
- **No records**: Too many consecutive batches with no valid records globally.
- **Target reached**: Target number of records reached (for progress tracking).

### Progress Checkpoints

Partial results are saved at 25%, 50%, and 75% completion based on total expected records across all groups.

---

## Record Validation

Validation occurs at two levels:

### 1. Record-Level Validation (record_utils.py)

The `TimeSeriesDataProcessor` uses `extract_and_validate_timeseries_records()`:

```python
def extract_and_validate_timeseries_records(
    jsonl_string: str,
    schema: dict,
    time_column: str,
    interval_seconds: int | None,
    time_format: str,
) -> tuple[list[dict], list[str], list[tuple[str, str]]]:
```

**Validation Steps:**
1. Parse JSON and validate against schema.
2. Check for large number issues.
3. Verify timestamp column exists.
4. Parse timestamp using the specified format.
5. **If interval_seconds is provided**: Validate sequential increment.
   - Handles day rollovers (adds 24h offset when timestamp wraps).
   - Rejects records if interval doesn't match expected step.
6. **Early termination**: Stops at first invalid record (streaming validation).

### 2. Batch-Level Chronological Validation (timeseries_backend.py)

After record-level validation, the backend performs per-group chronological checks using `_check_chronological_for_group()`:

```python
def _is_chronological_for_group(self, records: list[dict], group_state: GroupState) -> bool:
    """Check if records continue from the group's last timestamp."""
    if not records:
        return False
    
    first_record = records[0]
    timestamp_seconds = self._parse_timestamp_seconds(first_record.get(self._time_column))
    
    if group_state.last_timestamp_seconds is not None:
        expected_ts = self._advance_expected_time(group_state.last_timestamp_seconds)
        if timestamp_seconds != expected_ts:
            return False
    
    return True
```

**Key Design Decision**: If records don't continue from the group's last timestamp, all records in that response are moved to `invalid_records` with an "Out-of-order time step" error, and the response is discarded. This is done per-group during `_process_group_result()`.

---

## Evaluation Metrics (ToDo)

### Autocorrelation Similarity

Located in `src/nemo_safe_synthesizer/evaluation/components/autocorrelation_similarity.py`.

**Purpose**: Measures how well the synthetic data preserves temporal autocorrelation patterns from the reference data.

**Algorithm:**

1. **Compute ACF vectors** for each numeric column:
   - Mean-center the data.
   - Calculate autocorrelation for lags 1 to `max_lag`.
   - Clip values to [-1, 1].

2. **Calculate distance** between reference and synthetic ACF:
   - MAE: Mean Absolute Error.
   - Euclidean: L2 norm.

3. **Normalize and score**:
   - Normalize by realistic max distance.
   - Use RMS of normalized differences across columns.
   - Final score = 1 - RMS (higher is better).

**Evaluation Modes:**

- **Global**: Computes ACF over entire dataset.
- **Per-Group**: Computes ACF per group, averages scores.

**Auto-Enable**: When `is_timeseries=True`, autocorrelation similarity is automatically computed even without explicit evaluation config.

### Integration with Multimodal Report

The multimodal report automatically includes time series components when:
- `is_timeseries=True` in config.
- Time series evaluation is explicitly enabled.

Components added:
- `AutocorrelationSimilarity`: Temporal pattern preservation score.
- `TimeSeriesLineChart`: Visual comparison of time series.

---

## Usage Examples

### Running the Pipeline

Use the `safe-synthesizer run` command with a config file and data URL:

```bash
EXPORT SHARED_DIR=/lustre/fsw/portfolios/llmservice/users/kendrickb/shared_safe_synthesizer/configs
uv run safe-synthesizer run --config $SHARED_DIR/smollm_grouped-utility.yaml --url $SHARED_DIR/synthetic_kpi_qoe_policy_dataset.csv
```

### Example Configuration (Grouped Time Series)

```yaml
data:
  group_training_examples_by: Cell_ID
  holdout: 0
  max_holdout: 2000

time_series:
  is_timeseries: true
  timestamp_column: Timestamp
  timestamp_interval_seconds: null  # Will be inferred
  timestamp_format: null            # Will be inferred
  start_timestamp: null             # From data
  stop_timestamp: null              # From data

generation:
  enforce_timeseries_fidelity: true
  invalid_fraction_threshold: 0.95
  num_records: 1000
  patience: 1
  temperature: 0.8

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  learning_rate: 0.0005
  pretrained_model: HuggingFaceTB/SmolLM3-3B
```

### Single Time Sequence (No Group Column)

For single-sequence time series, simply omit the `group_training_examples_by` parameter. The system will automatically add a pseudo-group column internally.

```yaml
data:
  # No group_training_examples_by - treated as single sequence

time_series:
  is_timeseries: true
  timestamp_column: timestamp
  timestamp_interval_seconds: 60  # Or let it be inferred
```

---

## Design Decisions & Rationale

### 1. Unified Grouped Architecture

**Decision**: All time series (including single-sequence) use the grouped architecture via a pseudo-group column.

**Rationale**:
- Single code path simplifies maintenance and testing.
- Single-sequence is just 1 group, no special-casing needed.
- Parallel generation optimizations apply to all cases.

### 2. Multiple Samples per Prompt

**Decision**: Generate 5 samples per prompt, keep the best one.

**Rationale**:
- Time series generation is more constrained than tabular.
- Multiple samples increase the chance of getting a valid continuation.
- Best sample = most valid records (longer valid sequence is better).

### 3. Parallel Group Generation

**Decision**: Process multiple groups simultaneously in the same batch.

**Rationale**:
- GPU utilization is maximized.
- Independent groups have no cross-dependencies.
- Reduces total wall-clock time significantly.

### 4. Consistent Start/Stop Across Groups

**Decision**: Require all groups to have identical start and stop timestamps.

**Rationale**:
- Simplifies generation logic (fixed expected_records per group).
- Many real-world applications have synchronized sensors.
- Future: Could relax this for asynchronous time series.

### 5. Random Token Budget for Training

**Decision**: Sample token budget from [70%, 100%] of max for training examples.

**Rationale**:
- Prevents overfitting to fixed sequence lengths.
- Creates variable-length examples for robustness.
- 100% used for validation for consistency.

### 6. Single-Group Examples

**Decision**: Each training example contains records from exactly one group.

**Rationale**:
- Ensures model learns patterns within a group's sequence.
- Prevents cross-group contamination during training.
- Sequence continuation across example boundaries is natural.

### 7. Time-Range Based Generation

**Decision**: Number of records is determined by time range and interval, not target count.

**Rationale**:
- Ensures complete time coverage for each group.
- `num_records` parameter is only for progress tracking.
- Groups complete when stop timestamp is reached.

---

## Next Steps

- **Support for variable/irregular intervals** — Planned via `timestamp_validation_mode` (see below).
- More testing on datasets and different models
- Compare NSS to other models
- Add more unit tests

### Timestamp Validation Mode (Planned)

> ⚠️ **Not Yet Implemented** — This section describes planned functionality.

The `timestamp_validation_mode` parameter controls how timestamps are handled during training and generation. This is particularly useful for **time series with varying/irregular intervals** where enforcing a fixed interval is not possible.

| Mode | Training Behavior | Generation Behavior | Use Case |
|------|-------------------|---------------------|----------|
| `ignore` | Exclude timestamp column from training and generation | Generate only non-timestamp columns | When temporal order matters but exact timestamps don't |
| `replace` | Exclude timestamp column from training | Generate rows, then add timestamps post-hoc | When you need specific timestamps but model shouldn't/couldn't learn them |
| `strict` | Include timestamp column, capture exact values | Enforce generated timestamps match training distribution | The temporal context is significant for the model to learn |

---

## File Reference

| File | Purpose |
|------|---------|
| `config/time_series.py` | Time series configuration parameters |
| `config/evaluate.py` | Evaluation config including ACF parameters |
| `training/timeseries_preprocessing.py` | Preprocessing and validation logic |
| `data_processing/assembler.py` | `SequentialExampleAssembler` class for time series |
| `data_processing/record_utils.py` | `extract_and_validate_timeseries_records()` |
| `generation/processors.py` | `TimeSeriesDataProcessor` class |
| `generation/timeseries_backend.py` | `TimeseriesBackend` generation class |
| `evaluation/components/autocorrelation_similarity.py` | ACF evaluation metric |
| `llm/metadata.py` | `initial_prefill`, `avg_records_per_example`, and `avg_tokens_per_record` fields for time series |
| `defaults.py` | `PSEUDO_GROUP_COLUMN` constant |