<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Python SDK

The Python SDK lets you run the full Safe Synthesizer pipeline -- load data, configure, train, generate, and evaluate -- from a single script or notebook cell. Every parameter is optional; sensible defaults are provided so you can start with a few lines and refine later.

## Prerequisites

- `nemo-safe-synthesizer` installed (`uv pip install nemo-safe-synthesizer`)
- Python 3.11+
- GPU with 80 GB+ VRAM for training and generation

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.config import SafeSynthesizerParameters
```

## Data Sources

Pass your data to `with_data_source()`. It accepts three types depending on where your data lives:

=== "DataFrame"

    Use a DataFrame when the data is already loaded or pre-processed in your notebook:

    ```python
    import pandas as pd

    df = pd.read_csv("customers.csv")
    synthesizer = SafeSynthesizer().with_data_source(df)
    ```

=== "File path"

    Point to a local CSV file:

    ```python
    synthesizer = SafeSynthesizer().with_data_source("/data/customers.csv")
    ```

=== "URL"

    Fetch from remote storage:

    ```python
    synthesizer = SafeSynthesizer().with_data_source("https://storage.example.com/customers.csv")
    ```

!!! note
    File paths and URLs are read with `pd.read_csv()` internally, so the data must be CSV-formatted.

## Configuration

There are three ways to configure the pipeline. Choose whichever fits your workflow.

=== "Builder only"

    Best for interactive exploration in a notebook. All parameters have sensible defaults, so you only set what you want to change:

    ```python
    synthesizer = (
        SafeSynthesizer()
        .with_data_source(df)
        .with_train(learning_rate=0.0003)
        .with_generate(num_records=5000)
    )
    ```

=== "From YAML"

    Best for reproducible production runs. Define your configuration in a version-controlled YAML file:

    ```python
    config = SafeSynthesizerParameters.from_yaml("config.yaml")
    synthesizer = SafeSynthesizer(config).with_data_source(df)
    ```

    Also available: `from_json()` for JSON configs and `from_params(**kwargs)` for flat keyword arguments.

=== "YAML + overrides"

    The most common approach -- load a baseline config from YAML, then override specific parameters per experiment:

    ```python
    config = SafeSynthesizerParameters.from_yaml("config.yaml")
    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source(df)
        .with_train(learning_rate=0.0001)
        .with_generate(num_records=10000, temperature=0.8)
    )
    ```

    This lets you version-control a baseline while tuning parameters without editing the file.

!!! tip "Parameter precedence"
    Keyword arguments passed to `with_*` methods take priority over values in a config dictionary or YAML file, which take priority over model defaults.

## Configuring Pipeline Stages

Each `with_*` method configures one stage of the pipeline. You can pass keyword arguments directly, a configuration dictionary, or a Pydantic model object -- all three work the same way.

### Data handling

Control how your data is split and prepared for training.

```python
synthesizer = SafeSynthesizer().with_data_source(df).with_data(
    holdout=0.2,
    group_training_examples_by="customer_id",
    random_state=42,
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `holdout` | `0.2` | Fraction of records held out for evaluation. The held-out set drives privacy and quality metrics, so keep it large enough to be representative. |
| `group_training_examples_by` | `None` | Column to group records by (e.g., `customer_id`). Teaches the model correlations between rows that belong to the same entity. |
| `order_training_examples_by` | `None` | Column to order records within each group. Requires `group_training_examples_by` to be set. |
| `random_state` | random | Seed for the train/test split. Set this for reproducible experiments. |

### Training

Configure how the base LLM is fine-tuned on your data. Calling `with_train()` automatically enables synthesis.

```python
synthesizer = SafeSynthesizer().with_data_source(df).with_train(
    pretrained_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    num_input_records_to_sample=20000,
    learning_rate=0.0005,
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `pretrained_model` | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` | Base LLM to fine-tune. |
| `num_input_records_to_sample` | `"auto"` | How many records the model sees during training. Think of this as a proxy for training duration -- setting it to 2x your dataset size is like training for 2 epochs. |
| `learning_rate` | `0.0005` | Initial learning rate for the AdamW optimizer. |
| `batch_size` | `1` | Batch size per GPU. Lower values use less memory. |
| `gradient_accumulation_steps` | `8` | Number of steps to accumulate gradients before updating. Increasing this compensates for a smaller `batch_size` -- the effective batch size is `batch_size * gradient_accumulation_steps`. |
| `lora_r` | `32` | LoRA rank. Higher values give the model more capacity to learn your data, at the cost of more GPU memory. |

For the complete list of training parameters, see the [:material-api: API reference](../reference/nemo_safe_synthesizer/config/training.md).

### Generation

Configure how synthetic records are produced from the fine-tuned model. Calling `with_generate()` automatically enables synthesis.

```python
synthesizer = SafeSynthesizer().with_data_source(df).with_generate(
    num_records=5000,
    temperature=0.9,
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `num_records` | `1000` | Number of synthetic rows to produce. |
| `temperature` | `0.9` | Controls randomness. Higher values produce more diverse output; lower values are more conservative. |
| `top_p` | `1.0` | Nucleus sampling cutoff. Restricting this (e.g., `0.95`) trims unlikely tokens. |
| `use_structured_generation` | `False` | Forces output to match the dataset schema. Reduces invalid records at the cost of generation speed. |

### Evaluation

Control which quality and privacy metrics are computed after generation.

```python
synthesizer = SafeSynthesizer().with_data_source(df).with_evaluate(
    enabled=True,
    mia_enabled=True,
    aia_enabled=True,
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `enabled` | `True` | Set to `False` to skip evaluation entirely -- useful when you only need the synthetic CSV. |
| `mia_enabled` | `True` | Membership inference attack. Measures whether an attacker can tell if a specific record was in the training set. |
| `aia_enabled` | `True` | Attribute inference attack. Measures whether an attacker can predict sensitive attributes from the synthetic data. |
| `pii_replay_enabled` | `True` | Checks whether PII from the original data (names, emails, etc.) appears verbatim in the synthetic output. |

### PII replacement

Detect and replace personally identifiable information before training, so the model never sees real names, emails, phone numbers, or other sensitive values.

```python
synthesizer = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_replace_pii()
)
```

Calling `with_replace_pii()` with no arguments uses sensible defaults that cover common entity types (names, emails, phone numbers, SSNs, credit cards, and more). Pass a configuration dictionary only if you need to disable or customize specific entity detection:

```python
synthesizer = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_replace_pii(config={"globals": {"classify": {"enable_classify": False}}})
)
```

!!! note
    PII replacement runs on the training data only. The test set is kept untouched so that privacy metrics can accurately measure how well the model protects sensitive information.

### Differential privacy

Add formal, mathematical privacy guarantees to training with DP-SGD. This adds calibrated noise during gradient updates so that no single training record has an outsized influence on the trained model.

```python
synthesizer = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_differential_privacy(dp_enabled=True, epsilon=8.0)
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `dp_enabled` | `False` | Must be set to `True` explicitly. |
| `epsilon` | `8.0` | Privacy budget. Lower values give stronger guarantees but may reduce data quality. |
| `delta` | `"auto"` | Probability of accidental leakage. The `"auto"` setting uses `1/n^1.2` where `n` is the number of training records. |
| `per_sample_max_grad_norm` | `1.0` | Maximum L2 norm for per-sample gradient clipping. |

!!! warning
    Differential privacy requires `max_sequences_per_example=1` in the data config and is incompatible with the Unsloth training backend. These constraints are enforced automatically -- you will get a clear error if they are violated.

### Time series

Enable time-series mode when your data has temporal ordering per entity -- for example, transaction logs, sensor readings, or event sequences.

```python
synthesizer = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_data(group_training_examples_by="device_id")
    .with_time_series(
        is_timeseries=True,
        timestamp_column="event_time",
        timestamp_format="%Y-%m-%d %H:%M:%S",
    )
)
```

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `is_timeseries` | `False` | Enable time-series mode. |
| `timestamp_column` | `None` | Column containing timestamps. Required unless `timestamp_interval_seconds` is set. |
| `timestamp_interval_seconds` | `None` | Fixed interval between records (in seconds). Use this instead of `timestamp_column` for evenly spaced data. |
| `timestamp_format` | `None` (auto-detected) | Python strftime format (e.g., `"%Y-%m-%d %H:%M:%S"`) or `"elapsed_seconds"` for numeric timestamps. |

Time-series mode uses a specialized generation backend and requires `group_training_examples_by` in the data config so the model learns temporal patterns within each entity.

### Enabling synthesis explicitly

Calling `with_train()` or `with_generate()` automatically enables synthesis. The `synthesize()` method is only needed when you want to run with default training and generation settings without configuring either:

```python
synthesizer = SafeSynthesizer().with_data_source(df).synthesize()
```

## Running the Pipeline

### Full pipeline

Best for production or batch jobs. Runs data processing, training, generation, and evaluation end-to-end:

```python
synthesizer = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_train(learning_rate=0.0005)
    .with_generate(num_records=5000)
)
synthesizer.run()
```

### Step-by-step execution

Best when you want to inspect intermediate state -- for example, checking the train/test split before committing to a long training run:

```python
synthesizer = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_train(learning_rate=0.0005)
    .with_generate(num_records=5000)
)

synthesizer.process_data()
# Inspect: synthesizer._train_df, synthesizer._test_df

synthesizer.train()
synthesizer.generate()
synthesizer.evaluate()
```

### PII replacement only

When you need de-identified data without generating synthetic records -- for example, preparing a dataset for sharing with a third party:

```python
synthesizer = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_replace_pii()
)
synthesizer.run()

de_identified_df = synthesizer.results.synthetic_data
```

When `with_replace_pii()` is called without `with_train()`, `with_generate()`, or `synthesize()`, the pipeline applies PII replacement to the full dataset and returns immediately.

### Resume from a saved run

Training is the most expensive step (typically 15--60 minutes). You can reuse a trained adapter to regenerate and re-evaluate without retraining:

```python
synthesizer = SafeSynthesizer(save_path="/path/to/previous/run")
synthesizer.load_from_save_path()

synthesizer.generate()
synthesizer.evaluate()
```

`load_from_save_path()` loads the configuration, model metadata, and cached train/test split from the previous run to ensure evaluation metrics are consistent.

## Working with Results

After the pipeline completes, `synthesizer.results` contains everything you need:

```python
results = synthesizer.results

# The synthetic data as a DataFrame
results.synthetic_data.head()

# Quality and privacy scores
print(f"Quality:  {results.summary.synthetic_data_quality_score}")
print(f"Privacy:  {results.summary.data_privacy_score}")
print(f"Valid:    {results.summary.num_valid_records} / "
      f"{results.summary.num_valid_records + (results.summary.num_invalid_records or 0)}")
```

The `results.summary` object includes:

| Field | Description |
|-------|-------------|
| `synthetic_data_quality_score` | Overall quality score combining distribution and correlation stability |
| `data_privacy_score` | Overall privacy score combining MIA, AIA, and PII replay |
| `membership_inference_protection_score` | How well the data resists membership inference attacks |
| `attribute_inference_protection_score` | How well the data resists attribute inference attacks |
| `column_distribution_stability_score` | How closely synthetic column distributions match the original |
| `column_correlation_stability_score` | How well inter-column correlations are preserved |
| `num_valid_records` / `num_invalid_records` | Count of valid and invalid generated records |
| `valid_record_fraction` | Fraction of generated records that parsed correctly |

### Saving results to disk

```python
synthesizer.save_results()
```

This writes the synthetic CSV and evaluation report HTML to the run's output directory:

```text
<artifact-path>/<config>---<dataset>/<run_name>/
├── safe-synthesizer-config.json
├── train/
│   └── adapter/
├── generate/
│   ├── synthetic_data.csv
│   └── evaluation_report.html
└── dataset/
    ├── training.csv
    └── test.csv
```

You can also pass an explicit output path:

```python
synthesizer.save_results(output_file="my_synthetic_data.csv")
```

## Recipes

### Minimal tabular synthesis

"I have a CSV and want synthetic data with defaults."

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

synthesizer = SafeSynthesizer().with_data_source("customers.csv").synthesize()
synthesizer.run()

synthesizer.results.synthetic_data.to_csv("synthetic_customers.csv", index=False)
```

### Synthesis with PII replacement

"My data contains names, emails, and phone numbers."

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

synthesizer = (
    SafeSynthesizer()
    .with_data_source("patients.csv")
    .with_replace_pii()
    .with_train(learning_rate=0.0005)
    .with_generate(num_records=5000)
)
synthesizer.run()
synthesizer.save_results()
```

### Synthesis with differential privacy

"I need formal privacy guarantees for regulatory compliance."

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

synthesizer = (
    SafeSynthesizer()
    .with_data_source("financial_records.csv")
    .with_replace_pii()
    .with_differential_privacy(dp_enabled=True, epsilon=8.0)
    .with_train(learning_rate=0.0003)
    .with_generate(num_records=5000)
)
synthesizer.run()

print(f"Privacy score: {synthesizer.results.summary.data_privacy_score}")
```

### Time series synthesis

"My data has temporal ordering per entity."

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

synthesizer = (
    SafeSynthesizer()
    .with_data_source("sensor_readings.csv")
    .with_data(group_training_examples_by="device_id")
    .with_time_series(
        is_timeseries=True,
        timestamp_column="reading_time",
        timestamp_format="%Y-%m-%d %H:%M:%S",
    )
    .with_generate(num_records=10000)
)
synthesizer.run()
synthesizer.save_results()
```

### Regenerate from a trained model

"Training took 45 minutes. Now I want to regenerate without retraining."

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

synthesizer = SafeSynthesizer(save_path="/artifacts/my-run/2025_0601_120000")
synthesizer.load_from_save_path()

synthesizer.generate()
synthesizer.evaluate()
synthesizer.save_results()
```

`load_from_save_path()` restores the full configuration, trained adapter, and cached train/test split from the previous run.

## Next Steps

- [CLI Reference](cli.md) -- run the same pipeline from the command line
- [Parameters Reference](parameters.md) -- full list of all configuration fields
- [Evaluation](../product-overview/evaluation.md) -- what each quality and privacy score means
- API Reference: [:material-api: `SafeSynthesizer`](../reference/nemo_safe_synthesizer/sdk/library_builder.md), [:material-api: `ConfigBuilder`](../reference/nemo_safe_synthesizer/sdk/config_builder.md)
