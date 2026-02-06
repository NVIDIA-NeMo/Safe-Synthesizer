# Python SDK

The Python SDK provides a programmatic interface for building and running Safe Synthesizer workflows using a fluent builder pattern.

## Basic Usage

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.config import SafeSynthesizerParameters

# Load config from YAML
config = SafeSynthesizerParameters.from_yaml("config.yaml")

# Build and run the pipeline
synthesizer = SafeSynthesizer(config).with_data_source("data.csv")
synthesizer.run()

# Access results
results = synthesizer.results
```

## Builder Pattern

The SDK uses a fluent builder pattern for configuration:

```python
synthesizer = (
    SafeSynthesizer(config)
    .with_data_source(df)
    .with_train(learning_rate=0.0001)
    .with_generate(num_records=10000)
    .with_evaluate(enabled=True)
)
```

## Step-by-Step Execution

For finer control, call individual pipeline stages:

```python
synthesizer = SafeSynthesizer(config).with_data_source(df)

# Run stages individually
synthesizer.process_data()
synthesizer.train()
synthesizer.generate()
synthesizer.evaluate()
```

## Key Classes

| Class | Description |
|-------|-------------|
| `SafeSynthesizer` | Main entry point for SDK workflows |
| `ConfigBuilder` | Base builder for configuration management |
| `JobBuilder` | Builder for job-based workflows |

## API Reference

For complete class and method documentation, see:

- [:material-api: `SafeSynthesizer`](../reference/nemo_safe_synthesizer/sdk/library_builder.md)
- [:material-api: `ConfigBuilder`](../reference/nemo_safe_synthesizer/sdk/config_builder.md)
