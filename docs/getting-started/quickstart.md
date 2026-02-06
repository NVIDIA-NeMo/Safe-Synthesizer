# Quick Start

This guide walks you through running your first synthetic data pipeline with NeMo Safe Synthesizer.

## Running the Full Pipeline

The `run` command executes the complete Safe Synthesizer pipeline -- data processing, training, generation, and evaluation:

```bash
safe-synthesizer run \
  --config config.yaml \
  --url data.csv
```

Without a subcommand, `run` executes all stages end-to-end.

## Running Individual Stages

You can also run stages independently:

### Train only

```bash
safe-synthesizer run train \
  --config config.yaml \
  --url data.csv
```

### Generate only

After training, generate synthetic data using a saved adapter:

```bash
safe-synthesizer run generate \
  --config config.yaml \
  --url data.csv \
  --run-path /path/to/trained/run
```

Or use auto-discovery to find the latest trained adapter:

```bash
safe-synthesizer run generate \
  --config config.yaml \
  --url data.csv \
  --auto-discover-adapter \
  --artifact-path ./safe-synthesizer-artifacts
```

## Using the Python SDK

For programmatic workflows, use the SDK:

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.config import SafeSynthesizerParameters

config = SafeSynthesizerParameters.from_yaml("config.yaml")

synthesizer = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .with_train(learning_rate=0.0001)
    .with_generate(num_records=10000)
    .with_evaluate(enabled=True)
)

synthesizer.run()
results = synthesizer.results
```

## Output Artifacts

After a run, you'll find the following structure in your artifact directory:

```text
<artifact-path>/<config>---<dataset>/<run_name>/
├── safe-synthesizer-config.json
├── train/
│   ├── safe-synthesizer-config.json
│   └── adapter/
├── generate/
│   ├── synthetic_data.csv
│   └── evaluation_report.html
└── dataset/
    ├── training.csv
    ├── test.csv
    └── validation.csv
```

## Next Steps

- [CLI Reference](../user-guide/cli.md) -- Full CLI documentation
- [Configuration](../user-guide/configuration.md) -- YAML config options and environment variables
- [Python SDK](../user-guide/sdk.md) -- Programmatic usage with the builder pattern
