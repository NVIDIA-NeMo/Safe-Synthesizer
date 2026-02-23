<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuration

NeMo Safe Synthesizer uses YAML configuration files validated by [Pydantic](https://docs.pydantic.dev/) models.

## Configuration File

Pass a YAML config file to the CLI:

```bash
safe-synthesizer run --config config.yaml --url data.csv
```

## Configuration Sections

The top-level configuration (`SafeSynthesizerParameters`) aggregates the following sections:

### Data Parameters

Controls dataset loading and preprocessing.

<!-- TODO: Document DataParameters fields -->

### Training Hyperparameters

Controls LLM fine-tuning behavior.

<!-- TODO: Document TrainingHyperparams fields (learning_rate, epochs, batch_size, etc.) -->

### Generation Parameters

Controls synthetic data generation.

<!-- TODO: Document GenerateParameters fields (temperature, top_p, num_records, etc.) -->

### Evaluation Parameters

Controls which evaluation components are enabled.

<!-- TODO: Document EvaluationParameters fields -->

### PII Replacer Config

Controls PII detection and replacement behavior.

<!-- TODO: Document PiiReplacerConfig fields -->

### Differential Privacy Hyperparameters

Controls DP-SGD training parameters.

<!-- TODO: Document DifferentialPrivacyHyperparams fields (epsilon, delta, clipping_norm) -->

## Dataset Registry

Safe Synthesizer supports a dataset registry to simplify working with a standard set of datasets. Datasets in the registry can be referenced by name.

### Registry Format

```yaml
base_url: /root/data/location
datasets:
  - name: dataset1
    url: dataset1.csv
  - name: dataset2
    url: dataset2.jsonl
    overrides:
      data:
        group_training_examples_by: id
  - name: dataset3
    url: https://myhost.com/path/to/dataset.json
    load_args:
      keyword: custom_arg_for_data_reader
```

Provide a registry via CLI (`--dataset-registry`) or environment variable (`NSS_DATASET_REGISTRY`).

## API Reference

For the full list of configuration fields, see the auto-generated API docs:

- [:material-api: `SafeSynthesizerParameters`](../reference/nemo_safe_synthesizer/config/parameters.md)
- [:material-api: `DataParameters`](../reference/nemo_safe_synthesizer/config/data.md)
- [:material-api: `TrainingHyperparams`](../reference/nemo_safe_synthesizer/config/training.md)
