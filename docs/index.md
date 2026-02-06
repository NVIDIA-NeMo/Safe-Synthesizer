---
hide:
  - navigation
---

# NeMo Safe Synthesizer

**Generate synthetic data with privacy guarantees.**

NeMo Safe Synthesizer is a comprehensive package for generating safe synthetic data. It uses LLM fine-tuning with optional differential privacy to produce high-quality synthetic datasets that preserve the statistical properties of your data while protecting individual privacy.

---

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install the package, set up your environment, and run your first synthetic data pipeline in minutes.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   **User Guide**

    ---

    Learn how to use the CLI, configure pipelines, and work with the Python SDK.

    [:octicons-arrow-right-24: User Guide](user-guide/pipeline.md)

-   **Architecture**

    ---

    Understand the pipeline design, component architecture, and key design patterns.

    [:octicons-arrow-right-24: Architecture](architecture/design.md)

-   **API Reference**

    ---

    Auto-generated reference documentation from source code docstrings.

    [:octicons-arrow-right-24: API Reference](reference/nemo_safe_synthesizer/sdk/library_builder.md)

</div>

---

## Key Features

- **Privacy-first synthetic data** -- PII detection and replacement, optional differential privacy via Opacus
- **LLM fine-tuning** -- LoRA fine-tuning with HuggingFace or Unsloth backends, quantization support
- **Fast inference** -- VLLM-powered generation with structured output enforcement
- **Comprehensive evaluation** -- Privacy metrics, quality scores, distribution analysis, and HTML reports
- **Flexible interfaces** -- CLI for scripting, Python SDK for programmatic workflows, YAML configuration

## Quick Example

```bash
# Install and bootstrap
make bootstrap-tools
make bootstrap-nss cpu

# Run the full pipeline
safe-synthesizer run --config config.yaml --url data.csv
```

Or use the Python SDK:

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.config import SafeSynthesizerParameters

config = SafeSynthesizerParameters.from_yaml("config.yaml")
synthesizer = SafeSynthesizer(config).with_data_source(df)
synthesizer.run()
results = synthesizer.results
```

---

## License

NeMo Safe Synthesizer is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/LICENSE).
