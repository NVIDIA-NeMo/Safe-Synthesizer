---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
hide:
  - navigation
---

# NeMo Safe Synthesizer

**Generate synthetic data with privacy guarantees.**

NeMo Safe Synthesizer is a comprehensive package for generating safe, synthetic data. It uses LLM fine-tuning with optional differential privacy to produce high-quality synthetic datasets that preserve the statistical properties of your data while protecting individual privacy.

---

## Key Features

- **Privacy-first synthetic data** -- PII detection and replacement, optional differential privacy via Opacus
- **LLM fine-tuning** -- LoRA fine-tuning with HuggingFace or Unsloth backends, quantization support
- **Fast inference** -- VLLM-powered generation with structured output enforcement
- **Comprehensive evaluation** -- Privacy metrics, quality scores, distribution analysis, and HTML reports
- **Flexible interfaces** -- CLI for scripting, Python SDK for programmatic workflows, YAML configuration

---

<div class="grid cards" markdown>

-   **Product Overview**

    ---

    Pipeline, data synthesis, PII replacement, and evaluation.

    [:octicons-arrow-right-24: Product Overview](product-overview/pipeline.md)

-   **Getting Started**

    ---

    Install the package, set up your environment, and run your first synthetic data pipeline in minutes.

    [:octicons-arrow-right-24: Getting Started](getting-started/installation.md)

-   **Tutorials**

    ---

    Step-by-step tutorials to get you up and running.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   **User Guide**

    ---

    Python SDK, CLI reference, parameters, and troubleshooting.

    [:octicons-arrow-right-24: User Guide](user-guide/sdk.md)

-   **Developer Guide**

    ---

    Architecture documentation and auto-generated API reference.

    [:octicons-arrow-right-24: Developer Guide](developer-guide/architecture.md)

-   **Developer Notes**

    ---

    Blog posts and release notes.

    [:octicons-arrow-right-24: Developer Notes](blog/index.md)

</div>

---

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
