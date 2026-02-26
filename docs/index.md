---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
hide:
  - navigation
---

# NeMo Safe Synthesizer

NeMo Safe Synthesizer is a comprehensive package for generating safe, synthetic versions of your data.
It uses LLM fine-tuning with optional [differential privacy](https://desfontain.es/blog/friendly-intro-to-differential-privacy.html) to produce high-quality synthetic datasets that preserve the statistical properties of your data while protecting sensitive information.

## Key Features

- Privacy-first synthetic data -- PII detection and replacement, optional differential privacy via [Opacus](https://github.com/meta-pytorch/opacus)
- LLM fine-tuning -- LoRA fine-tuning optimized for tabular data, including numeric, categorical, and text columns
- Fast inference -- [vLLM](https://github.com/vllm-project/vllm)-powered generation with optional structured output enforcement
- Comprehensive evaluation -- Privacy metrics, quality scores, distribution analysis, and HTML reports
- Flexible interfaces -- CLI for scripting, Python SDK for programmatic workflows, YAML configuration

## Next Steps

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install the package, set up your environment, and run your first synthetic data pipeline in minutes.

    [:octicons-arrow-right-24: Getting Started](getting-started/installation.md)

-   **Product Overview**

    ---

    Data synthesis, PII replacement, evaluation, and pipeline.

    [:octicons-arrow-right-24: Product Overview](product-overview/pipeline.md)

-   **Tutorials**

    ---

    Step-by-step tutorials to get you up and running.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   **User Guide**

    ---

    Python SDK, CLI reference, available parameters, and troubleshooting.

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

## Contact

- [Need help? Ask us a question](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/discussions)
- [Report a bug](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues/new?template=bug-report.yml)
- [Make a feature request](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues/new?template=feature-request.yml)
- [Report a security vulnerability](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/security/policy)

## License

NeMo Safe Synthesizer is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/LICENSE).
