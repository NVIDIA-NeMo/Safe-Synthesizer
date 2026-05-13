---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
hide:
  - navigation
---

# NeMo Safe Synthesizer

NeMo Safe Synthesizer creates private, safe versions of sensitive tabular datasets -- entirely synthetic data with no one-to-one mapping to your original records. It uses LLM fine-tuning with optional [differential privacy](https://desfontain.es/blog/friendly-intro-to-differential-privacy.html) to produce high-quality datasets that preserve the statistical properties and utility of your data for downstream AI tasks while ensuring privacy compliance and protecting sensitive information.

## Key Features

- Privacy-first synthetic data -- PII detection and replacement, optional differential privacy while fine-tuning via [Opacus](https://opacus.ai/)
- LLM fine-tuning -- LoRA fine-tuning optimized for tabular data, including numeric, categorical, and text columns
- Fast inference -- [vLLM](https://github.com/vllm-project/vllm)-powered generation with optional structured output enforcement
- Comprehensive evaluation -- Privacy and quality metrics in an in-depth HTML report
- Flexible interfaces -- CLI for scripting, Python SDK for programmatic workflows, YAML configuration

!!! info "System Requirements"
    NeMo Safe Synthesizer requires a Linux machine with an NVIDIA GPU (A100 80GB+ recommended) and CUDA 12.8+ to run the training and generation pipeline. macOS, Windows, and Apple Silicon are not supported for pipeline execution. A CPU-only install is available for development and configuration validation -- see [Getting Started](user-guide/getting-started.md#install-the-package).

## Next Steps

<div class="grid cards" markdown>

-   **Getting Started**

    ---

    Install the package, set up your environment, and run your first synthetic data pipeline in minutes.

    [:octicons-arrow-right-24: Getting Started](user-guide/getting-started.md)

-   **Product Overview**

    ---

    Learn about the pipeline steps: replace PII, synthesize data, evaluate.

    [:octicons-arrow-right-24: Product Overview](product-overview/pipeline.md)

-   **Tutorials**

    ---

    Follow hands-on tutorials to generate synthetic data.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   **User Guide**

    ---

    Configure and run the pipeline via YAML, CLI, SDK, or environment variables.

    [:octicons-arrow-right-24: User Guide](user-guide/getting-started.md)

-   **Developer Guide**

    ---

    Browse the auto-generated API reference and dive into the architecture details.

    [:octicons-arrow-right-24: Developer Guide](developer-guide/architecture.md)

-   **Developer Notes**

    ---

    Read developer blog posts and check release notes.

    [:octicons-arrow-right-24: Developer Notes](blog/index.md)

</div>

## Telemetry & Privacy

NeMo Safe Synthesizer includes an optional function to share anonymous telemetry data with NVIDIA for product improvement. Data collected is limited to run-level operational metrics (such as final run status, processing time, record and token counts, configuration parameters, top-level quality and privacy scores, base model used, deployment type, and GPU type). No user or device information is collected. This data is used to prioritize product improvements and will be shared in aggregate with the community. It is not used to track any individual user behavior.

You may opt out of telemetry collection at any time. Opting out applies only to data collection by the NeMo Safe Synthesizer library itself. To disable telemetry in a YAML config, set:

```yaml
emit_telemetry: false
```

To disable telemetry for one CLI invocation, pass `--emit_telemetry false`:

```bash
safe-synthesizer run --emit_telemetry false --data-source my_data.csv
```

To disable telemetry for the current shell, set `NEMO_TELEMETRY_ENABLED=false` (other accepted disabling values: `0`, `no`) in your environment before running:

```bash
export NEMO_TELEMETRY_ENABLED=false
```

Use of third-party endpoints, including NVIDIA Build: NeMo Safe Synthesizer can be configured to use various inference endpoints, including build.nvidia.com (NVIDIA Build). If you choose to use NVIDIA Build or any other third-party endpoint, that endpoint's own terms of service and privacy practices apply independently of this library. Any opt-out you exercise within NeMo Safe Synthesizer does not extend to data collection by your chosen endpoint. NVIDIA Build is intended for evaluation and testing purposes only and may not be used in production environments. Do not submit any confidential information or personal data when using NVIDIA Build.

## Contact

- [Need help? Ask us a question](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/discussions)
- [Report a bug](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues/new?template=bug-report.yml)
- [Make a feature request](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues/new?template=feature-request.yml)
- [Report a security vulnerability](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/security/policy)

## License

NeMo Safe Synthesizer is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/LICENSE).
