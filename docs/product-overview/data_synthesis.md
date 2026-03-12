<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Data Synthesis

The synthesizer component is the core of the NeMo Safe Synthesizer product. It uses LLM-based fine-tuning to generate realistic synthetic data that maintains the utility of your original dataset while providing privacy protection.

## How It Works

NeMo Safe Synthesizer employs a novel approach to synthetic data generation:

1. Tabular Fine-Tuning: Fine-tunes a language model on your tabular data to learn patterns, correlations, and statistical properties
2. Generation: Uses the fine-tuned model to generate new synthetic records that maintain data utility
3. Privacy Protection: Optionally applies differential privacy during training for mathematical privacy guarantees

Creating synthetic versions of private data allows you to unlock insights without compromising privacy, enabling downstream use cases like AI model training and analytics.

## Key Features

### LLM-Based Fine-Tuning

NeMo Safe Synthesizer adapts language models to understand and generate tabular data:

- Converts tabular data into text sequences suitable for LLM training
- Fine-tunes on your dataset to capture patterns and correlations
- Generates new records that maintain statistical properties with no one-to-one mapping to original records
- Supports various model sizes and architectures

### Supported Data Types

NeMo Safe Synthesizer supports diverse tabular data:

- Numeric: Continuous and discrete numerical values
- Categorical: Text labels and categories
- Text: Free-form text fields
- Temporal: Event sequences and time series (Note: Temporal dataset support is currently experimental.)

### Differential Privacy

A high level of privacy protection is achieved simply through the process of generating synthetic data, and is often a sufficient balance between privacy and utility. For use cases that require maximum privacy, you can fine-tune with differential privacy.

Differential privacy (DP) is the gold standard for privacy protection, providing mathematical guarantees that individual records cannot be identified. When you enable DP, NeMo Safe Synthesizer introduces calibrated noise during model training to ensure that the synthetic data generation process is provably private.

#### Mathematical Guarantee

DP ensures that the output of an algorithm is nearly identical whether or not any single record is included in the training data:

$$
P[M(D_1) \in S] \le \exp(\varepsilon) \cdot P[M(D_2) \in S] + \delta
$$

Where:

- $M$ is the mechanism (process of training the model)
- $D_1$ and $D_2$ are datasets differing by one record
- $\varepsilon$ (epsilon) controls privacy loss -- lower values provide stronger privacy
- $\delta$ (delta) is the failure probability
- $S$ is any subset of possible outputs

#### DP-SGD Implementation

NeMo Safe Synthesizer uses Differentially Private Stochastic Gradient Descent (DP-SGD) to add privacy guarantees during model training:

1. Per-sample gradient computation: Calculate gradients for each training example individually
2. Gradient clipping: Clip L2 norm to `per_sample_max_grad_norm` to bound sensitivity
3. Noise injection: Add calibrated Gaussian noise to gradients based on privacy budget
4. Privacy accounting: Track cumulative privacy loss using Rényi Differential Privacy (RDP)

By default, `record-level` differential privacy is used. When `group_training_examples_by` is set, `group-level` privacy applies -- guarantees cover entire groups of records rather than individual records.

#### Privacy vs Utility Trade-off

Enabling DP provides strong privacy guarantees but affects synthetic data quality:

- Lower epsilon: stronger privacy, but more noise and potentially lower utility
- Training speed: DP training is usually 2-4x slower due to per-sample gradient computation
- Data requirements: DP works best with larger datasets (10,000+ records recommended, compared to 1,000+ records without DP)
- Quality impact: Added noise may reduce statistical fidelity of synthetic data

#### Configuration Parameters

For the complete list of configuration parameters, see the [Parameters Reference](../user-guide/configuration.md). Some commonly used parameters are listed below:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dp_enabled` | bool | `false` | Enable differential privacy |
| `epsilon` | float | `8.0` | Privacy budget (lower = more private, typical range: 4-12) |
| `delta` | float/auto | `"auto"` | Failure probability (auto = 1/n^1.2 based on dataset size) |
| `per_sample_max_grad_norm` | float | `1.0` | Gradient clipping threshold |

#### Recommendations

Starting point: Begin with $\varepsilon \in [8, 12]$ and reduce as needed based on privacy requirements and acceptable quality trade-offs.

Delta calculation: Use `"auto"` (recommended), which sets $\delta = \frac{1}{n^{1.2}}$ based on dataset size $n$. Manual values are typically between 1e-6 and 1e-4.

Data size: DP performs best with 10,000+ training records. Smaller datasets may experience significant quality degradation due to the noise required for privacy guarantees.

For hands-on guidance, refer to [Differential Privacy Tutorial](../tutorials/differential-privacy.md). For complete parameter documentation, refer to [Parameters Reference](../user-guide/configuration.md).

## Configuration

Synthesis behavior is controlled through configuration parameters:

- Training: Model selection, training parameters, sequence configuration
- Generation: Number of records, temperature, sampling strategies
- Privacy: Differential privacy parameters (epsilon, delta, clipping)

For a complete list of all available parameters and their defaults, refer to [Parameters Reference](../user-guide/configuration.md).

## Related Topics

- [Parameters Reference](../user-guide/configuration.md): Complete parameter reference
- [Differential Privacy](../tutorials/differential-privacy.md): Learn about differential privacy in practice
- [Tutorials](../tutorials/index.md): Run the pipeline

