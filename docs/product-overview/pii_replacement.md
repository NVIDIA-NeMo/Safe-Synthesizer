<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Privacy

NeMo Safe Synthesizer provides multiple layers of privacy protection for synthetic data generation.

## PII Replacement

PII (Personally Identifiable Information) replacement is a critical privacy protection step that detects and replaces sensitive information in your datasets before synthesis. This ensures that the model has no chance of learning the most sensitive information like names, addresses, and other identifiers.

### How It Works

The PII replacement pipeline operates in multiple stages:

1. Detection: identifies PII entities using configurable detection methods
2. Classification: categorizes detected entities by type (name, email, address, etc.)
3. Transformation: replaces or redacts PII using configurable rules
4. Validation: verifies that sensitive information has been properly handled

### Detection Methods

Safe Synthesizer supports multiple PII detection approaches:

| Method | Best For | Speed | Accuracy |
|--------|----------|-------|----------|
| **GLiNER** | General PII detection | Fast | High |
| **LLM Classification** | Context-aware, custom entities | Slow | Very High |
| **Regex** | Structured patterns (SSN, phone) | Very Fast | Perfect (for known formats) |

#### GLiNER Detection

Uses the GLiNER model for entity recognition:

- Zero-shot entity detection
- Supports custom entity types
- High accuracy for standard PII categories
- Configurable confidence thresholds

#### LLM Classification

Leverages language models for PII detection:

- Contextual understanding of entities
- Handles complex PII patterns
- Flexible entity definitions
- Configurable prompts and models

#### Regex Detection

Pattern-based detection for structured PII:

- Fast and deterministic
- Ideal for known formats (SSN, phone numbers)
- Customizable patterns
- Low computational overhead

### Replacement Strategies

After detection, PII can be handled in multiple ways:

- Fake Data: generate realistic replacements using the [Faker](https://faker.readthedocs.io/) library
- Redaction: replace with placeholder tokens
- Hashing: one-way hashing for consistency
- Custom Rules: define your own transformation logic

### Supported Entity Types

Safe Synthesizer recognizes many PII types out of the box:

- Personal: names, dates of birth, ages
- Contact: emails, phone numbers, addresses
- Identifiers: SSN, passport numbers, license numbers
- Financial: credit card numbers, bank accounts, IBAN, SWIFT codes
- Medical: patient IDs, medical record numbers
- Digital: IP addresses, URLs, UUIDs, API keys, JWTs
- Custom: define your own entity types

### Configuration

PII replacement is configured through the `replace_pii` section of your YAML config:

```yaml
enable_replace_pii: true
replace_pii:
  globals:
    locales:
      - en_US
  steps:
    - rows:
        update:
          - entity:
              - email
              - phone_number
            value: "column.entity | fake"
```

### When to Use PII Replacement

Consider using PII replacement when:

- Your data contains names, addresses, or other direct identifiers
- Compliance requires PII removal before processing
- You want to ensure the model cannot memorize sensitive values
- You need to share synthetic data with external parties

PII replacement can be used standalone (without synthesis) or as a preprocessing step before synthesis.

---

## Differential Privacy

Safe Synthesizer supports Differential Privacy (DP) training via [Opacus](https://opacus.ai/) integration. DP provides mathematical guarantees that synthetic data doesn't reveal information about individual records in the training data.

### Key Concepts

- Epsilon (epsilon): privacy budget -- lower values mean stronger privacy
    - epsilon = 1: Very strong privacy
    - epsilon = 6-10: Moderate privacy
    - epsilon > 10: Weak privacy
- Delta (delta): probability of privacy guarantee failure. Typically set to 1/n^2 where n is the dataset size (e.g., 1e-5)
- Max Gradient Norm: maximum gradient clipping norm, controls the sensitivity of the training process

### How It Works

DP-SGD (Differentially Private Stochastic Gradient Descent) adds calibrated noise to gradients during training, providing a mathematical privacy guarantee that limits what can be learned about any individual record:

1. Gradient Clipping: per-sample gradients are clipped to a maximum norm
2. Noise Injection: calibrated Gaussian noise is added to the clipped gradients
3. Privacy Accounting: the total privacy budget (epsilon) is tracked across training

### Configuration

Enable differential privacy in your YAML config:

```yaml
differential_privacy:
  dp_enabled: true
  epsilon: 8.0
  delta: 1e-5
  per_sample_max_grad_norm: 1.0
```

Or via the Python SDK:

```python
synthesizer = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .with_train(learning_rate=0.0005)
    .with_differential_privacy(dp_enabled=True, epsilon=8.0, delta=1e-5)
    .with_generate(num_records=5000)
)
synthesizer.run()
```

### Choosing Epsilon

| Sensitivity | Recommended Epsilon | Use Case |
|-------------|-------------------|----------|
| **High** (medical, financial) | 1.0 - 3.0 | Maximum privacy, some utility loss |
| **Medium** (general business) | 3.0 - 6.0 | Good balance of privacy and utility |
| **Low** (public-adjacent data) | 6.0 - 10.0 | Better utility, moderate privacy |

### Privacy Budget Management

!!! warning
    Privacy budget compounds across releases. Each additional synthetic dataset released from the same original data increases the total privacy cost.

Best practices:

1. Single Release: only release one synthetic dataset per original dataset when possible
2. Composition: if multiple releases are needed, divide the privacy budget accordingly
3. Documentation: track all data releases and cumulative privacy budget
4. Renewal: privacy budget doesn't reset -- consider this in your data lifecycle

---

## Privacy Evaluation

After generation, the evaluation system measures privacy through several metrics:

- PII Replay Detection -- Checks if original PII values appear in synthetic data
- Membership Inference Protection -- Measures resistance to membership inference attacks (can an attacker tell if a specific record was in the training data?)
- Attribute Inference Protection -- Measures resistance to attribute inference attacks (can sensitive attributes be inferred from other known attributes?)

See [Evaluation](evaluation.md) for full details on privacy and quality metrics.