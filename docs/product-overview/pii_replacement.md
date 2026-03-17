<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# PII Replacement

PII (Personally Identifiable Information) replacement is a critical privacy protection step that detects and replaces sensitive information in your datasets before synthesis. This ensures that the model has no chance of learning the most sensitive information like names, addresses, and other identifiers.

## How It Works

The PII replacement pipeline operates in multiple stages:

1. Detection: Classifies PII entities within free text and entire columns.
2. Replacement: Substitutes PII using configurable rules.

## Detection Methods

NeMo Safe Synthesizer supports multiple PII detection approaches described in the table below:

| Method | Scope | Description | Key Features |
|--------|--------|--------------|---------------|
| LLM Classification | Entire columns | Leverages language models for column classification when the entire column is a single entity | - Contextual understanding of entities<br>- Handles complex PII patterns<br>- Flexible entity definitions<br>- Configurable prompts and models |
| [GLiNER PII](https://huggingface.co/nvidia/gliner-PII#evaluation-datasets) | Free text | Uses the GLiNER PII model for entity recognition within free text columns | - Zero-shot entity detection<br>- Supports custom entity types<br>- High accuracy for standard PII categories<br>- Configurable confidence thresholds |


## Replacement Methods

After detection, PII can be handled in multiple ways:

| Strategy   | Description                                  | Example                     |
|------------|----------------------------------------------|-----------------------------|
| Annotate   | Add identified entity to original PII        | Alice → &lt;entity type="first_name" value="Alice"&gt; |
| Redact     | Replace PII with a generic tag               | Alice → &lt;first_name&gt;     |
| Hash       | Replace PII with a hashed value               | Alice → 3bf676c57     |
| Substitute | Replace PII with a context-relevant alternative | Alice → Erica            |

## Supported Entity Types
GLiNER PII will attempt to identify any custom entity type you provide. However, it has specifically been fine-tuned to detect the following entities, organized by category:

### Personal Information
- `first_name` - Given names
- `last_name` - Surnames and family names
- `name` - Full names
- `age` - Ages
- `email` - Email addresses
- `phone_number` - Phone numbers in various formats
- `fax_number` - Fax numbers in various formats



### Addresses

- `address` - Complete physical addresses (for example, 123 Main Street, Anytown, CA 90210)
- `street_address` - Street addresses (for example, 123 Main Street)
- `city` - City names
- `county` - County names
- `state` - State/province names
- `postcode` - Postal/ZIP codes
- `country` - Country names

### Personal Identifiers
- `ssn` - Social Security Numbers
- `national_id` - National ID numbers
- `tax_id` - Tax ID numbers
- `certificate_license_number` - Driver’s license numbers
- `unique_identifier` - Generic unique IDs
- `customer_id` - Customer identifiers
- `employee_id` - Employee identifiers

### Financial Information

- `credit_debit_card` - Credit and debit card numbers
- `cvv` - Credit card verification code
- `pin` - Personal identification numbers
- `account_number` - Bank account numbers
- `bank_routing_number` - Bank routing numbers
- `swift_bic` - Swift/BIC codes
- `iban` - International bank account numbers

### Medical Information

- `medical_record_number` - Medical record numbers
- `health_plan_beneficiary_number` - Insurance IDs
- `biometric_identifier` - Biometric data references

### Technical Identifiers

- `url` - Web URLs
- `ipv4` - IPv4 addresses
- `ipv6` - IPv6 addresses
- `mac_address` - Hardware MAC addresses
- `api_key` - API keys and tokens
- `user_name` - Usernames
- `password` - Passwords
- `http_cookie` - HTTP Cookies
- `device_identifier` - Device IDs

### Vehicle Identifiers

- `vehicle_identifier` - Vehicle identification numbers (VINs)
- `license_plate` - License plates

### Geographic Information

- `latitude` - Latitude coordinates
- `longitude` - Longitude coordinates
- `coordinate` - Coordinate pairs

### Quasi Identifiers

- `date` - Date values
- `date_time` - Date and time values
- `blood_type` - Blood type information
- `gender` - Gender information
- `sexuality` - Sexual orientation
- `political_view` - Political affiliations
- `race` - Race
- `ethnicity` - Ethnicity information
- `religious_belief` - Religious affiliations
- `language` - Language preferences
- `education` - Education level
- `job_title` - Professional titles
- `employment_status` - Employment information
- `company_name` - Organization names

### Custom Entity Types

Beyond these built-in types, you can define custom entities using:

```yaml title="Custom entity configuration"
globals:
  classify:
    enable_classify: true
    entities:
      - first_name
      - last_name
      - email
      - employee_id
      - project_code
```

## Configuration

PII replacement is configured through the `replace_pii` section. For the full schema, refer to [Configuration Reference -- Replacing PII](../user-guide/configuration.md#replacing-pii).

```yaml title="replace_pii section"
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

## When to Use PII Replacement

Consider using PII replacement when:

- Your data contains names, addresses, or other direct identifiers
- Compliance requires PII removal before processing
- You want to ensure the model cannot memorize sensitive values
- You need to share synthetic data with external parties

PII replacement is on by default as a pre-processing step before synthesis.
