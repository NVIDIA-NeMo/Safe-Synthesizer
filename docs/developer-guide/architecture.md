<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Architecture

<!-- Migrated from design.md in the repository root -->

## Overview

NeMo Safe Synthesizer is a comprehensive package for generating safe synthetic data with privacy guarantees. The architecture follows a pipeline design with configurable stages for data processing, PII replacement, training, generation, and evaluation.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph entryPoints [Entry Points]
        CLI["CLI Interface"]
        SDK["SDK Interface"]
    end

    subgraph configLayer [Configuration Layer]
        ConfigBuilder["ConfigBuilder"]
        SafeSynthesizerParams["SafeSynthesizerParameters"]

        subgraph configComponents [Config Components]
            DataConfig["DataParameters"]
            TrainConfig["TrainingHyperparams"]
            GenConfig["GenerateParameters"]
            EvalConfig["EvaluationParameters"]
            PIIConfig["PiiReplacerConfig"]
            DPConfig["DifferentialPrivacyHyperparams"]
        end
    end

    subgraph dataProcessing [Data Processing Pipeline]
        DataSource["Input Data"]
        Holdout["Holdout - Train/Test Split"]
        PIIReplacer["PII Replacer - NemoPII"]
        DataActions["ActionExecutor"]
        Assembler["ExampleAssembler"]
    end

    subgraph trainingBackend [Training Backend]
        TrainingBackendBase["TrainingBackend - Abstract"]
        HFBackend["HuggingFaceBackend"]
        UnslothBackend["UnslothBackend"]

        subgraph trainingComponents [Training Components]
            ModelLoader["Model Loader"]
            QuantizationComp["Quantization 4-bit/8-bit"]
            LoRA["LoRA Config - PEFT"]
            DPTrainer["Differential Privacy"]
            Callbacks["Training Callbacks"]
        end
    end

    subgraph generationBackend [Generation Backend]
        GenBackend["GeneratorBackend - Abstract"]
        VLLMBackend["VllmBackend"]

        subgraph generationComponents [Generation Components]
            RegexManager["RegexManager"]
            BatchGen["BatchGenerator"]
            Processors["Processors"]
            Stopping["Stopping Criteria"]
        end
    end

    subgraph evaluationSystem [Evaluation System]
        EvaluatorComp["Evaluator"]

        subgraph evaluationComponents [Evaluation Components]
            DataPrivacy["Data Privacy Score"]
            PIIReplay["PII Replay Detection"]
            MembershipInf["Membership Inference Protection"]
            AttributeInf["Attribute Inference Protection"]
            Distribution["Column Distributions"]
            Correlation["Correlations"]
            TextSimilarity["Text Semantic Similarity"]
            StructureSimilarity["Text Structure Similarity"]
            SQS["SQS Score"]
        end

        subgraph reporting [Reporting]
            ReportGen["Report Generator"]
            HTMLReport["HTML Report"]
        end
    end

    CLI --> ConfigBuilder
    SDK --> ConfigBuilder
    ConfigBuilder --> SafeSynthesizerParams
    SafeSynthesizerParams --> DataConfig
    SafeSynthesizerParams --> TrainConfig
    SafeSynthesizerParams --> GenConfig
    SafeSynthesizerParams --> EvalConfig
    SafeSynthesizerParams --> PIIConfig
    SafeSynthesizerParams --> DPConfig

    DataSource --> Holdout
    Holdout --> PIIReplacer
    PIIReplacer --> DataActions
    DataActions --> Assembler
    Assembler --> TrainingBackendBase

    TrainingBackendBase --> HFBackend
    TrainingBackendBase --> UnslothBackend
    HFBackend --> ModelLoader
    HFBackend --> QuantizationComp
    HFBackend --> LoRA
    HFBackend --> DPTrainer
    HFBackend --> Callbacks

    HFBackend --> GenBackend
    GenBackend --> VLLMBackend
    VLLMBackend --> RegexManager
    VLLMBackend --> BatchGen
    VLLMBackend --> Processors
    VLLMBackend --> Stopping

    VLLMBackend --> EvaluatorComp
    EvaluatorComp --> DataPrivacy
    EvaluatorComp --> PIIReplay
    EvaluatorComp --> MembershipInf
    EvaluatorComp --> AttributeInf
    EvaluatorComp --> Distribution
    EvaluatorComp --> Correlation
    EvaluatorComp --> TextSimilarity
    EvaluatorComp --> StructureSimilarity
    EvaluatorComp --> SQS

    EvaluatorComp --> ReportGen
    ReportGen --> HTMLReport
```

---

## Configuration System

Two paths produce a `SafeSynthesizerParameters` object: the CLI path (via Click
decorators and YAML merging) and the SDK path (via the builder pattern). Both
converge on the same Pydantic model and handle nullable sub-configs
(`replace_pii`, `privacy`) uniformly -- `None` means disabled.

```mermaid
flowchart TB
    subgraph cli [CLI Entry Points]
        run_cmd["safe-synthesizer run"]
        train_cmd["safe-synthesizer run train"]
        gen_cmd["safe-synthesizer run generate"]
        val_cmd["safe-synthesizer config validate"]
        mod_cmd["safe-synthesizer config modify"]
        create_cmd["safe-synthesizer config create"]
    end

    subgraph decorators [Decorator Layer]
        common["@common_run_options"]
        pydantic["@pydantic_options SSP"]
    end

    subgraph collector ["pydantic_click_options.py"]
        collect["_collect_params"]
        leaf["LeafParam"]
        flag["FlagParam"]
        parse["parse_overrides"]
    end

    subgraph settings [CLI Settings]
        clisettings["CLISettings.from_cli_kwargs"]
        common_setup["common_setup"]
    end

    subgraph merge [Config Assembly]
        merge_overrides["merge_overrides"]
        model_validate["SSP.model_validate"]
        from_yaml["SSP.from_yaml"]
    end

    subgraph sdk [SDK Entry Point]
        builder["SafeSynthesizer / ConfigBuilder"]
        with_methods["with_replace_pii / with_privacy / with_train / ..."]
        resolve["_resolve_nss_config"]
    end

    subgraph config [SafeSynthesizerParameters]
        data_p["DataParameters"]
        training_p["TrainingHyperparams"]
        gen_p["GenerateParameters"]
        eval_p["EvaluationParameters"]
        pii_p["PiiReplacerConfig | None"]
        dp_p["DifferentialPrivacyHyperparams | None"]
        ts_p["TimeSeriesParameters"]
    end

    subgraph runtime [Runtime Checks]
        pii_check["replace_pii is not None"]
        dp_check["privacy is not None"]
    end

    run_cmd & train_cmd & gen_cmd --> common
    run_cmd & train_cmd & gen_cmd & val_cmd & mod_cmd & create_cmd --> pydantic

    pydantic --> collect
    collect --> leaf & flag

    flag -->|"--no_replace_pii"| parse
    flag -->|"--no_privacy"| parse
    leaf -->|"--training__lr etc"| parse

    parse -->|"overrides dict"| clisettings
    clisettings --> common_setup
    common_setup --> merge_overrides

    val_cmd & mod_cmd & create_cmd -->|"direct"| merge_overrides

    merge_overrides --> from_yaml
    merge_overrides --> model_validate
    model_validate --> config

    builder --> with_methods
    with_methods --> resolve
    resolve -->|"SSP constructor"| config

    config --> data_p & training_p & gen_p & eval_p & pii_p & dp_p & ts_p

    pii_p --> pii_check
    dp_p --> dp_check
```

Configuration precedence (highest to lowest):

1. CLI flags / SDK `with_*()` overrides
2. Dataset registry overrides
3. YAML config file
4. Pydantic model defaults (including `default_factory`)

Nullable sub-configs (`PiiReplacerConfig | None`, `DifferentialPrivacyHyperparams | None`)
use `None` as the sole disabled signal. The `@pydantic_options` decorator auto-generates
`--no_<field>` is-flags for these fields; `parse_overrides` translates them into
`{field: None}` in the overrides dict.

---

## Execution Flow

```mermaid
sequenceDiagram
    participant User
    participant CLI_SDK as CLI/SDK
    participant ConfigBuilder
    participant Holdout
    participant PIIReplacer
    participant Assembler
    participant Training
    participant Generation
    participant Evaluation
    participant Report

    User->>CLI_SDK: Input Data + Config
    CLI_SDK->>ConfigBuilder: Build Configuration
    ConfigBuilder->>Holdout: Split Train/Test
    Holdout->>PIIReplacer: Process Training Data
    Note over PIIReplacer: On by default: replace PII entities with synthetic values
    PIIReplacer->>Assembler: Tokenize and Assemble
    Note over Assembler: Create training examples with proper formatting
    Assembler->>Training: Training Dataset
    Note over Training: Fine-tune LLM with LoRA + optional DP
    Training->>Generation: Adapter Path
    Note over Generation: Generate synthetic records using VLLM backend
    Generation->>Evaluation: Synthetic Data
    Note over Evaluation: Run all evaluation components and metrics
    Evaluation->>Report: Evaluation Results
    Report->>User: HTML Report + Synthetic Data
```

## Simple Overview

```mermaid
flowchart LR
    B[("data")]
    B --> C("PII replacement\non by default")
    C --> D("assemble examples")
    D --> E("Fine-tune")
    E --> F["Generate Samples"]
    F --> G["Evaluate"]
```

---

## Component Details

### 1. Configuration Layer

Path: `src/nemo_safe_synthesizer/config/`

- SafeSynthesizerParameters: main configuration class that aggregates all parameters
- DataParameters: dataset and preprocessing configurations
- TrainingHyperparams: training settings (learning rate, epochs, batch size, etc.)
- GenerateParameters: generation settings (temperature, top_p, num_records, etc.)
- EvaluationParameters: evaluation component toggles and settings
- PiiReplacerConfig: PII detection and replacement settings
- DifferentialPrivacyHyperparams: DP training parameters (epsilon, delta, clipping norm)

### 2. Data Processing Pipeline

Path: `src/nemo_safe_synthesizer/data_processing/`

- Holdout (`holdout/`): splits data into train/test sets with stratification support
- NemoPII (`pii_replacer/`): detects PII entities (names, emails, SSN, etc.) and replaces with synthetic but realistic values
- ActionExecutor (`actions/`): executes data transformations (date normalization, distributions)
- ExampleAssembler (`assembler.py`): converts records to JSON format, tokenizes for model training, handles truncation and padding

### 3. Training Backend

Path: `src/nemo_safe_synthesizer/training/`

| Backend | Description |
|---------|-------------|
| **HuggingFaceBackend** | Quantization (4-bit, 8-bit), LoRA via PEFT, Differential Privacy via Opacus |
| **UnslothBackend** | Optimized training with Unsloth library |

### 4. Generation Backend

Path: `src/nemo_safe_synthesizer/generation/`

- VllmBackend: fast inference using VLLM with LoRA adapter support
- RegexManager: enforces structured output (JSON format)
- BatchGenerator: manages batch generation with retry logic
- Processors: post-processing of generated text

### 5. Evaluation System

Path: `src/nemo_safe_synthesizer/evaluation/`

Components include: Data Privacy Score, PII Replay Detection, Membership Inference Protection, Attribute Inference Protection, Column Distributions, Correlations, Text Semantic Similarity, Text Structure Similarity, and SQS Score.

### 6. Supporting Modules

- LLM Utilities (`llm/`): model metadata, loading, and memory management
- Privacy Module (`privacy/dp_transformers/`): Opacus integration for DP-SGD
- Artifacts (`artifacts/`): data quality checks, field analysis, metadata management
- Records System (`data_processing/records/`): JSON record and fragment handling

---

## Key Design Patterns

### Builder Pattern

The `ConfigBuilder` and `SafeSynthesizer` classes use the builder pattern for fluent configuration:

```python
synthesizer = (
    SafeSynthesizer(config)
    .with_data_source(df)
    .with_train(learning_rate=0.0001)
    .with_generate(num_records=10000)
    .with_evaluate(enabled=True)
)
synthesizer.run()
```

### Backend Abstraction

Training and generation backends use abstract base classes to allow multiple implementations:

- Training: HuggingFace, Unsloth
- Generation: VLLM (extensible to others)

### Component-Based Evaluation

Evaluation uses a modular component system where each metric is a separate component that can be enabled/disabled.

### Pipeline Architecture

The execution follows a clear pipeline: Data --> PII Replacement --> Training --> Generation --> Evaluation

---

## Technology Stack

| Category | Tools |
|----------|-------|
| ML Frameworks | PyTorch, Transformers, PEFT (LoRA) |
| Inference | VLLM |
| Privacy | Opacus for Differential Privacy |
| Data | Pandas, HuggingFace Datasets |
| Config | Pydantic |
| CLI | Click |
| Visualization | Jinja2, HTML/CSS/JS |

---

## Extension Points

1. Custom Training Backend: implement `TrainingBackend` abstract class
2. Custom Generation Backend: implement `GeneratorBackend` abstract class
3. Custom Evaluation Component: extend `Component` base class
4. Custom Data Actions: add to `data_processing/actions/`
5. Custom PII Detectors: extend NER pipeline
