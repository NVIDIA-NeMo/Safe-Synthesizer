<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Synthetic Data Quality

Reference for diagnosing and improving synthetic data quality and privacy. Covers differential
privacy errors, PII replacement issues, evaluation metric behavior, and score
interpretation for operational use. For conceptual explanations of what SQS and
DPS measure and how to read the HTML report, see
[Product Overview -- Evaluation](../product-overview/evaluation.md).
For runtime errors, OOM issues, and configuration problems, see
[Program Runtime](troubleshooting.md). For environment variables and model
caching, see [Environment Variables](environment.md).

There is always an inherent trade off between privacy and quality. A high level of privacy protection is achieved simply through the process of generating synthetic data, and is often a sufficient balance between privacy and utility. However, parameter tuning can often help you improve one without sacrificing the other.

---

## Differential Privacy

Differentially private (DP) training has strict requirements. Violating them produces errors that may
not immediately point to the root cause.

### Requirements

For the full list of DP compatibility constraints (`use_unsloth`,
`max_sequences_per_example`, gradient checkpointing), see
[Configuration -- Differential Privacy](configuration.md#differential-privacy).

!!! note
    `data_fraction` and `true_dataset_size` must be available at runtime --
    these are set automatically when running the full pipeline.

### Common DP Errors

```text
Unable to automatically determine a noise multiplier
```

The privacy budget (epsilon) is too low for your dataset size. Either increase
`privacy.epsilon` or add more training records. Generally, you need a minimum of ~10,000 records to achieve reasonable quality when DP is enabled.

```text
Discrete mean differs
```

The [PRV accountant](https://github.com/microsoft/prv_accountant) failed and
the system is falling back to the [Opacus](https://opacus.ai/) RDP accountant.
This is handled automatically -- there is no user-facing config to select the
accountant. The fallback may produce slightly different privacy guarantees,
since the two accountants use different composition methods: PRV uses privacy
loss random variables ([Gopi et al. 2021](https://arxiv.org/abs/2106.02848)),
while RDP uses Rényi divergence
([Mironov 2017](https://arxiv.org/abs/1702.07476)).

```text
Number of entities in dataset is low
```

Small datasets cause poor privacy budget utilization. Consider lowering
`training.batch_size` or adding more records.

---

## PII Replacement

Entity detection and classification issues during PII replacement.

### PII Uses Unexpected Entity Types

If PII replacement is not detecting the entity types you expect, the column
classifier may have failed silently. When the classifier fails to initialize
or classify, it falls back to default entity types.

Look for the following log lines if PII replacement seems to use unexpected entity types:

```text
Could not initialize column classifier, falling back to default entities.
```

or

```text
Could not perform classify, falling back to default entities.
```

Fix: set entity types explicitly in your config, or check that `NIM_ENDPOINT_URL`
is reachable. PII classify config is deeply nested -- use YAML or SDK:

=== "Config reference"

    ```yaml
    replace_pii:
      globals:
        classify:
          enable_classify: true
          entities: ["name", "email", "phone_number"]
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig

    pii_config = PiiReplacerConfig.get_default_config()
    pii_config.globals.classify.enable_classify = True
    pii_config.globals.classify.entities = ["name", "email", "phone_number"]

    synthesizer = (
        SafeSynthesizer(config)
        .with_data_source("data.csv")
        .with_replace_pii(config=pii_config)
    )
    ```

---

## Evaluation

For out-of-memory errors during evaluation, see
[Troubleshooting > OOM During Evaluation](troubleshooting.md#out-of-memory-during-evaluation).

### Minimum Data Requirements

Several evaluation metrics have minimum data requirements:

| Metric | Minimum | Behavior if Unmet |
|--------|---------|-------------------|
| Holdout split | 200 records | Raises `ValueError` (pipeline stops) |
| Text semantic similarity | 200 records | Skipped; score marked UNAVAILABLE |
| Attribute Inference Protection | FAISS installed + `evaluation.quasi_identifier_count` columns (default 3; auto-reduced for smaller datasets) | Skipped if FAISS missing; UNAVAILABLE if too few columns |
| Deep Structure Stability | 2x2 matrix | Skipped with warning; score marked UNAVAILABLE |

### UNAVAILABLE Metrics

`UNAVAILABLE` is the literal string that appears in the evaluation report when
a metric could not be computed. Many evaluation components catch errors and
return this grade instead of failing the pipeline.

Common reasons a metric shows `UNAVAILABLE`:

- Column type mismatch -- [`ColumnDistribution`][nemo_safe_synthesizer.evaluation.components.column_distribution.ColumnDistribution], [`DeepStructure`][nemo_safe_synthesizer.evaluation.components.deep_structure.DeepStructure] (PCA), and
  [`Correlation`][nemo_safe_synthesizer.evaluation.components.correlation.Correlation] apply only to numeric and categorical columns; [`TextSemanticSimilarity`][nemo_safe_synthesizer.evaluation.components.text_semantic_similarity.TextSemanticSimilarity]
  and [`TextStructureSimilarity`][nemo_safe_synthesizer.evaluation.components.text_structure_similarity.TextStructureSimilarity] apply only to text columns. A dataset with no
  text columns will show `UNAVAILABLE` for text metrics, and vice versa. This is
  by design.
- No holdout split -- [`TextSemanticSimilarity`][nemo_safe_synthesizer.evaluation.components.text_semantic_similarity.TextSemanticSimilarity] and [`MembershipInferenceProtection`][nemo_safe_synthesizer.evaluation.components.membership_inference_protection.MembershipInferenceProtection]
  both require a held-out test set. If `data.holdout` is `0` (no holdout), these
  metrics are skipped and marked `UNAVAILABLE`.
- Too few records or columns -- see the minimums table above.
- Model download failure -- the SentenceTransformer model must be present in
  your Hugging Face cache (`$HF_HOME`, default `~/.cache/huggingface`). Run
  once with internet access before switching to offline mode.

If the reason is not obvious, check the logs for warnings and exceptions logged
during the evaluation stage.

### Report Truncation

SQS reports are limited to `sqs_report_columns=250` columns and
`sqs_report_rows=5000` rows by default. Larger datasets are silently
truncated in the HTML report. Adjust these in `evaluation` config if needed.
See [Configuration -- Evaluation](configuration.md#evaluation) for the full list of `evaluation` fields.

### Low SQS Scores

If the SQS (Synthetic Quality Score) report shows low quality scores:

1. Review column distributions in the HTML report -- large divergences
   indicate the model did not learn the data patterns well
2. Check that training data is representative and not too small
3. Consider increasing `generation.num_records` for a larger sample
4. Modify `training.num_input_records_to_sample` -- this controls how much
   data the model sees during training (analogous to training duration) and
   affects generation quality. Increasing it is usually the first thing to try,
   but note that very small input datasets can lead to over-training, so
   try both increasing and decreasing it if quality remains poor

---

## Interpreting Results

For a conceptual overview of evaluation metrics -- what SQS and DPS measure,
how to read the HTML report, and what score ranges indicate -- see
[Product Overview -- Evaluation](../product-overview/evaluation.md).
