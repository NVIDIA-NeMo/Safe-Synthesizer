# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation results as structured data, which the HTML report is rendered from and other interfaces can present.

``schema_version`` is raised only when a field is removed or changes meaning; readers ignore unknown fields.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

REPORT_DATA_SCHEMA_VERSION = 1


class ReportDataModel(BaseModel):
    """Base of the report data models: ignores unknown fields so older readers accept newer data."""

    model_config = ConfigDict(extra="ignore")


class MetricScore(ReportDataModel):
    """A metric's score and grade."""

    raw_score: float | None = Field(description="Unscaled metric value, or null if it could not be computed.")
    score: float | None = Field(description="Score from 0 to 10, or null if it could not be computed.")
    grade: str = Field(description="Qualitative grade, such as 'Unavailable', 'Poor', 'Good', or 'Excellent'.")
    notes: str | None = Field(description="Warning or error about the score, or null.")


class Distribution(ReportDataModel):
    """Percentage of training and synthetic values in each category or bin."""

    labels: list[str] = Field(
        description=(
            "Category names, with categories under 5 training records folded into one 'Other (N categories)' label, "
            "or bin labels such as '[0, 10)', or '3' for a one-value bin."
        )
    )
    bin_edges: list[float] | None = Field(
        description="For numeric values, the increasing bin edges, one more than the labels; null for categories."
    )
    training_percent: list[float] = Field(description="Percentage of training values per label.")
    synthetic_percent: list[float] = Field(description="Percentage of synthetic values per label.")


class Points(ReportDataModel):
    """Points on the first two principal components."""

    x: list[float] = Field(description="First principal component of each point.")
    y: list[float] = Field(description="Second principal component of each point.")


class Projection(ReportDataModel):
    """Training and synthetic records projected onto the same two principal components."""

    axes: list[str] = Field(description="Names of the two components, such as 'pc1', with any explained variance.")
    training: Points = Field(description="Training records.")
    synthetic: Points = Field(description="Synthetic records.")


class Metric(ReportDataModel):
    """A metric's score."""

    score: MetricScore = Field(description="The metric's score.")


class CorrelationMatrices(ReportDataModel):
    """Pairwise column correlations, with rows and columns in the order of ``columns``."""

    columns: list[str] = Field(description="Column names.")
    training: list[list[float | None]] = Field(description="Training correlations, or null where undefined.")
    synthetic: list[list[float | None]] = Field(description="Synthetic correlations, or null where undefined.")
    difference: list[list[float | None]] = Field(description="Absolute difference of the two, or null where undefined.")


class CorrelationMetric(Metric):
    """Column correlation stability, with the correlations it compares."""

    correlations: CorrelationMatrices | None = Field(description="Correlation matrices, or null if not computed.")


class DeepStructureMetric(Metric):
    """Deep structure stability, with the principal component projection it compares."""

    projection: Projection | None = Field(
        description="Projected records, sampled to at most 5000 per dataset, or null if not computed."
    )


class TextSemanticMetric(Metric):
    """Text semantic similarity, with each text column's embeddings projected onto two principal components."""

    projections: dict[str, Projection] = Field(description="Projected embeddings by text column.")


class TextStructureDistributions(ReportDataModel):
    """Distributions of per-record text statistics in one column."""

    sentence_count: Distribution | None = Field(description="Sentences per record.")
    words_per_sentence: Distribution | None = Field(description="Average words per sentence.")
    characters_per_word: Distribution | None = Field(description="Average characters per word.")


class TextStructureMetric(Metric):
    """Text structure similarity, with the statistics it compares."""

    columns: dict[str, TextStructureDistributions] = Field(description="Text statistics by text column.")


class AttackOutcome(ReportDataModel):
    """How many membership inference attack evaluations got one grade."""

    grade: str = Field(description="Protection grade, such as 'Poor' or 'Excellent'.")
    count: int | None = Field(description="Attack evaluations with this grade, or null if only percentages exist.")
    percentage: float | None = Field(description="Share of attack evaluations with this grade.")


class MembershipInferenceMetric(Metric):
    """Membership inference protection, with its attack outcomes and the columns left out of the attack."""

    outcomes: list[AttackOutcome] = Field(description="Outcomes from the best grade to the worst.")
    excluded_columns: list[str] = Field(description="Columns left out via `evaluation.mia_excluded_columns`.")


class ColumnRisk(ReportDataModel):
    """Attribute inference risk of one column."""

    column: str = Field(description="Column name.")
    risk: float | None = Field(description="Protection from 0 to 10; higher is better.")
    grade: str = Field(description="Protection grade.")


class AttributeInferenceMetric(Metric):
    """Attribute inference protection, with the protection of each column."""

    columns: list[ColumnRisk] = Field(description="Columns from least to most protected.")


class PIIReplayEntry(ReportDataModel):
    """Replay counts for one PII type in one column."""

    column_name: str = Field(description="Column name.")
    column_assigned_type: str = Field(description="Column type assigned by PII detection.")
    pii_type: str = Field(description="PII type, or 'none' for a whole column checked by name.")
    total_training_data: int = Field(description="Training rows containing a value of this type.")
    unique_training_data: int = Field(description="Distinct training values of this type.")
    total_synthetic_data: int = Field(description="Synthetic rows containing one of the training values.")
    unique_synthetic_data: int = Field(description="Distinct training values found in the synthetic data.")
    unique_synthetic_data_percentage: float = Field(description="unique_synthetic_data / unique_training_data * 100.")


class PIIReplayMetric(Metric):
    """How many training PII values reappear in the synthetic data."""

    enabled: bool | None = Field(
        description="Whether `evaluation.pii_replay_enabled` was on, or null when the evaluation had no configuration."
    )
    training_total_records: int = Field(description="Training rows checked.")
    synthetic_total_records: int = Field(description="Synthetic rows checked.")
    entries: list[PIIReplayEntry] = Field(description="Replay counts per column and PII type.")
    skipped_reason: str | None = Field(description="Why PII replay did not run in full, or null.")


class TimeSeriesMetric(Metric):
    """A time-series metric's score and the values its charts are drawn from."""

    details: dict[str, Any] = Field(
        description=(
            "Per-group and per-column results, skipped comparisons, and summaries. Their shape is not yet covered "
            "by `schema_version` and may change between releases."
        )
    )


class Metrics(ReportDataModel):
    """Per-metric results. A metric is null when it is not part of the evaluation; a metric that was turned off or
    could not run is present with an 'Unavailable' grade.
    """

    column_correlation_stability: CorrelationMetric | None = Field(description="Column Correlation Stability.")
    deep_structure_stability: DeepStructureMetric | None = Field(description="Deep Structure Stability.")
    column_distribution_stability: Metric | None = Field(
        description="Column Distribution Stability; each column's distributions are in `columns`."
    )
    text_semantic_similarity: TextSemanticMetric | None = Field(description="Text Semantic Similarity.")
    text_structure_similarity: TextStructureMetric | None = Field(description="Text Structure Similarity.")
    membership_inference_protection: MembershipInferenceMetric | None = Field(
        description="Membership Inference Protection."
    )
    attribute_inference_protection: AttributeInferenceMetric | None = Field(
        description="Attribute Inference Protection."
    )
    pii_replay: PIIReplayMetric | None = Field(description="PII Replay.")
    autocorrelation_similarity: TimeSeriesMetric | None = Field(description="Autocorrelation Similarity.")
    window_membership_inference_protection: TimeSeriesMetric | None = Field(
        description="Window Membership Inference Protection."
    )


class Scores(ReportDataModel):
    """Overall scores."""

    synthetic_quality: MetricScore | None = Field(description="Synthetic Quality Score.")
    data_privacy: MetricScore | None = Field(description="Data Privacy Score.")


class DatasetCounts(ReportDataModel):
    """Row, column, and missing-value counts for the training and synthetic data."""

    training_rows: int = Field(description="Training rows evaluated.")
    training_cols: int = Field(description="Training columns evaluated.")
    training_missing: int = Field(description="Percentage of missing training values.")
    synthetic_rows: int = Field(description="Synthetic rows evaluated.")
    synthetic_cols: int = Field(description="Synthetic columns evaluated.")
    synthetic_missing: int = Field(description="Percentage of missing synthetic values.")
    memorized_lines: int = Field(description="Training rows duplicated in the synthetic data.")


class ColumnProfile(ReportDataModel):
    """Summary statistics of one column in one dataset."""

    type: str = Field(description="Inferred column type, such as 'numeric', 'categorical', or 'text'.")
    count: int = Field(description="Non-null values.")
    unique_count: int = Field(description="Distinct non-null values.")
    unique_percent: float | None = Field(description="Distinct values as a percentage of non-null values.")
    missing_count: int = Field(description="Null values.")
    missing_percent: float | None = Field(description="Null values as a percentage of all values.")
    avg_str_length: float = Field(description="Mean length of the values as strings, or 0 without values.")


class ColumnPIIStatistics(ReportDataModel):
    """PII detected in a training column and how it was replaced."""

    assigned_type: str | None = Field(description="Column type assigned by PII detection.")
    assigned_entity: str | None = Field(description="Entity assigned to the column.")
    detected_entity_counts: dict[str, int] = Field(description="Times each entity type was detected.")
    is_transformed: bool = Field(description="Whether the column was transformed.")
    transform_functions: list[str] = Field(description="Transforms applied to the column, sorted.")


class ColumnResult(ReportDataModel):
    """Statistics and distribution stability of one training column."""

    name: str = Field(description="Column name.")
    training: ColumnProfile = Field(description="Training column statistics.")
    synthetic: ColumnProfile = Field(description="Synthetic column statistics.")
    distribution_distance: float | None = Field(description="Jensen-Shannon distance between the distributions.")
    distribution_stability: MetricScore | None = Field(description="Distribution stability of the column.")
    distribution: Distribution | None = Field(
        description="Training and synthetic distributions, or null for columns that are not charted, such as text."
    )
    pii: ColumnPIIStatistics | None = Field(description="PII statistics, or null without PII detection.")


class PatternSymbol(ReportDataModel):
    """One symbol of a replacement pattern and what it stands for."""

    symbol: str = Field(description="The symbol as written in the pattern.")
    meaning: str = Field(description="What the symbol is replaced with.")


class PIIDependency(ReportDataModel):
    """A column whose values a replaced column depends on."""

    column_name: str = Field(description="Column name.")
    entity_type: str | None = Field(description="Entity type of the column, or null.")


class PIIReplacementColumn(ReportDataModel):
    """How one training column was replaced."""

    column_name: str = Field(description="Column name.")
    entity_type: str = Field(description="Entity type the column was replaced as.")
    replaced_with: str = Field(description="How replacement values were generated.")
    pattern: str | None = Field(description="Replacement pattern, or null.")
    pattern_symbols: list[PatternSymbol] = Field(description="The pattern's symbols in order of first use.")
    depends_on: list[PIIDependency] = Field(description="Columns the replacement values depend on.")
    found_by: list[str] = Field(description="Free-text detectors that found the replaced values.")
    entity_counts: dict[str, int] = Field(description="Times each entity type was found in the column.")


class PIIReplacement(ReportDataModel):
    """The PII replacement applied to the training data."""

    replaced: bool = Field(description="False when PII replacement was off and PII was detected only to check replay.")
    columns: list[PIIReplacementColumn] = Field(description="Replaced columns, in training order.")
    kept_columns: list[str] = Field(description="Training columns left as they were.")
    sampler: str | None = Field(description="Sampler that generated replacement values, or null.")
    locale: str | None = Field(description="Sampler locale, or null.")
    seed: int | None = Field(description="Sampler seed, or null for a random seed.")
    value_mappings: dict[str, dict[str, list[str]]] = Field(
        description=(
            "`replace_pii.data_to_sampler_value_mapping`: values in each dependency column mapped to sampler values."
        )
    )


class DifferentialPrivacySettings(ReportDataModel):
    """Differential privacy settings of the training run."""

    enabled: bool = Field(description="Whether training used differential privacy.")
    epsilon: float | None = Field(description="Privacy budget, or null when disabled.")
    delta: float | str | None = Field(description="Privacy delta, 'auto', or null when disabled.")


class EvaluationReportData(ReportDataModel):
    """Evaluation results, including everything the HTML report shows. Metric keys match the scores in ``SafeSynthesizerSummary``."""

    schema_version: int = Field(description="Schema version; readers accept higher versions.")
    library_version: str = Field(description="nemo-safe-synthesizer version that built the data.")
    job_id: str | None = Field(description="Job ID from the NEMO_JOB_ID environment variable, or null.")
    scores: Scores = Field(description="Overall scores.")
    metrics: Metrics = Field(description="Per-metric results.")
    dataset: DatasetCounts | None = Field(description="Dataset counts, or null when they were not computed.")
    columns: list[ColumnResult] = Field(
        description="Evaluated training columns, in training order: those also in the synthetic data, sampled down to `evaluation.sqs_report_columns`."
    )
    differential_privacy: DifferentialPrivacySettings = Field(description="Differential privacy settings.")
    pii_replacement: PIIReplacement | None = Field(
        description="PII replacement applied to the training data, or null when none was applied or checked."
    )
