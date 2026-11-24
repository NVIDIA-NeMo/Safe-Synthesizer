# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation results as structured data: scores, per-metric values and charts, and column statistics.

``EvaluationReportData`` carries the values behind the evaluation report so another interface can
present them. Training values are left out, such as detected PII values and per-column distinct
values. Every field is always present in the JSON; a field typed ``X | None`` is ``null`` when there
is no value for it.

Readers should accept any ``schema_version`` at or above the one they were written for and ignore
fields they do not know. The version is raised only when a field is removed or its meaning changes.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

REPORT_DATA_SCHEMA_VERSION = 1
"""Current ``EvaluationReportData.schema_version``."""


class ReportDataModel(BaseModel):
    """Base of the report data models: ignores unknown fields so older readers accept newer data."""

    model_config = ConfigDict(extra="ignore")


class MetricScore(ReportDataModel):
    """A metric's score and grade."""

    raw_score: float | None = Field(description="Unscaled metric value, or null if it could not be computed.")
    score: float | None = Field(description="Score from 0 to 10, or null if it could not be computed.")
    grade: str = Field(description="Qualitative grade, such as 'Unavailable', 'Poor', 'Good', or 'Excellent'.")
    notes: str | None = Field(description="Warning or error about the score, or null.")


class MetricFigure(ReportDataModel):
    """One chart of a metric."""

    title: str = Field(description="Chart title: the column name for per-column charts, else the metric name.")
    figure: dict[str, Any] = Field(
        description=(
            "Plotly figure JSON with `data` and `layout`, without `layout.template`. Arrays may be base64 typed "
            "arrays (`{dtype, bdata}`), which need plotly.js 2.28 or later."
        )
    )


class Metric(ReportDataModel):
    """A metric's score and charts."""

    score: MetricScore = Field(description="The metric's score.")
    figures: list[MetricFigure] = Field(description="Charts in order; empty when the metric has none.")


class MembershipInferenceMetric(Metric):
    """Membership inference protection, with the columns left out of the attack."""

    excluded_columns: list[str] = Field(description="Columns left out via `evaluation.mia_excluded_columns`.")


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

    training_total_records: int = Field(description="Training rows checked.")
    synthetic_total_records: int = Field(description="Synthetic rows checked.")
    entries: list[PIIReplayEntry] = Field(description="Replay counts per column and PII type.")
    skipped_reason: str | None = Field(description="Why PII replay did not run in full, or null.")


class Metrics(ReportDataModel):
    """Per-metric results. A metric is null when it is not part of the evaluation; a metric that was turned off or
    could not run is present with an 'Unavailable' grade.
    """

    column_correlation_stability: Metric | None = Field(description="Column Correlation Stability.")
    deep_structure_stability: Metric | None = Field(description="Deep Structure Stability.")
    column_distribution_stability: Metric | None = Field(
        description="Column Distribution Stability, with a chart per tabular column."
    )
    text_semantic_similarity: Metric | None = Field(description="Text Semantic Similarity.")
    text_structure_similarity: Metric | None = Field(description="Text Structure Similarity.")
    membership_inference_protection: MembershipInferenceMetric | None = Field(
        description="Membership Inference Protection."
    )
    attribute_inference_protection: Metric | None = Field(description="Attribute Inference Protection.")
    pii_replay: PIIReplayMetric | None = Field(description="PII Replay.")


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
    avg_str_length: float | None = Field(description="Mean length of the values as strings.")


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
    pii: ColumnPIIStatistics | None = Field(description="PII statistics, or null without PII detection.")


class DifferentialPrivacySettings(ReportDataModel):
    """Differential privacy settings of the training run."""

    enabled: bool = Field(description="Whether training used differential privacy.")
    epsilon: float | None = Field(description="Privacy budget, or null when disabled.")
    delta: float | str | None = Field(description="Privacy delta, 'auto', or null when disabled.")


class EvaluationReportData(ReportDataModel):
    """Evaluation results for presenting outside the HTML report. Metric keys match ``SafeSynthesizerSummary``."""

    schema_version: int = Field(description="Schema version; readers accept higher versions.")
    library_version: str = Field(description="nemo-safe-synthesizer version that built the data.")
    job_id: str | None = Field(description="Job ID from the NEMO_JOB_ID environment variable, or null.")
    scores: Scores = Field(description="Overall scores.")
    metrics: Metrics = Field(description="Per-metric results.")
    dataset: DatasetCounts | None = Field(description="Dataset counts, or null when they were not computed.")
    columns: list[ColumnResult] = Field(description="Every training column, in training order.")
    differential_privacy: DifferentialPrivacySettings = Field(description="Differential privacy settings.")
