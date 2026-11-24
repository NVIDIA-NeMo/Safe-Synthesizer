# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured data behind the evaluation report, for rendering it without the HTML template.

``EvaluationReportData`` holds what the HTML report displays and nothing more: scores, the
values its cards and tables show, and each chart as Plotly figure JSON. Training values the
report does not show, such as detected PII values, per-column distinct values, and group IDs,
are left out. Every field is always present in the JSON; a field typed ``X | None`` is
``null`` when the report has no value for it, and a card is ``null`` when its component did
not run.

Readers should accept any ``schema_version`` at or above the one they were written for and
ignore fields they do not know. The version is raised only when a field is removed or its
meaning changes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from ...config.base import NSSBaseModel

REPORT_DATA_SCHEMA_VERSION = 1
"""Current ``EvaluationReportData.schema_version``."""


class ReportScore(NSSBaseModel):
    """A metric's score and grade."""

    raw_score: float | None = Field(description="Unscaled metric value, or null if it could not be computed.")
    score: float | None = Field(description="Score from 0 to 10, or null if it could not be computed.")
    grade: str = Field(description="Qualitative grade, such as 'Unavailable', 'Poor', 'Good', or 'Excellent'.")
    notes: str | None = Field(description="Warning or error shown with the score, or null.")


class ReportFigure(NSSBaseModel):
    """One chart of a report card."""

    title: str = Field(description="Chart title: the column name for per-column charts, else the metric name.")
    figure: dict[str, Any] = Field(
        description=(
            "Plotly figure JSON with `data` and `layout`, without `layout.template`. Arrays may be base64 typed "
            "arrays (`{dtype, bdata}`), which need plotly.js 2.28 or later."
        )
    )


class ReportCard(NSSBaseModel):
    """A metric with its score."""

    name: str = Field(description="Metric display name.")
    score: ReportScore = Field(description="The metric's score.")


class FigureCard(ReportCard):
    """A metric with its score and charts."""

    figures: list[ReportFigure] = Field(description="Charts in display order; empty when the metric has none.")


class DatasetStatisticsCard(ReportCard):
    """Row, column, and missing-value counts for the training and synthetic data."""

    training_rows: int = Field(description="Training rows evaluated.")
    training_cols: int = Field(description="Training columns evaluated.")
    training_missing: int = Field(description="Percentage of missing training values.")
    synthetic_rows: int = Field(description="Synthetic rows evaluated.")
    synthetic_cols: int = Field(description="Synthetic columns evaluated.")
    synthetic_missing: int = Field(description="Percentage of missing synthetic values.")
    memorized_lines: int = Field(description="Training rows duplicated in the synthetic data.")


class MembershipInferenceCard(FigureCard):
    """Membership inference protection, with the columns left out of the attack."""

    excluded_columns: list[str] = Field(description="Columns left out via `evaluation.mia_excluded_columns`.")


class ColumnProfile(NSSBaseModel):
    """Summary statistics of one column in one dataset."""

    type: str = Field(description="Inferred column type, such as 'numeric', 'categorical', or 'text'.")
    count: int = Field(description="Non-null values.")
    unique_count: int = Field(description="Distinct non-null values.")
    unique_percent: float | None = Field(description="Distinct values as a percentage of non-null values.")
    missing_count: int = Field(description="Null values.")
    missing_percent: float | None = Field(description="Null values as a percentage of all values.")
    avg_str_length: float | None = Field(description="Mean length of the values as strings.")


class ColumnPIIStatistics(NSSBaseModel):
    """PII detected in a training column and how it was replaced."""

    assigned_type: str | None = Field(description="Column type assigned by PII detection.")
    assigned_entity: str | None = Field(description="Entity assigned to the column.")
    detected_entity_counts: dict[str, int] = Field(description="Times each entity type was detected.")
    is_transformed: bool = Field(description="Whether the column was transformed.")
    transform_functions: list[str] = Field(description="Transforms applied to the column, sorted.")


class ColumnReport(NSSBaseModel):
    """One row of the Training Data table."""

    name: str = Field(description="Column name.")
    training: ColumnProfile = Field(description="Training column statistics.")
    synthetic: ColumnProfile = Field(description="Synthetic column statistics.")
    distribution_distance: float | None = Field(description="Jensen-Shannon distance between the distributions.")
    distribution_stability: ReportScore | None = Field(description="Distribution stability of the column.")
    pii: ColumnPIIStatistics | None = Field(description="PII statistics, or null without PII detection.")


class ColumnDistributionCard(FigureCard):
    """Column distribution stability, with a chart per tabular column and the Training Data table."""

    columns: list[ColumnReport] = Field(description="Every training column, in training order.")


class PIIReplayEntry(NSSBaseModel):
    """Replay counts for one PII type in one column."""

    column_name: str = Field(description="Column name.")
    column_assigned_type: str = Field(description="Column type assigned by PII detection.")
    pii_type: str = Field(description="PII type, or 'none' for a whole column checked by name.")
    total_training_data: int = Field(description="Training rows containing a value of this type.")
    unique_training_data: int = Field(description="Distinct training values of this type.")
    total_synthetic_data: int = Field(description="Synthetic rows containing one of the training values.")
    unique_synthetic_data: int = Field(description="Distinct training values found in the synthetic data.")
    unique_synthetic_data_percentage: float = Field(description="unique_synthetic_data / unique_training_data * 100.")


class PIIReplayCard(ReportCard):
    """How many training PII values reappear in the synthetic data."""

    training_total_records: int = Field(description="Training rows checked.")
    synthetic_total_records: int = Field(description="Synthetic rows checked.")
    entries: list[PIIReplayEntry] = Field(description="Replay counts per column and PII type.")
    skipped_reason: str | None = Field(description="Why PII replay did not run in full, or null.")


class PIIEntity(NSSBaseModel):
    """One PII type of a column in the PII Replacement and Replay table. Replay counts are null without replay."""

    pii_type: str = Field(description="PII type.")
    found: int | None = Field(description="Values of this type found in the training data.")
    unique: int | None = Field(description="Distinct training values of this type.")
    synthetic: int | None = Field(description="Synthetic rows containing one of the training values.")
    replayed: int | None = Field(description="Distinct training values found in the synthetic data.")
    replay_percentage: float | None = Field(description="replayed / unique * 100.")


class PIIPatternSymbol(NSSBaseModel):
    """A symbol of a replacement pattern and what it stands for."""

    symbol: str = Field(description="Pattern symbol.")
    meaning: str = Field(description="What the symbol produces.")


class PIIDependency(NSSBaseModel):
    """A column a replacement depends on."""

    column_name: str = Field(description="Column the replacement reads.")
    entity_type: str | None = Field(description="That column's entity type, or null.")


class PIIColumn(NSSBaseModel):
    """One column of the PII Replacement and Replay table."""

    column_name: str = Field(description="Column name.")
    replaced_with: str | None = Field(description="Description of the replacement, or null if not replaced.")
    pattern: str | None = Field(description="Replacement pattern, or null.")
    pattern_symbols: list[PIIPatternSymbol] = Field(description="Symbols of `pattern` with their meanings.")
    depends_on: list[PIIDependency] = Field(description="Columns the replacement depends on.")
    found_by: list[str] = Field(description="Detectors that found PII in a free-text column.")
    entities: list[PIIEntity] = Field(description="PII types in the column, most frequent first.")


class PIIReplacementReport(NSSBaseModel):
    """The PII Replacement and Replay card."""

    replacement_off: bool = Field(description="True when PII replacement was off and PII was only detected.")
    headline: str = Field(description="Short summary, such as '3 columns replaced'.")
    sampler: str | None = Field(description="Replacement value sampler, or null when replacement was off.")
    locale: str | None = Field(description="Sampler locale, or null when replacement was off.")
    seed: int | None = Field(description="Sampler seed, or null for a random seed or when replacement was off.")
    replay_note: str | None = Field(description="Why replay results are missing or partial, or null.")
    columns: list[PIIColumn] = Field(description="Replaced columns in training order, then other columns with PII.")
    kept_columns: list[str] = Field(description="Training columns kept as is.")
    value_mappings: dict[str, dict[str, list[str]]] = Field(
        description="Per column, the configured dataset values and the sampler values they map to."
    )


class SkippedComparison(NSSBaseModel):
    """A time-series group and column pair left out of a metric."""

    column: str = Field(description="Column name.")
    group: str | None = Field(description="Group label, or null for one global sequence.")
    reason: str = Field(description="Why the pair was skipped.")


class TimeSeriesFigure(ReportFigure):
    """A time-series chart, with the selector and tooltip shown beside it."""

    column_select: bool = Field(
        description="Whether the chart has a column selector; its traces repeat per `summary_columns` entry."
    )
    tooltip: str = Field(description="Explanation of the chart.")


class TimeSeriesCard(ReportCard):
    """A time-series metric card."""

    section: Literal["time_series", "data_privacy"] = Field(description="Report section the card is shown in.")
    description: str = Field(description="One-line description of the metric.")
    score_colored_traces: bool = Field(description="Whether chart traces are colored by score, not by dataset.")
    figures: list[TimeSeriesFigure] = Field(description="Charts in display order.")
    summary_columns: list[str] = Field(description="Numeric columns, in the order of the column selector.")
    summary_traces_per_column: int = Field(description="Traces per column in a chart with a column selector.")
    skipped: list[SkippedComparison] = Field(description="The first 12 skipped comparisons, as the report lists.")
    groups_only_in_training: int = Field(description="Groups found only in the training data.")
    groups_only_in_synthetic: int = Field(description="Groups found only in the synthetic data.")


class AutocorrelationCard(TimeSeriesCard):
    """Autocorrelation similarity."""

    evaluated_profile_count: int = Field(description="Group and column pairs summarized.")
    evaluated_group_count: int = Field(description="Groups summarized.")
    evaluated_column_count: int = Field(description="Numeric columns summarized.")
    pair_score_column_count: int = Field(description="Lowest-scoring columns shown in the pair scores chart.")
    constant_synthetic_profile_count: int = Field(description="Pairs whose synthetic values are constant.")


class WindowCounts(NSSBaseModel):
    """Windows compared per split, after and before sampling."""

    training_windows: int = Field(description="Training windows compared.")
    training_windows_total: int = Field(description="Training windows before sampling.")
    holdout_windows: int = Field(description="Holdout windows compared.")
    holdout_windows_total: int = Field(description="Holdout windows before sampling.")
    holdout_groups_with_windows: int = Field(description="Holdout groups with at least one window.")
    synthetic_windows: int = Field(description="Synthetic windows compared.")
    synthetic_windows_total: int = Field(description="Synthetic windows before sampling.")


class WindowAttack(NSSBaseModel):
    """Share of windows flagged as close to a synthetic window."""

    training_flag_rate: float = Field(description="Share of training windows flagged.")
    holdout_flag_rate: float = Field(description="Share of holdout windows flagged.")


class WindowMembershipInferenceCard(TimeSeriesCard):
    """Window membership inference protection."""

    evaluation_mode: str | None = Field(
        description="'full', 'copy_diagnostics_only', 'unavailable', or null if the metric did not get that far."
    )
    window_size: int | None = Field(description="Steps per window.")
    training_group_count: int = Field(description="Training groups with windows.")
    counts: WindowCounts | None = Field(description="Window counts, or null if windows were not built.")
    exact_copy_rate: float | None = Field(description="Share of training windows copied exactly.")
    attack: WindowAttack | None = Field(description="Flag rates, or null without a usable holdout.")
    closest_match_exact_copy: bool | None = Field(description="Whether the closest match is an exact copy.")
    closest_match_distance_per_value: float | None = Field(
        description="Closest match distance in training standard deviations per value."
    )


class ReportSections(NSSBaseModel):
    """Which report sections have data."""

    synthetic_quality: bool = Field(description="The Synthetic Quality section has a score.")
    data_privacy: bool = Field(description="The Data Privacy section has a score.")
    transformed_columns: bool = Field(description="At least one column was transformed by PII replacement.")
    time_series: bool = Field(description="Time-series metrics ran.")


class DifferentialPrivacyReport(NSSBaseModel):
    """Differential privacy settings shown with the Data Privacy Score."""

    enabled: bool = Field(description="Whether training used differential privacy.")
    epsilon: float | None = Field(description="Privacy budget, or null when disabled.")
    delta: float | str | None = Field(description="Privacy delta, 'auto', or null when disabled.")


class EvaluationReportData(NSSBaseModel):
    """Everything the evaluation report displays, for rendering it natively.

    Cards are named like ``SafeSynthesizerSummary`` scores and are null when their component did not run.
    """

    schema_version: int = Field(description="Schema version; readers accept higher versions.")
    library_version: str = Field(description="nemo-safe-synthesizer version that built the data.")
    job_id: str | None = Field(description="Job ID from the NEMO_JOB_ID environment variable, or null.")
    sections: ReportSections = Field(description="Which report sections have data.")
    differential_privacy: DifferentialPrivacyReport = Field(description="Differential privacy settings.")
    tooltips: dict[str, str] = Field(description="Explanations shown in the report's info tooltips, by key.")

    synthetic_quality_score: ReportCard | None = Field(description="Synthetic Quality Score.")
    data_privacy_score: ReportCard | None = Field(description="Data Privacy Score.")
    dataset_statistics: DatasetStatisticsCard | None = Field(description="Dataset Statistics.")
    column_correlation_stability: FigureCard | None = Field(description="Column Correlation Stability.")
    deep_structure_stability: FigureCard | None = Field(description="Deep Structure Stability.")
    column_distribution_stability: ColumnDistributionCard | None = Field(
        description="Column Distribution Stability and the Training Data table."
    )
    text_semantic_similarity: FigureCard | None = Field(description="Text Semantic Similarity.")
    text_structure_similarity: FigureCard | None = Field(description="Text Structure Similarity.")
    membership_inference_protection: MembershipInferenceCard | None = Field(
        description="Membership Inference Protection."
    )
    attribute_inference_protection: FigureCard | None = Field(description="Attribute Inference Protection.")
    autocorrelation_similarity: AutocorrelationCard | None = Field(description="Autocorrelation Similarity.")
    window_membership_inference_protection: WindowMembershipInferenceCard | None = Field(
        description="Window Membership Inference Protection."
    )
    pii_replay: PIIReplayCard | None = Field(description="PII Replay counts.")
    pii_replacement: PIIReplacementReport | None = Field(
        description="PII Replacement and Replay card, or null when neither replacement nor replay ran."
    )
