# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from nemo_safe_synthesizer.config import SafeSynthesizerResults, SafeSynthesizerSummary, SafeSynthesizerTiming
from nemo_safe_synthesizer.evaluation.render import render_report
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport
from nemo_safe_synthesizer.generation.results import GenerateJobResults


def make_nss_summary(
    timing: SafeSynthesizerTiming,
    results: GenerateJobResults | pd.DataFrame | None = None,
    report: MultimodalReport | None = None,
) -> SafeSynthesizerSummary:
    # Extract scores from report if available, otherwise use None for all scores
    # (e.g., when running PII-only mode without evaluation)
    if report is not None:
        synthetic_data_quality_score = report.get_score_by_name("Synthetic Quality Score")
        column_correlation_stability_score = report.get_score_by_name("Column Correlation Stability")
        deep_structure_stability_score = report.get_score_by_name("Deep Structure Stability")
        column_distribution_stability_score = report.get_score_by_name("Column Distribution Stability")
        text_semantic_similarity_score = report.get_score_by_name("Text Semantic Similarity")
        text_structure_similarity_score = report.get_score_by_name("Text Structure Similarity")
        data_privacy_score = report.get_score_by_name("Data Privacy Score")
        membership_inference_protection_score = report.get_score_by_name("Membership Inference Protection")
        attribute_inference_protection_score = report.get_score_by_name("Attribute Inference Protection")
    else:
        synthetic_data_quality_score = None
        column_correlation_stability_score = None
        deep_structure_stability_score = None
        column_distribution_stability_score = None
        text_semantic_similarity_score = None
        text_structure_similarity_score = None
        data_privacy_score = None
        membership_inference_protection_score = None
        attribute_inference_protection_score = None

    num_valid_records = None
    num_invalid_records = None
    num_prompts = None
    valid_record_fraction = None
    if isinstance(results, GenerateJobResults):
        num_valid_records = results.num_valid_records
        num_invalid_records = results.num_invalid_records
        num_prompts = results.num_prompts
        valid_record_fraction = results.valid_record_fraction

    return SafeSynthesizerSummary(
        timing=timing,
        num_valid_records=num_valid_records,
        num_invalid_records=num_invalid_records,
        num_prompts=num_prompts,
        valid_record_fraction=valid_record_fraction,
        synthetic_data_quality_score=synthetic_data_quality_score,
        column_correlation_stability_score=column_correlation_stability_score,
        deep_structure_stability_score=deep_structure_stability_score,
        column_distribution_stability_score=column_distribution_stability_score,
        text_semantic_similarity_score=text_semantic_similarity_score,
        text_structure_similarity_score=text_structure_similarity_score,
        data_privacy_score=data_privacy_score,
        membership_inference_protection_score=membership_inference_protection_score,
        attribute_inference_protection_score=attribute_inference_protection_score,
    )


def make_nss_results(
    generate_results: GenerateJobResults | pd.DataFrame,
    total_time: float | None = None,
    training_time: float | None = None,
    generation_time: float | None = None,
    evaluation_time: float | None = None,
    report: MultimodalReport | None = None,
) -> SafeSynthesizerResults:
    timing = SafeSynthesizerTiming(
        total_time_sec=total_time,
        evaluation_time_sec=evaluation_time,
        training_time_sec=training_time,
        generation_time_sec=generation_time,
    )
    summary = make_nss_summary(timing, generate_results, report)
    if generate_results is None:
        raise ValueError("generate_results is required")
    if isinstance(generate_results, pd.DataFrame) and generate_results.empty:
        raise ValueError("generate_results are empty")
    return SafeSynthesizerResults(
        synthetic_data=generate_results.df if isinstance(generate_results, GenerateJobResults) else generate_results,
        summary=summary,
        evaluation_report_html=render_report(report) if report else None,
    )
