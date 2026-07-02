# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result compilation for Safe Synthesizer pipeline runs.

Assembles generation output, evaluation scores, and timing into the
``SafeSynthesizerResults`` and ``SafeSynthesizerSummary`` containers
consumed by the SDK and CLI.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .config import SafeSynthesizerResults, SafeSynthesizerSummary, SafeSynthesizerTiming
from .evaluation.render import render_report
from .evaluation.reports.multimodal.multimodal_report import MultimodalReport
from .generation.results import GenerateJobResults

# Fields passed through unchanged from ``GenerateJobResults`` to
# ``SafeSynthesizerSummary``.  Both models use identical names for these,
# so a single tuple drives both the extraction and the forwarding step.
_GENERATE_RESULT_FIELDS: tuple[str, ...] = (
    "num_valid_records",
    "num_invalid_records",
    "num_prompts",
    "valid_record_fraction",
    "num_completion_tokens",
    "num_valid_record_tokens",
    "num_invalid_record_tokens",
    "num_non_record_tokens",
    "tokens_per_prompt",
    "tokens_per_second",
    "valid_tokens_per_second",
    "tokenization_overhead_sec",
)

# Score fields extracted from a ``MultimodalReport`` when available.  The
# dict maps the ``SafeSynthesizerSummary`` field name to the report score
# name that ``MultimodalReport.get_score_by_name`` understands.
_REPORT_SCORE_FIELDS: dict[str, str] = {
    "synthetic_data_quality_score": "Synthetic Quality Score",
    "column_correlation_stability_score": "Column Correlation Stability",
    "deep_structure_stability_score": "Deep Structure Stability",
    "column_distribution_stability_score": "Column Distribution Stability",
    "text_semantic_similarity_score": "Text Semantic Similarity",
    "text_structure_similarity_score": "Text Structure Similarity",
    "data_privacy_score": "Data Privacy Score",
    "membership_inference_protection_score": "Membership Inference Protection",
    "attribute_inference_protection_score": "Attribute Inference Protection",
}


def make_nss_summary(
    timing: SafeSynthesizerTiming,
    results: GenerateJobResults | pd.DataFrame | None = None,
    report: MultimodalReport | None = None,
) -> SafeSynthesizerSummary:
    """Build a pipeline summary from timing, generation results, and evaluation.

    Extracts evaluation scores from ``report`` when available. If ``report``
    is ``None`` (e.g. PII-only mode), all scores default to ``None``.

    Args:
        timing: Wall-clock timing breakdown for the pipeline.
        results: Generation output -- a ``GenerateJobResults`` with record
            counts, or a raw ``DataFrame``, or ``None``.
        report: Evaluation report containing component scores.

    Returns:
        A populated ``SafeSynthesizerSummary``.
    """
    gen_fields: dict[str, Any] = {}
    if isinstance(results, GenerateJobResults):
        gen_fields = {field: getattr(results, field) for field in _GENERATE_RESULT_FIELDS}
        completion_tokens = gen_fields["num_completion_tokens"]
        valid_record_tokens = gen_fields["num_valid_record_tokens"]
        if completion_tokens and valid_record_tokens is not None:
            gen_fields["valid_record_token_fraction"] = valid_record_tokens / completion_tokens

    # None when ``report`` is missing (e.g. PII-only mode without evaluation).
    report_scores: dict[str, Any] = {
        field: report.get_score_by_name(name) if report is not None else None
        for field, name in _REPORT_SCORE_FIELDS.items()
    }

    return SafeSynthesizerSummary(timing=timing, **gen_fields, **report_scores)


def make_nss_results(
    generate_results: GenerateJobResults | pd.DataFrame,
    total_time: float | None = None,
    training_time: float | None = None,
    generation_time: float | None = None,
    evaluation_time: float | None = None,
    report: MultimodalReport | None = None,
) -> SafeSynthesizerResults:
    """Build the final pipeline results container.

    Combines generation output, timing, and an optional evaluation report
    into a single ``SafeSynthesizerResults`` object.

    Args:
        generate_results: Generation output -- a ``GenerateJobResults`` or
            a raw ``DataFrame`` of synthetic records.
        total_time: Total wall-clock time in seconds.
        training_time: Training phase time in seconds.
        generation_time: Generation phase time in seconds.
        evaluation_time: Evaluation phase time in seconds.
        report: Evaluation report to render as HTML.

    Returns:
        A ``SafeSynthesizerResults`` with synthetic data, summary, and
        optional HTML evaluation report.

    Raises:
        ValueError: If ``generate_results`` is ``None`` or an empty
            ``DataFrame``.
    """
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
