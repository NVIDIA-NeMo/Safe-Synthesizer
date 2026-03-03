# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    Field,
)

from ..configurator.parameters import (
    Parameters,
)
from ..configurator.validators import (
    ValueValidator,
    range_validator,
)

__all__ = ["GenerateParameters", "ValidationParameters"]


class ValidationParameters(Parameters, BaseModel):
    """Configuration for record and sequence validation.

    These parameters control the validation and automatic fixes when going
    from LLM output to tabular data.
    """

    group_by_accept_no_delineator: Annotated[
        bool,
        Field(
            title="group_by_accept_no_delineator",
            description="Whether to accept completions without both beginning and end of sequence delineators as a single sequence.",
        ),
    ] = False

    group_by_ignore_invalid_records: Annotated[
        bool,
        Field(
            title="group_by_ignore_invalid_records",
            description="Whether to ignore invalid records in a sequence and proceed with the valid records.",
        ),
    ] = False

    group_by_fix_non_unique_value: Annotated[
        bool,
        Field(
            title="group_by_fix_non_unique_value",
            description="Whether to automatically fix non-unique group by values in a sequence by using the first unique value for all records.",
        ),
    ] = False

    group_by_fix_unordered_records: Annotated[
        bool,
        Field(
            title="group_by_fix_unordered_records",
            description="Whether to automatically fix unordered records in a sequence by sorting the records.",
        ),
    ] = False


class GenerateParameters(Parameters, BaseModel):
    """Configuration parameters for synthetic data generation.

    These parameters control how synthetic data is generated after the model is trained.
    They affect the quality, diversity, and validity of the generated synthetic records.

    Attributes:
        num_records: Number of synthetic records to generate. Maximum is 130,000 records.
        temperature: Sampling temperature for controlling randomness (higher = more random).
        repetition_penalty: Penalty for token repetition (≥1.0, higher = less repetition).
        top_p: Nucleus sampling probability for token selection (0 < value ≤ 1).
        patience: Number of invalid records fraction before stopping.
        invalid_fraction_threshold: "The fraction of invalid records that will stop generation after the `patience` limit is reached."
        use_structured_generation: Whether to use structured generation for better format control.
        attention_backend: The attention backend for the vLLM engine. If None, vLLM will
            auto-select the best available backend.

    """

    num_records: Annotated[
        int,
        Field(
            title="num_records",
            description="Number of records to generate.",
        ),
    ] = 1000

    temperature: Annotated[
        float,
        Field(
            title="temperature",
            description="Sampling temperature.",
        ),
    ] = 0.9

    repetition_penalty: Annotated[
        float,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="repetition_penalty",
            description="The value used to control the likelihood of the model repeating the same token. Must be > 0.",
        ),
    ] = 1.0

    top_p: Annotated[
        float,
        ValueValidator(value_func=lambda v: 0 < v <= 1),
        Field(
            title="top_p",
            description="Nucleus sampling probability. Must be in (0, 1].",
        ),
    ] = 1.0

    patience: Annotated[
        int,
        ValueValidator(value_func=lambda v: v >= 1),
        Field(
            title="patience",
            description=(
                "Number of consecutive generations where the `invalid_fraction_threshold` "
                "is reached before stopping generation. Must be >= 1."
            ),
        ),
    ] = 3

    invalid_fraction_threshold: Annotated[
        float,
        ValueValidator(lambda p: range_validator(p, lambda v: 0 <= v <= 1)),
        Field(
            title="invalid_fraction_threshold",
            description=(
                "The fraction of invalid records that will stop generation after the `patience` limit is reached. "
                "Must be in [0, 1]."
            ),
        ),
    ] = 0.8

    use_structured_generation: Annotated[
        bool,
        Field(
            title="use_structured_generation",
            description="Use structured generation.",
        ),
    ] = False

    structured_generation_backend: Annotated[
        Literal["auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"],
        Field(
            title="structured_generation_backend",
            description=(
                "The backend used by VLLM when use_structured_generation=True. "
                "Supported backends (from vllm) are 'outlines', 'guidance', 'xgrammar', 'lm-format-enforcer'. 'auto' will allow vllm to choose the backend."
            ),
        ),
    ] = "auto"

    structured_generation_schema_method: Annotated[
        Literal["regex", "json_schema"],
        Field(
            title="structured_generation_schema_method",
            description=(
                "The method used to generate the schema from your dataset and pass it to the generation backend. "
                "auto will usually default to 'json_schema'. Use 'regex to use our custom regex construction method, which "
                "tends to be more comprehensive  than 'json_schema' at the cost of speed."
            ),
        ),
    ] = "regex"

    structured_generation_use_single_sequence: Annotated[
        bool,
        Field(
            title="structured_generation_use_single_sequence",
            description="Whether to use a regex that matches exactly one sequence or record if max_sequences_per_example is 1.",
        ),
    ] = False

    # TODO: We will merge this with `timestamp_validation_mode` described in the MR !5153
    enforce_timeseries_fidelity: Annotated[
        bool,
        Field(
            title="enforce_timeseries_fidelity",
            description="Enforce timeseries fidelity by enforcing the time series order, intervals, start and end times of the records.",
        ),
    ] = False

    validation: ValidationParameters = Field(
        description="Validation parameters controlling validation logic and automatic fixes when parsing LLM output and converting to tabular data.",
        default_factory=ValidationParameters,
    )

    attention_backend: Annotated[
        str | None,
        Field(
            title="attention_backend",
            description=(
                "The attention backend for the vLLM engine. Common values: 'FLASHINFER', "
                "'FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION'. "
                "If None or 'auto', vLLM will auto-select the best available backend."
            ),
        ),
    ] = "auto"
