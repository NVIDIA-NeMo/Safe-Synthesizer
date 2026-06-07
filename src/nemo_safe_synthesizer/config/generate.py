# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Annotated, Any, ClassVar, Literal, Self

from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

from ..configurator.parameter_paths import ParameterSchema
from ..configurator.parameters import (
    Parameters,
)
from ..configurator.validators import (
    ValueValidator,
    range_validator,
)
from ..errors import ParameterError

StructuredGenerationSchemaMethod = Literal["auto", "regex", "json_schema", "structural_tag"]
ResolvedStructuredGenerationSchemaMethod = Literal["regex", "json_schema", "structural_tag"]
StructuredGenerationBackend = Literal["auto", "xgrammar", "guidance", "outlines", "lm-format-enforcer"]

STRUCTURAL_TAG_COMPATIBLE_BACKENDS = frozenset({"auto", "xgrammar"})

__all__ = [
    "GenerateParameters",
    "RemoteParameters",
    "ResolvedStructuredGenerationSchemaMethod",
    "StructuredGenerationParameters",
    "StructuredGenerationBackend",
    "StructuredGenerationSchemaMethod",
    "STRUCTURAL_TAG_COMPATIBLE_BACKENDS",
    "ValidationParameters",
    "resolve_structured_generation_schema_method",
    "structural_tag_backend_error_message",
]


def resolve_structured_generation_schema_method(
    schema_method: StructuredGenerationSchemaMethod,
    backend: StructuredGenerationBackend | str,
) -> ResolvedStructuredGenerationSchemaMethod:
    """Resolve ``auto`` schema method from the configured structured-output backend.

    ``auto`` picks ``structural_tag`` on xgrammar-capable backends and ``regex``
    elsewhere, preserving legacy behavior for outlines/guidance configs that omit
    an explicit schema method.
    """
    if schema_method != "auto":
        return schema_method
    if backend in STRUCTURAL_TAG_COMPATIBLE_BACKENDS:
        return "structural_tag"
    return "regex"


def structural_tag_backend_error_message(backend: str) -> str | None:
    """Return an error message when *backend* cannot serve ``structural_tag``.

    vLLM only supports XGrammar Structural Tag constraints when the guided
    decoding backend is ``xgrammar`` or ``auto`` (which selects xgrammar for
    this schema method).
    """
    if backend in STRUCTURAL_TAG_COMPATIBLE_BACKENDS:
        return None
    return (
        "Invalid structured generation configuration: "
        "`schema_method='structural_tag'` requires "
        f"`backend` to be 'xgrammar' or 'auto', got {backend!r}."
    )


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
            description="Whether to automatically fix non-unique group-by values in a sequence by using the first unique value for all records.",
        ),
    ] = False

    group_by_fix_unordered_records: Annotated[
        bool,
        Field(
            title="group_by_fix_unordered_records",
            description="Whether to automatically fix unordered records in a sequence by sorting the records.",
        ),
    ] = False


class StructuredGenerationParameters(Parameters, BaseModel):
    """Configuration for vLLM structured generation.

    These parameters control whether generation is constrained to schema-shaped
    output, which backend enforces the constraint, and how the constraint schema
    is built.
    """

    enabled: Annotated[
        bool,
        Field(
            title="enabled",
            description="Whether to use structured generation for better format control.",
        ),
    ] = False

    backend: Annotated[
        StructuredGenerationBackend,
        Field(
            title="backend",
            description=(
                "The backend used by vLLM when structured generation is enabled. "
                "Supported backends: 'outlines', 'guidance', 'xgrammar', 'lm-format-enforcer'. "
                "'auto' will allow vLLM to choose the backend."
            ),
        ),
    ] = "auto"

    schema_method: Annotated[
        StructuredGenerationSchemaMethod,
        Field(
            title="schema_method",
            description=(
                "The method used to generate the schema from your dataset and pass it to the generation backend. "
                "'auto' picks 'structural_tag' on xgrammar-capable backends and 'regex' otherwise. "
                "'regex' uses a custom regex construction method that tends to be more comprehensive "
                "than 'json_schema' at the cost of speed. 'structural_tag' uses XGrammar Structural Tag "
                "to compose schema-constrained JSONL output."
            ),
        ),
    ] = "auto"

    use_single_sequence: Annotated[
        bool,
        Field(
            title="use_single_sequence",
            description="Whether to use a regex that matches exactly one sequence or record if ``max_sequences_per_example`` is 1.",
        ),
    ] = False

    @model_validator(mode="after")
    def _validate_structural_tag_backend(self) -> Self:
        if not self.enabled:
            return self
        if self.schema_method != "structural_tag":
            return self
        if message := structural_tag_backend_error_message(self.backend):
            raise ParameterError(message)
        return self


class RemoteParameters(Parameters, BaseModel):
    """Connection to an external vLLM OpenAI-compatible inference server.

    When set on :class:`GenerateParameters`, generation issues HTTP requests
    to this endpoint instead of loading a local vLLM engine. The server must
    already serve the base model with the fine-tuned LoRA adapter attached,
    registered under ``model``. No GPU is used locally.

    Structured generation maps to vLLM's ``structured_outputs`` request field
    (``regex`` / ``json``); the ``structural_tag`` schema method is not
    supported over the remote API.
    """

    endpoint_url: Annotated[
        str,
        Field(
            title="endpoint_url",
            description="Base URL of the OpenAI-compatible server, e.g. 'http://localhost:8000/v1'.",
        ),
    ]

    model: Annotated[
        str,
        Field(
            title="model",
            description="Model name as registered on the server (the served base model or LoRA adapter name).",
        ),
    ]

    api_key_env: Annotated[
        str | None,
        Field(
            title="api_key_env",
            description=(
                "Name of the environment variable holding the bearer token for the endpoint. "
                "When unset, no Authorization header is sent."
            ),
        ),
    ] = None

    timeout_seconds: Annotated[
        float,
        ValueValidator(value_func=lambda v: v > 0),
        Field(
            title="timeout_seconds",
            description="Per-request timeout in seconds. Must be > 0.",
        ),
    ] = 300.0

    max_concurrency: Annotated[
        int,
        ValueValidator(value_func=lambda v: v >= 1),
        Field(
            title="max_concurrency",
            description="Maximum number of concurrent in-flight requests per batch. Must be >= 1.",
        ),
    ] = 16


class GenerateParameters(Parameters, BaseModel):
    """Configuration parameters for synthetic data generation.

    These parameters control how synthetic data is generated after the model is trained.
    They affect the quality, diversity, and validity of the generated synthetic records.
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
            description="Sampling temperature for controlling randomness (higher = more random).",
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
            description="Nucleus sampling probability for token selection. Must be in (0, 1].",
        ),
    ] = 1.0

    patience: Annotated[
        int,
        ValueValidator(value_func=lambda v: v >= 1),
        Field(
            title="patience",
            description=(
                "Number of consecutive generations where the ``invalid_fraction_threshold`` "
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
                "The fraction of invalid records that will stop generation after the ``patience`` limit is reached. "
                "Must be in [0, 1]."
            ),
        ),
    ] = 0.8

    structured_generation: StructuredGenerationParameters = Field(
        description="Structured generation parameters controlling schema-constrained output.",
        default_factory=StructuredGenerationParameters,
    )

    # TODO: We will merge this with `timestamp_validation_mode` described in the MR !5153
    enforce_timeseries_fidelity: Annotated[
        bool,
        Field(
            title="enforce_timeseries_fidelity",
            description="Enforce time-series fidelity by enforcing order, intervals, start and end times of the records.",
        ),
    ] = False

    validation: ValidationParameters = Field(
        description="Validation parameters controlling validation logic and automatic fixes when parsing LLM output and converting to tabular data.",
        default_factory=ValidationParameters,
    )

    remote: Annotated[
        RemoteParameters | None,
        Field(
            title="remote",
            description=(
                "When set, generate by calling an external vLLM OpenAI-compatible server instead of "
                "loading a local vLLM engine. The server must already serve the trained LoRA adapter. "
                "Not supported for time-series generation."
            ),
        ),
    ] = None

    attention_backend: Annotated[
        str | None,
        Field(
            title="attention_backend",
            description=(
                "The attention backend for the vLLM engine. Common values: 'FLASHINFER', "
                "'FLASH_ATTN', 'TRITON_ATTN', 'FLEX_ATTENTION'. "
                "If ``None`` or 'auto', vLLM will auto-select the best available backend."
            ),
        ),
    ] = "auto"

    parameter_aliases: ClassVar[Mapping[str, str]] = {
        "use_structured_generation": "structured_generation.enabled",
        "structured_generation_backend": "structured_generation.backend",
        "structured_generation_schema_method": "structured_generation.schema_method",
        "structured_generation_use_single_sequence": "structured_generation.use_single_sequence",
    }

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_structured_generation_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data
        return ParameterSchema.from_model(cls).normalize_aliases(data)

    @property
    def use_structured_generation(self) -> bool:
        """Deprecated flat alias for ``structured_generation.enabled``."""
        warnings.warn(
            "use_structured_generation is deprecated; use structured_generation.enabled instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.structured_generation.enabled

    @use_structured_generation.setter
    def use_structured_generation(self, value: bool) -> None:
        warnings.warn(
            "use_structured_generation is deprecated; use structured_generation.enabled instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.structured_generation.enabled = value

    @property
    def structured_generation_backend(self) -> StructuredGenerationBackend:
        """Deprecated flat alias for ``structured_generation.backend``."""
        warnings.warn(
            "structured_generation_backend is deprecated; use structured_generation.backend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.structured_generation.backend

    @structured_generation_backend.setter
    def structured_generation_backend(self, value: StructuredGenerationBackend) -> None:
        warnings.warn(
            "structured_generation_backend is deprecated; use structured_generation.backend instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.structured_generation.backend = value

    @property
    def structured_generation_schema_method(self) -> StructuredGenerationSchemaMethod:
        """Deprecated flat alias for ``structured_generation.schema_method``."""
        warnings.warn(
            "structured_generation_schema_method is deprecated; use structured_generation.schema_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.structured_generation.schema_method

    @structured_generation_schema_method.setter
    def structured_generation_schema_method(self, value: StructuredGenerationSchemaMethod) -> None:
        warnings.warn(
            "structured_generation_schema_method is deprecated; use structured_generation.schema_method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.structured_generation.schema_method = value

    @property
    def structured_generation_use_single_sequence(self) -> bool:
        """Deprecated flat alias for ``structured_generation.use_single_sequence``."""
        warnings.warn(
            "structured_generation_use_single_sequence is deprecated; use structured_generation.use_single_sequence instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.structured_generation.use_single_sequence

    @structured_generation_use_single_sequence.setter
    def structured_generation_use_single_sequence(self, value: bool) -> None:
        warnings.warn(
            "structured_generation_use_single_sequence is deprecated; use structured_generation.use_single_sequence instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.structured_generation.use_single_sequence = value
