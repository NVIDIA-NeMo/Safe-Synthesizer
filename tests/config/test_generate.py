# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.generate import (
    GenerateParameters,
    StructuredGenerationParameters,
    resolve_structured_generation_schema_method,
)
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.errors import ParameterError


@pytest.mark.unit
class TestResolveStructuredGenerationSchemaMethod:
    @pytest.mark.parametrize(
        ("backend", "expected"),
        [
            ("auto", "structural_tag"),
            ("xgrammar", "structural_tag"),
            ("outlines", "regex"),
            ("guidance", "regex"),
            ("lm-format-enforcer", "regex"),
        ],
    )
    def test_auto_resolves_from_backend(self, backend: str, expected: str) -> None:
        assert resolve_structured_generation_schema_method("auto", backend) == expected

    @pytest.mark.parametrize("method", ["regex", "json_schema", "structural_tag"])
    def test_explicit_methods_pass_through(self, method: str) -> None:
        schema_method = cast(Literal["regex", "json_schema", "structural_tag"], method)
        assert resolve_structured_generation_schema_method(schema_method, "outlines") == method


@pytest.mark.unit
class TestStructuredGenerationParametersStructuralTagValidation:
    @staticmethod
    def _structured_generation_kwargs(
        *, schema_method: str = "structural_tag", backend: str = "xgrammar"
    ) -> dict[str, Any]:
        return {
            "enabled": True,
            "schema_method": schema_method,
            "backend": backend,
        }

    @pytest.mark.parametrize("backend", ["xgrammar", "auto"])
    def test_compatible_backends_validate(self, backend: str) -> None:
        StructuredGenerationParameters(
            **self._structured_generation_kwargs(backend=backend),
        )

    def test_incompatible_backend_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="requires `backend`"):
            StructuredGenerationParameters(**self._structured_generation_kwargs(backend="outlines"))

    @pytest.mark.parametrize("backend", ["outlines", "guidance", "lm-format-enforcer"])
    def test_auto_with_incompatible_backend_validates(self, backend: str) -> None:
        StructuredGenerationParameters(
            **self._structured_generation_kwargs(schema_method="auto", backend=backend),
        )

    def test_default_schema_method_is_auto(self) -> None:
        params = GenerateParameters()
        assert params.structured_generation.schema_method == "auto"

    def test_skipped_when_structured_generation_disabled(self) -> None:
        StructuredGenerationParameters(
            enabled=False,
            schema_method="structural_tag",
            backend="outlines",
        )

    def test_generate_parameters_accepts_nested_structured_generation(self) -> None:
        params = GenerateParameters(
            structured_generation={  # ty: ignore[invalid-argument-type]
                "enabled": True,
                "schema_method": "json_schema",
                "backend": "outlines",
            }
        )
        assert params.structured_generation.enabled is True
        assert params.structured_generation.schema_method == "json_schema"
        assert params.structured_generation.backend == "outlines"

    def test_from_params_accepts_nested_structured_generation(self) -> None:
        params = SafeSynthesizerParameters.from_params(
            generation={
                "structured_generation": {
                    "enabled": True,
                    "schema_method": "json_schema",
                    "backend": "outlines",
                }
            }
        )
        assert params.generation.structured_generation.enabled is True
        assert params.generation.structured_generation.schema_method == "json_schema"
        assert params.generation.structured_generation.backend == "outlines"


_REMOVED_FLAT_FIELDS = (
    "use_structured_generation",
    "structured_generation_backend",
    "structured_generation_schema_method",
    "structured_generation_use_single_sequence",
)
_REMOVED_FLAT_MESSAGE = r"flat structured-generation fields were removed.*structured_generation\.enabled"


@pytest.mark.unit
class TestRemovedFlatStructuredGenerationFields:
    """The pre-0.2.0 flat structured-generation names fail with a pointer to the nested object."""

    @pytest.mark.parametrize("name", _REMOVED_FLAT_FIELDS)
    def test_model_validate_rejects_flat_field(self, name: str) -> None:
        with pytest.raises((ParameterError, ValidationError), match=_REMOVED_FLAT_MESSAGE):
            GenerateParameters.model_validate({name: True})

    @pytest.mark.parametrize("name", _REMOVED_FLAT_FIELDS)
    def test_nested_flat_field_is_rejected_even_when_unknown_fields_are_ignored(self, name: str) -> None:
        with pytest.raises((ParameterError, ValidationError), match=_REMOVED_FLAT_MESSAGE):
            SafeSynthesizerParameters.model_validate({"unknown_fields": "ignore", "generation": {name: True}})

    @pytest.mark.parametrize("name", [*_REMOVED_FLAT_FIELDS, *(f"generation.{n}" for n in _REMOVED_FLAT_FIELDS)])
    def test_from_params_rejects_flat_name(self, name: str) -> None:
        with pytest.raises(ParameterError, match=_REMOVED_FLAT_MESSAGE):
            SafeSynthesizerParameters.from_params(**{name: True})

    def test_flat_attribute_is_gone(self) -> None:
        assert not hasattr(GenerateParameters(), "use_structured_generation")


@pytest.mark.unit
class TestMaxTokensMultiplier:
    def test_default_matches_metadata_constant(self) -> None:
        """The config default mirrors the metadata safety-margin constant."""
        from nemo_safe_synthesizer.llm.metadata import GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER

        assert GenerateParameters().max_tokens_multiplier == GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER

    def test_accepts_widened_value(self) -> None:
        """Users can widen the budget for long free-text datasets."""
        assert GenerateParameters(max_tokens_multiplier=1.8).max_tokens_multiplier == 1.8

    @pytest.mark.parametrize("value", [0, -0.5, float("inf"), float("-inf"), float("nan")])
    def test_rejects_non_positive(self, value: float) -> None:
        """Non-positive and non-finite multipliers are rejected by the validator."""
        with pytest.raises(ValidationError):
            GenerateParameters(max_tokens_multiplier=value)
