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

    def test_generate_parameters_migrates_legacy_flat_keys(self) -> None:
        params = GenerateParameters.model_validate(
            {
                "use_structured_generation": True,
                "structured_generation_schema_method": "json_schema",
                "structured_generation_backend": "outlines",
            }
        )
        assert params.structured_generation.enabled is True
        assert params.structured_generation.schema_method == "json_schema"
        assert params.structured_generation.backend == "outlines"

    def test_from_params_rejects_incompatible_legacy_backend(self) -> None:
        with pytest.raises(ValidationError, match="outlines"):
            SafeSynthesizerParameters.from_params(
                use_structured_generation=True,
                structured_generation_schema_method="structural_tag",
                structured_generation_backend="outlines",
            )

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


@pytest.mark.unit
class TestMixedInputMigration:
    """Regression coverage for mixed-input: legacy flat keys coexist with nested structured_generation dict.

    The migration contract: when both forms are present, legacy flat keys win as explicit overrides.
    Un-overridden nested dict fields are preserved.
    """

    def test_legacy_backend_overrides_nested_dict_backend(self) -> None:
        """A legacy flat backend key beats the nested dict backend when both are supplied."""
        params = GenerateParameters.model_validate(
            {
                "structured_generation": {"enabled": True, "backend": "outlines"},
                "structured_generation_backend": "xgrammar",
            }
        )
        assert params.structured_generation.backend == "xgrammar"
        assert params.structured_generation.enabled is True  # nested value preserved

    def test_legacy_enabled_overrides_nested_dict_enabled(self) -> None:
        """A legacy use_structured_generation key beats the nested dict enabled field."""
        params = GenerateParameters.model_validate(
            {
                "use_structured_generation": True,
                "structured_generation": {"enabled": False, "schema_method": "json_schema"},
            }
        )
        assert params.structured_generation.enabled is True
        assert params.structured_generation.schema_method == "json_schema"  # preserved

    def test_legacy_schema_method_overrides_nested_dict_schema_method(self) -> None:
        """A legacy schema_method key beats the nested dict schema_method field."""
        params = GenerateParameters.model_validate(
            {
                "structured_generation_schema_method": "regex",
                "structured_generation": {"enabled": True, "schema_method": "json_schema"},
            }
        )
        assert params.structured_generation.schema_method == "regex"
        assert params.structured_generation.enabled is True  # preserved

    def test_from_params_legacy_backend_overrides_generation_section_backend(self) -> None:
        """from_params: a top-level legacy backend key overrides the backend inside generation={}."""
        params = SafeSynthesizerParameters.from_params(
            generation={"structured_generation": {"enabled": True, "backend": "outlines"}},
            structured_generation_backend="xgrammar",
        )
        assert params.generation.structured_generation.backend == "xgrammar"
        assert params.generation.structured_generation.enabled is True  # preserved

    def test_from_params_legacy_enabled_overrides_structured_generation_kwarg(self) -> None:
        """from_params: a top-level legacy use_structured_generation overrides structured_generation kwarg."""
        params = SafeSynthesizerParameters.from_params(
            structured_generation={"enabled": False, "schema_method": "json_schema"},
            use_structured_generation=True,
        )
        assert params.generation.structured_generation.enabled is True
        assert params.generation.structured_generation.schema_method == "json_schema"  # preserved

    def test_from_params_legacy_alias_and_dotted_name_are_duplicate_paths(self) -> None:
        with pytest.raises(ParameterError, match=r"generation\.structured_generation\.backend"):
            SafeSynthesizerParameters.from_params(
                structured_generation_backend="xgrammar",
                **{"generation.structured_generation.backend": "outlines"},
            )

    def test_nested_dict_with_no_legacy_keys_uses_dict_values(self) -> None:
        """When no legacy flat keys are present, nested dict values are used as-is."""
        params = GenerateParameters.model_validate(
            {
                "structured_generation": {
                    "enabled": True,
                    "backend": "outlines",
                    "schema_method": "json_schema",
                }
            }
        )
        assert params.structured_generation.enabled is True
        assert params.structured_generation.backend == "outlines"
        assert params.structured_generation.schema_method == "json_schema"

    def test_legacy_keys_with_no_nested_dict_are_migrated(self) -> None:
        """When only legacy flat keys are present, migration produces the correct nested values."""
        params = GenerateParameters.model_validate(
            {
                "use_structured_generation": True,
                "structured_generation_backend": "xgrammar",
                "structured_generation_schema_method": "structural_tag",
                "structured_generation_use_single_sequence": True,
            }
        )
        assert params.structured_generation.enabled is True
        assert params.structured_generation.backend == "xgrammar"
        assert params.structured_generation.schema_method == "structural_tag"
        assert params.structured_generation.use_single_sequence is True


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
