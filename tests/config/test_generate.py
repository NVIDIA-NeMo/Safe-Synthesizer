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
