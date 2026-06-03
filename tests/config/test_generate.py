# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, cast

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.generate import (
    GenerateParameters,
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
class TestGenerateParametersStructuralTagValidation:
    @staticmethod
    def _generation_kwargs(*, schema_method: str = "structural_tag", backend: str = "xgrammar") -> dict[str, Any]:
        return {
            "use_structured_generation": True,
            "structured_generation_schema_method": schema_method,
            "structured_generation_backend": backend,
        }

    @pytest.mark.parametrize("backend", ["xgrammar", "auto"])
    def test_compatible_backends_validate(self, backend: str) -> None:
        GenerateParameters(**self._generation_kwargs(backend=backend))

    def test_incompatible_backend_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="requires `structured_generation_backend`"):
            GenerateParameters(**self._generation_kwargs(backend="outlines"))

    @pytest.mark.parametrize("backend", ["outlines", "guidance", "lm-format-enforcer"])
    def test_auto_with_incompatible_backend_validates(self, backend: str) -> None:
        GenerateParameters(**self._generation_kwargs(schema_method="auto", backend=backend))

    def test_default_schema_method_is_auto(self) -> None:
        params = GenerateParameters()
        assert params.structured_generation_schema_method == "auto"

    def test_skipped_when_structured_generation_disabled(self) -> None:
        GenerateParameters(
            use_structured_generation=False,
            structured_generation_schema_method="structural_tag",
            structured_generation_backend="outlines",
        )

    def test_from_params_rejects_incompatible_backend(self) -> None:
        with pytest.raises(ValidationError, match="outlines"):
            SafeSynthesizerParameters.from_params(**self._generation_kwargs(backend="outlines"))
