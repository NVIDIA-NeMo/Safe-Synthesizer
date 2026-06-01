# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.config.generate import GenerateParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters


@pytest.mark.unit
class TestGenerateParametersStructuralTagValidation:
    @staticmethod
    def _generation_kwargs(*, backend: str) -> dict[str, Any]:
        return {
            "use_structured_generation": True,
            "structured_generation_schema_method": "structural_tag",
            "structured_generation_backend": backend,
        }

    @pytest.mark.parametrize("backend", ["xgrammar", "auto"])
    def test_compatible_backends_validate(self, backend: str) -> None:
        GenerateParameters(**self._generation_kwargs(backend=backend))

    def test_incompatible_backend_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError, match="requires `structured_generation_backend`"):
            GenerateParameters(**self._generation_kwargs(backend="outlines"))

    def test_skipped_when_structured_generation_disabled(self) -> None:
        GenerateParameters(
            use_structured_generation=False,
            structured_generation_schema_method="structural_tag",
            structured_generation_backend="outlines",
        )

    def test_from_params_rejects_incompatible_backend(self) -> None:
        with pytest.raises(ValidationError, match="outlines"):
            SafeSynthesizerParameters.from_params(**self._generation_kwargs(backend="outlines"))
