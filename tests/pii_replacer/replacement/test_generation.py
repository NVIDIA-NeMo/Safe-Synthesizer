# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from nemo_safe_synthesizer.config.replace_pii import EntityType
from nemo_safe_synthesizer.pii_replacer.replacement.generation import (
    ReplacementGenerationRequest,
    ReplacementGenerator,
)


class _FakeGenerator:
    def generate(self, request: ReplacementGenerationRequest) -> str:
        return f"synthetic-{request.entity_type.value}"


def _generate(generator: ReplacementGenerator, request: ReplacementGenerationRequest) -> str:
    return generator.generate(request)


@pytest.mark.unit
class TestReplacementGenerator:
    def test_generator_is_a_structural_interface_for_one_replacement(self) -> None:
        request = ReplacementGenerationRequest(
            entity_type=EntityType.EMAIL,
            canonical_original_value="ada@example.com",
            effective_dependency_tuple=((EntityType.ORGANIZATION, "example"),),
            pattern="{first_name}.{last_name}@example.com",
            seed=42,
        )

        assert _generate(_FakeGenerator(), request) == "synthetic-email"

    def test_request_is_immutable_and_hides_sensitive_inputs_from_repr(self) -> None:
        request = ReplacementGenerationRequest(
            entity_type=EntityType.FULL_NAME,
            canonical_original_value="Ada Lovelace",
            effective_dependency_tuple=((EntityType.ORGANIZATION, "Analytical Engines"),),
            pattern=None,
            seed=42,
        )

        with pytest.raises(FrozenInstanceError):
            setattr(request, "seed", 7)
        assert "Ada Lovelace" not in repr(request)
        assert "Analytical Engines" not in repr(request)
