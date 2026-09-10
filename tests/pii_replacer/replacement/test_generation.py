# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import FrozenInstanceError

import pytest

from nemo_safe_synthesizer.config.replace_pii import (
    EntityType,
    PiiReplacementSettings,
    PiiSamplerBackend,
    PiiSamplerConfig,
)
from nemo_safe_synthesizer.pii_replacer.replacement.generation import (
    FakerReplacementGenerator,
    ManagedReplacementGenerator,
    ReplacementGenerationRequest,
    ReplacementGenerator,
)
from nemo_safe_synthesizer.pii_replacer.replacement.types import CanonicalValue


class _FakeGenerator:
    backend = PiiSamplerBackend.FAKER

    def generate(self, request: ReplacementGenerationRequest) -> str:
        return f"synthetic-{request.entity_type.value}"


def _generate(generator: ReplacementGenerator, request: ReplacementGenerationRequest) -> str:
    return generator.generate(request)


@pytest.mark.unit
class TestReplacementGenerator:
    @pytest.mark.parametrize(
        ("generator_class", "backend"),
        [
            (ManagedReplacementGenerator, PiiSamplerBackend.MANAGED),
            (FakerReplacementGenerator, PiiSamplerBackend.FAKER),
        ],
    )
    def test_named_generator_adapters_declare_their_backend(
        self,
        generator_class: type[ReplacementGenerator],
        backend: PiiSamplerBackend,
    ) -> None:
        generator = generator_class(
            settings=PiiReplacementSettings(),
            sampler=PiiSamplerConfig(backend=backend),
        )

        assert generator.backend is backend
        with pytest.raises(NotImplementedError, match="replacement generation is not implemented"):
            generator.generate(
                ReplacementGenerationRequest(
                    entity_type=EntityType.FIRST_NAME,
                    original_value="Ada",
                    effective_dependency_tuple=(),
                    pattern=None,
                    seed=42,
                )
            )

    @pytest.mark.parametrize(
        ("generator_class", "backend"),
        [
            (ManagedReplacementGenerator, PiiSamplerBackend.FAKER),
            (FakerReplacementGenerator, PiiSamplerBackend.MANAGED),
        ],
    )
    def test_named_generator_adapters_reject_the_wrong_backend(
        self,
        generator_class: type[ReplacementGenerator],
        backend: PiiSamplerBackend,
    ) -> None:
        with pytest.raises(ValueError, match="requires the .* sampler backend"):
            generator_class(
                settings=PiiReplacementSettings(),
                sampler=PiiSamplerConfig(backend=backend),
            )

    def test_generator_is_a_structural_interface_for_one_replacement(self) -> None:
        request = ReplacementGenerationRequest(
            entity_type=EntityType.EMAIL,
            original_value="ada@example.com",
            effective_dependency_tuple=(
                (EntityType.ORGANIZATION, CanonicalValue(type_tag="string", normalized_value="example")),
            ),
            pattern="{first_name}.{last_name}@example.com",
            seed=42,
        )

        assert _generate(_FakeGenerator(), request) == "synthetic-email"

    def test_request_is_immutable_and_hides_sensitive_inputs_from_repr(self) -> None:
        request = ReplacementGenerationRequest(
            entity_type=EntityType.FULL_NAME,
            original_value="Ada Lovelace",
            effective_dependency_tuple=(
                (
                    EntityType.ORGANIZATION,
                    CanonicalValue(type_tag="string", normalized_value="Analytical Engines"),
                ),
            ),
            pattern=None,
            seed=42,
        )

        with pytest.raises(FrozenInstanceError):
            setattr(request, "seed", 7)
        assert "Ada Lovelace" not in repr(request)
        assert "Analytical Engines" not in repr(request)
