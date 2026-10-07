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
from nemo_safe_synthesizer.errors import InternalError
from nemo_safe_synthesizer.pii_replacer.replacement.generation import (
    FakerReplacementGenerator,
    ManagedReplacementGenerator,
    ReplacementGenerationRequest,
    ReplacementGenerator,
)
from nemo_safe_synthesizer.pii_replacer.replacement.types import CanonicalValue


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
        with pytest.raises(InternalError, match="requires the .* sampler backend"):
            generator_class(
                settings=PiiReplacementSettings(),
                sampler=PiiSamplerConfig(backend=backend),
            )

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

    @pytest.mark.parametrize(
        "dependency_tuple",
        [
            [(EntityType.ORGANIZATION, None)],
            (("organization", None),),
            ((EntityType.ORGANIZATION, "example"),),
            ((EntityType.ORGANIZATION,),),
        ],
    )
    def test_request_rejects_malformed_dependency_tuples(self, dependency_tuple: object) -> None:
        with pytest.raises(InternalError, match="effective_dependency_tuple must"):
            ReplacementGenerationRequest(
                entity_type=EntityType.EMAIL,
                original_value="ada@example.com",
                effective_dependency_tuple=dependency_tuple,  # ty: ignore[invalid-argument-type] -- deliberate invalid input
                pattern=None,
                seed=42,
            )

    def test_request_requires_resolved_dependency_labels_to_be_a_tuple(self) -> None:
        with pytest.raises(InternalError, match="resolved_dependency_labels must be a tuple"):
            ReplacementGenerationRequest(
                entity_type=EntityType.FIRST_NAME,
                original_value="Ada",
                effective_dependency_tuple=(),
                pattern=None,
                seed=42,
                resolved_dependency_labels=[],  # ty: ignore[invalid-argument-type] -- deliberate invalid input
            )
