# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed-asset structured PII replacement generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from ....config.replace_pii import PiiReplacementSettings, PiiSamplerBackend, PiiSamplerConfig
from ....errors import InternalError

if TYPE_CHECKING:
    from ..generation import ReplacementGenerationRequest

__all__ = ["ManagedReplacementGenerator"]


class ManagedReplacementGenerator:
    """Generate structured replacements using managed person-sampling assets.

    Args:
        settings: Locale and seed configuration shared by replacement
            generators.
        sampler: Managed sampler configuration, including its asset path.

    Replacement execution is introduced by a follow-up change.
    """

    backend: ClassVar[PiiSamplerBackend] = PiiSamplerBackend.MANAGED

    def __init__(self, *, settings: PiiReplacementSettings, sampler: PiiSamplerConfig) -> None:
        if sampler.backend is not self.backend:
            raise InternalError(
                f"{type(self).__name__} requires the {self.backend.value} sampler backend, got {sampler.backend.value}"
            )
        self._settings = settings
        self._sampler = sampler

    def generate(self, request: ReplacementGenerationRequest) -> str:
        """Generate a managed-asset replacement for ``request``."""
        raise NotImplementedError("managed replacement generation is not implemented")
