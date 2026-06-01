# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation-config checks that should fail before training starts."""

from __future__ import annotations

from ...config.generate import structural_tag_backend_error_message
from ..base import ConfigCheck, IssueCollector
from ..types import ConfigView, PreflightContext

__all__ = ["StructuralTagBackendCheck"]


class StructuralTagBackendCheck(ConfigCheck):
    """Reject ``structural_tag`` when the structured-output backend cannot support it."""

    name = "config.structured_tag_backend"
    label = "Structural Tag backend"
    category = "configuration"

    def enabled(self, ctx: PreflightContext) -> bool:
        if not super().enabled(ctx):
            return False
        generation = ctx.config.generation
        return (
            generation.use_structured_generation and generation.structured_generation_schema_method == "structural_tag"
        )

    def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
        message = structural_tag_backend_error_message(ctx.config.generation.structured_generation_backend)
        if message is not None:
            collector.error("structured_tag_backend_incompatible", message)
