# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placeholder configuration for the next PII replacement implementation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar, cast

from pydantic import model_validator

from ..config.unknown_fields import raise_if_removed_legacy_fields
from ..configurator.parameters import Parameters

__all__ = ["ReplacePiiConfig"]


class ReplacePiiConfig(Parameters):
    """Mark PII replacement as requested.

    The replacement engine and its full configuration contract are intentionally
    absent on this branch. Set ``replace_pii`` to ``None`` to run the pipeline.
    """

    removed_legacy_fields: ClassVar[frozenset[str]] = frozenset({"globals", "steps"})
    removed_legacy_fields_message: ClassVar[str] = (
        "PII replacement v2 configuration was removed. "
        "See docs/user-guide/configuration.md#replacing-pii "
        "for the current configuration."
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_v2_fields(cls, value: object) -> object:
        if isinstance(value, Mapping):
            raise_if_removed_legacy_fields(cls, cast(Mapping[str, object], value), path=())
        return value
