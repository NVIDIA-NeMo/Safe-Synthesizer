# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for presenting configuration validation failures."""

from __future__ import annotations

from pydantic import ValidationError

__all__ = ["format_pydantic_validation_error"]


def format_pydantic_validation_error(error: ValidationError) -> str:
    """Return compact, path-qualified details for a Pydantic validation error."""
    details: list[str] = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"])
        message = item["msg"]
        details.append(f"{location}: {message}" if location else message)
    return "; ".join(details)
