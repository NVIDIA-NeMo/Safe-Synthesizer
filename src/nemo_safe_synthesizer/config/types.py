# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable type aliases for configuration fields.

The ``Auto*Param`` and ``Optional*`` aliases let config fields accept the
sentinel string ``"auto"`` alongside their native type, enabling deferred
resolution at runtime. Collection aliases reduce boilerplate indownstream
Pydantic models.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

__all__ = [
    "AUTO_STR",
    "AutoStrParam",
    "AutoIntParam",
    "AutoFloatParam",
    "AutoBoolParam",
    "OptionalAutoStr",
    "OptionalAutoInt",
    "OptionalAutoFloat",
    "OptionalAutoBool",
    "OptionalStrList",
    "OptionalIntList",
    "OptionalListOrStr",
    "OptionalListOrInt",
    "OptionalDictNestedStr",
    "OptionalStrDict",
]

AUTO_STR = "auto"
AutoStrL = Literal["auto"]

AutoStrParam: TypeAlias = AutoStrL | str
AutoIntParam: TypeAlias = AutoStrL | int
AutoFloatParam: TypeAlias = AutoStrL | float
AutoBoolParam: TypeAlias = AutoStrL | bool

OptionalAutoStr: TypeAlias = AutoStrL | str | None
OptionalAutoInt: TypeAlias = AutoStrL | int | None
OptionalAutoFloat: TypeAlias = AutoStrL | float | None
OptionalAutoBool: TypeAlias = AutoStrL | bool | None

OptionalListOrStr = str | list[str] | None
OptionalListOrInt = int | list[int] | None

OptionalStrList: TypeAlias = list[str] | None
OptionalIntList: TypeAlias = list[int] | None

OptionalDictNestedStr = dict[str, str | dict | list] | None
OptionalStrDict = dict[str, str] | None
