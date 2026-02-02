# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias, Union

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
    "PathLike",
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

OptionalListOrStr = Union[str, list[str]] | None
OptionalListOrInt = Union[int, list[int]] | None

OptionalStrList: TypeAlias = list[str] | None
OptionalIntList: TypeAlias = list[int] | None

OptionalDictNestedStr = dict[str, Union[str, dict, list]] | None
OptionalStrDict = dict[str, str] | None

PathLike: TypeAlias = str | Path
