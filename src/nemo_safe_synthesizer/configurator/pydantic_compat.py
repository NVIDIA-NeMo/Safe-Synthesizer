# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-level Pydantic annotation compatibility helpers."""

from __future__ import annotations

import types
from typing import Annotated, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def unwrap_optional_annotation(annotation: object) -> object:
    """Unwrap ``Annotated`` and a union with one non-``None`` member."""
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin not in (types.UnionType, Union):
        return annotation
    members = tuple(unwrap_optional_annotation(item) for item in get_args(annotation) if item is not type(None))
    return members[0] if len(members) == 1 else annotation


def nested_model_type(annotation: object, expected_base: type[ModelT]) -> type[ModelT] | None:
    """Return the nested model type when an annotation has one compatible model."""
    annotation = unwrap_optional_annotation(annotation)
    if isinstance(annotation, type) and issubclass(annotation, expected_base):
        return annotation
    return None
