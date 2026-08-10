# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unknown-field policy shared by configuration input boundaries."""

from __future__ import annotations

import types
from collections.abc import Mapping, Sequence
from typing import Annotated, Literal, TypeAlias, Union, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from ..errors import ParameterError

UnknownFieldBehavior: TypeAlias = Literal["ignore", "reject"]

DEFAULT_UNKNOWN_FIELDS: UnknownFieldBehavior = "reject"

_UNKNOWN_FIELDS_ADAPTER: TypeAdapter[UnknownFieldBehavior] = TypeAdapter(UnknownFieldBehavior)


def validate_unknown_fields(value: object) -> UnknownFieldBehavior:
    """Validate one user-facing unknown-field policy value."""
    return _UNKNOWN_FIELDS_ADAPTER.validate_python(value)


def normalize_unknown_fields(
    model_type: type[BaseModel],
    source: Mapping[str, object],
    unknown_fields: UnknownFieldBehavior,
) -> dict[str, object]:
    """Recursively reject or remove fields absent from a Pydantic model tree."""
    return _normalize_model_mapping(model_type, source, unknown_fields, ())


def _normalize_model_mapping(
    model_type: type[BaseModel],
    source: Mapping[str, object],
    unknown_fields: UnknownFieldBehavior,
    path: tuple[str, ...],
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for name, value in source.items():
        field = model_type.model_fields.get(name)
        if field is None:
            if unknown_fields == "reject":
                location = ".".join((*path, name))
                raise ParameterError(f"Unknown configuration field {location!r}.")
            continue
        normalized[name] = _normalize_annotation(field.annotation, value, unknown_fields, (*path, name))
    return normalized


def _normalize_annotation(
    annotation: object,
    value: object,
    unknown_fields: UnknownFieldBehavior,
    path: tuple[str, ...],
) -> object:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if isinstance(value, Mapping):
            return _normalize_model_mapping(
                annotation,
                cast(Mapping[str, object], value),
                unknown_fields,
                path,
            )
        return value

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (types.UnionType, Union):
        candidates = tuple(item for item in args if item is not type(None))
        model_candidates = tuple(item for item in candidates if _annotation_contains_model(item))
        if len(model_candidates) == 1:
            return _normalize_annotation(model_candidates[0], value, unknown_fields, path)
        return value

    if origin in (list, set, frozenset, Sequence) and args and isinstance(value, Sequence):
        if isinstance(value, (str, bytes, bytearray)):
            return value
        items = (
            _normalize_annotation(args[0], item, unknown_fields, (*path, str(index)))
            for index, item in enumerate(value)
        )
        return type(value)(items)

    if origin is tuple and args and isinstance(value, tuple):
        item_annotations = args if args[-1:] != (Ellipsis,) else (args[0],) * len(value)
        return tuple(
            _normalize_annotation(item_annotations[index], item, unknown_fields, (*path, str(index)))
            for index, item in enumerate(value)
        )

    if origin in (dict, Mapping) and len(args) == 2 and isinstance(value, Mapping):
        return {
            key: _normalize_annotation(args[1], item, unknown_fields, (*path, str(key))) for key, item in value.items()
        }

    return value


def _annotation_contains_model(annotation: object) -> bool:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return True
    return any(_annotation_contains_model(item) for item in get_args(annotation) if item is not Ellipsis)
