# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private schema-aware parameter path primitives."""

from __future__ import annotations

import types
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Annotated, Self, Union, cast, get_args, get_origin

if TYPE_CHECKING:
    from .parameters import Parameters


@dataclass(frozen=True, slots=True)
class ParameterPath:
    """Canonical path to a parameter field."""

    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parts or any(not part for part in self.parts):
            raise ValueError("A parameter path cannot contain empty segments.")

    def __str__(self) -> str:
        return ".".join(self.parts)


class ParameterFieldKind(Enum):
    """Schema classification for a parameter field."""

    BRANCH = auto()
    LEAF = auto()


def classify_parameter_annotation(annotation: object) -> ParameterFieldKind:
    """Classify a Pydantic field annotation as a branch or leaf."""
    if _nested_parameters_type(annotation) is not None:
        return ParameterFieldKind.BRANCH
    return ParameterFieldKind.LEAF


def _unwrap_annotated(annotation: object) -> object:
    while get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    return annotation


def _nested_parameters_type(annotation: object) -> type[Parameters] | None:
    from .parameters import Parameters

    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)
    if origin in (types.UnionType, Union):
        members = tuple(_unwrap_annotated(member) for member in get_args(annotation) if member is not type(None))
        if len(members) != 1:
            return None
        annotation = members[0]
    if isinstance(annotation, type) and issubclass(annotation, Parameters):
        return annotation
    return None


@dataclass(frozen=True, slots=True)
class ParameterField:
    """One indexed field in a parameter schema."""

    path: ParameterPath
    kind: ParameterFieldKind


@dataclass(frozen=True, slots=True)
class ResolvedParameterName:
    """A parameter name resolved to one canonical path."""

    path: ParameterPath


@dataclass(frozen=True, slots=True)
class UnknownParameterName:
    """A parameter name not present in the schema."""

    name: str


@dataclass(frozen=True, slots=True)
class AmbiguousParameterName:
    """A bare parameter name with multiple canonical candidates."""

    name: str
    candidates: tuple[ParameterPath, ...]


ParameterNameResolution = ResolvedParameterName | UnknownParameterName | AmbiguousParameterName


@dataclass(frozen=True, slots=True)
class ParameterSchema:
    """Indexed field paths for one ``Parameters`` model type."""

    model_type: type[Parameters]
    fields: tuple[ParameterField, ...]

    @classmethod
    def from_model(cls, model_type: type[Parameters]) -> Self:
        """Build a schema from Pydantic field annotations."""
        from .parameters import Parameters

        if not issubclass(model_type, Parameters):
            raise TypeError(f"Expected a Parameters model type, received {model_type!r}.")
        fields = tuple(_iter_parameter_fields(model_type))
        return cls(model_type=model_type, fields=fields)

    def resolve(self, name: str) -> ParameterNameResolution:
        """Resolve a canonical dotted or bare parameter name."""
        if "." in name:
            try:
                requested = split_parameter_path(name)
            except ValueError:
                return UnknownParameterName(name)
            if any(field.path == requested for field in self.fields):
                return ResolvedParameterName(requested)
            return UnknownParameterName(name)

        top_level = next((field.path for field in self.fields if field.path.parts == (name,)), None)
        if top_level is not None:
            return ResolvedParameterName(top_level)
        candidates = tuple(field.path for field in self.fields if field.path.parts[-1] == name)
        if not candidates:
            return UnknownParameterName(name)
        if len(candidates) > 1:
            return AmbiguousParameterName(name, candidates)
        return ResolvedParameterName(candidates[0])


def _iter_parameter_fields(model_type: type[Parameters], prefix: tuple[str, ...] = ()) -> tuple[ParameterField, ...]:
    fields: list[ParameterField] = []
    for name, field_info in model_type.model_fields.items():
        path = ParameterPath((*prefix, name))
        kind = classify_parameter_annotation(field_info.annotation)
        fields.append(ParameterField(path, kind))
        if kind is ParameterFieldKind.BRANCH:
            nested_type = _nested_parameters_type(field_info.annotation)
            if nested_type is not None:
                fields.extend(_iter_parameter_fields(nested_type, path.parts))
    return tuple(fields)


def split_parameter_path(name: str, separator: str = ".") -> ParameterPath:
    """Split a parameter name into a canonical path."""
    if not separator:
        raise ValueError("A parameter path separator cannot be empty.")
    parts = tuple(name.split(separator))
    if any(not part for part in parts):
        raise ValueError(f"Invalid parameter path {name!r}: empty segment.")
    return ParameterPath(parts)


def insert_parameter_value(target: dict[str, object], path: ParameterPath, value: object) -> None:
    """Insert a value at an already resolved path."""
    current = target
    for part in path.parts[:-1]:
        value_at_part = current.get(part)
        if isinstance(value_at_part, dict):
            current = cast(dict[str, object], value_at_part)
            continue
        nested: dict[str, object] = {}
        current[part] = nested
        current = nested
    current[path.parts[-1]] = value
