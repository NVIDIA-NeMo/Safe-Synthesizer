# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private schema-aware parameter path primitives."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Self, cast

from pydantic import BaseModel
from typing_extensions import override

from ..errors import ParameterError
from .pydantic_compat import nested_model_type

if TYPE_CHECKING:
    from .parameters import Parameters


@dataclass(frozen=True, slots=True)
class ParameterPath:
    """Canonical path to a parameter field."""

    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parts or any(not part for part in self.parts):
            raise ValueError("A parameter path cannot contain empty segments.")

    @override
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


def _nested_parameters_type(annotation: object) -> type[Parameters] | None:
    from .parameters import Parameters

    return nested_model_type(annotation, Parameters)


@dataclass(frozen=True, slots=True)
class ParameterField:
    """One indexed field in a parameter schema."""

    path: ParameterPath
    kind: ParameterFieldKind


@dataclass(frozen=True, slots=True)
class ParameterAlias:
    """One accepted compatibility name and its canonical parameter path."""

    name: str
    path: ParameterPath


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
    aliases: tuple[ParameterAlias, ...]

    @classmethod
    def from_model(cls, model_type: type[Parameters]) -> Self:
        """Build a schema from Pydantic field annotations."""
        from .parameters import Parameters

        if not issubclass(model_type, Parameters):
            raise TypeError(f"Expected a Parameters model type, received {model_type!r}.")
        fields = tuple(_iter_parameter_fields(model_type))
        aliases = tuple(_iter_parameter_aliases(model_type))
        field_paths = {field.path for field in fields}
        for alias in aliases:
            if alias.path not in field_paths:
                raise TypeError(
                    f"Parameter alias {alias.name!r} on {model_type.__name__} targets unknown path {str(alias.path)!r}."
                )
        return cls(model_type=model_type, fields=fields, aliases=aliases)

    def resolve(self, name: str, *, infer_bare_name: bool = True) -> ParameterNameResolution:
        """Resolve a canonical dotted or bare parameter name."""
        if "." in name:
            try:
                requested = split_parameter_path(name)
            except ValueError:
                return UnknownParameterName(name)
            if any(field.path == requested for field in self.fields):
                return ResolvedParameterName(requested)
            return _resolution_from_candidates(name, self._alias_candidates(name))

        top_level = next((field.path for field in self.fields if field.path.parts == (name,)), None)
        if top_level is not None:
            return ResolvedParameterName(top_level)
        aliases = self._alias_candidates(name)
        if aliases:
            return _resolution_from_candidates(name, aliases)
        if not infer_bare_name:
            return UnknownParameterName(name)
        candidates = tuple(field.path for field in self.fields if field.path.parts[-1] == name)
        return _resolution_from_candidates(name, candidates)

    def require(self, name: str, *, infer_bare_name: bool = True) -> ParameterPath:
        """Resolve one name or raise a user-facing configuration error."""
        resolution = self.resolve(name, infer_bare_name=infer_bare_name)
        if not infer_bare_name and isinstance(resolution, UnknownParameterName) and "." not in name:
            inferred = tuple(field.path for field in self.fields if field.path.parts[-1] == name)
            if len(inferred) == 1:
                path = inferred[0]
                parent = path.parts[0]
                raise ParameterError(
                    f"Nested parameter name {name!r} is not a direct override; "
                    f"use {str(path)!r} or pass the {parent!r} mapping."
                )
            if len(inferred) > 1:
                resolution = AmbiguousParameterName(name, inferred)

        match resolution:
            case ResolvedParameterName() as resolved:
                return resolved.path
            case UnknownParameterName() as unknown:
                kind = "path" if "." in name else "name"
                raise ParameterError(f"Unknown parameter {kind} {unknown.name!r}.")
            case AmbiguousParameterName() as ambiguous:
                choices = ", ".join(str(path) for path in ambiguous.candidates)
                raise ParameterError(f"Ambiguous parameter name {ambiguous.name!r}; use one of: {choices}.")
        raise ParameterError(f"Unexpected parameter resolution for {name!r}.")

    def normalize_aliases(self, source: Mapping[str, object]) -> dict[str, object]:
        """Translate declared aliases to canonical paths, with aliases taking precedence."""
        values = dict(source)
        for name, field_info in self.model_type.model_fields.items():
            nested_type = _nested_parameters_type(field_info.annotation)
            value = values.get(name)
            if nested_type is not None and isinstance(value, Mapping):
                values[name] = ParameterSchema.from_model(nested_type).normalize_aliases(
                    cast(Mapping[str, object], value)
                )

        for name in tuple(values):
            if name in self.model_type.model_fields:
                continue
            candidates = self._alias_candidates(name)
            if not candidates:
                continue
            resolution = _resolution_from_candidates(name, candidates)
            if isinstance(resolution, AmbiguousParameterName):
                choices = ", ".join(str(path) for path in resolution.candidates)
                raise ParameterError(f"Ambiguous parameter alias {name!r}; use one of: {choices}.")
            if isinstance(resolution, ResolvedParameterName):
                _set_parameter_value(values, resolution.path, values.pop(name))
        return values

    def _alias_candidates(self, name: str) -> tuple[ParameterPath, ...]:
        return tuple(alias.path for alias in self.aliases if alias.name == name)


def _resolution_from_candidates(name: str, candidates: tuple[ParameterPath, ...]) -> ParameterNameResolution:
    unique = tuple(sorted(set(candidates), key=lambda path: path.parts))
    if not unique:
        return UnknownParameterName(name)
    if len(unique) > 1:
        return AmbiguousParameterName(name, unique)
    return ResolvedParameterName(unique[0])


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


def _iter_parameter_aliases(model_type: type[Parameters], prefix: tuple[str, ...] = ()) -> tuple[ParameterAlias, ...]:
    aliases: list[ParameterAlias] = []
    for name, target in model_type.parameter_aliases.items():
        target_path = split_parameter_path(target)
        canonical_path = ParameterPath((*prefix, *target_path.parts))
        aliases.append(ParameterAlias(name, canonical_path))
        if prefix:
            aliases.append(ParameterAlias(".".join((*prefix, name)), canonical_path))

    for name, field_info in model_type.model_fields.items():
        nested_type = _nested_parameters_type(field_info.annotation)
        if nested_type is not None:
            aliases.extend(_iter_parameter_aliases(nested_type, (*prefix, name)))
    return tuple(aliases)


def _set_parameter_value(target: dict[str, object], path: ParameterPath, value: object) -> None:
    current = target
    for part in path.parts[:-1]:
        branch = current.get(part)
        if isinstance(branch, BaseModel):
            nested = branch.model_dump(exclude_unset=True)
        elif isinstance(branch, Mapping):
            nested = dict(branch)
        else:
            nested = {}
        current[part] = nested
        current = nested
    current[path.parts[-1]] = value


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
    for index, part in enumerate(path.parts[:-1]):
        value_at_part = current.get(part)
        if isinstance(value_at_part, dict):
            current = cast(dict[str, object], value_at_part)
            continue
        if part in current:
            prefix = ".".join(path.parts[: index + 1])
            raise ValueError(f"Conflicting override paths for {str(path)!r}: {prefix!r} already has a parent value.")
        nested: dict[str, object] = {}
        current[part] = nested
        current = nested
    leaf = path.parts[-1]
    if leaf in current:
        raise ValueError(f"Conflicting override paths for {str(path)!r}: nested values already exist below this path.")
    current[leaf] = value
