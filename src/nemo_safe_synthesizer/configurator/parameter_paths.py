# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private schema-aware parameter path primitives."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Self, cast

from pydantic import BaseModel
from typing_extensions import override

from ..errors import ParameterError
from .pydantic_compat import nested_model_type

if TYPE_CHECKING:
    from .parameters import Parameters

PARAMETER_PATH_SEPARATOR = "."


def format_parameter_path(parts: Iterable[str]) -> str:
    """Join path segments into the canonical dotted string."""
    return PARAMETER_PATH_SEPARATOR.join(parts)


@dataclass(frozen=True, slots=True)
class ParameterPath:
    """Canonical path to a parameter field."""

    parts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.parts or any(not part for part in self.parts):
            raise ValueError("A parameter path cannot contain empty segments.")

    @override
    def __str__(self) -> str:
        return format_parameter_path(self.parts)


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
        if PARAMETER_PATH_SEPARATOR in name:
            try:
                requested = split_parameter_path(name)
            except ValueError:
                return UnknownParameterName(name)
            if any(field.path == requested for field in self.fields):
                return ResolvedParameterName(requested)
            return _resolution_from_candidates(name, self._alias_candidates(name))

        if (top_level := next((field.path for field in self.fields if field.path.parts == (name,)), None)) is not None:
            return ResolvedParameterName(top_level)
        if aliases := self._alias_candidates(name):
            return _resolution_from_candidates(name, aliases)
        if not infer_bare_name:
            return UnknownParameterName(name)
        return _resolution_from_candidates(name, self._fields_ending_with(name))

    def require(self, name: str, *, infer_bare_name: bool = True) -> ParameterPath:
        """Resolve one name or raise a user-facing configuration error."""
        resolution = self.resolve(name, infer_bare_name=infer_bare_name)
        if (
            not infer_bare_name
            and isinstance(resolution, UnknownParameterName)
            and PARAMETER_PATH_SEPARATOR not in name
        ):
            inferred = self._fields_ending_with(name)
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
                kind = "path" if PARAMETER_PATH_SEPARATOR in name else "name"
                raise ParameterError(f"Unknown parameter {kind} {unknown.name!r}.")
            case AmbiguousParameterName() as ambiguous:
                raise _ambiguous_error("name", ambiguous.name, ambiguous.candidates)
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
            match _resolution_from_candidates(name, candidates):
                case AmbiguousParameterName() as ambiguous:
                    raise _ambiguous_error("alias", name, ambiguous.candidates)
                case ResolvedParameterName() as resolved:
                    _set_parameter_value(values, resolved.path, values.pop(name))
                case UnknownParameterName():
                    pass
        return values

    def _alias_candidates(self, name: str) -> tuple[ParameterPath, ...]:
        return tuple(alias.path for alias in self.aliases if alias.name == name)

    def _fields_ending_with(self, name: str) -> tuple[ParameterPath, ...]:
        return tuple(field.path for field in self.fields if field.path.parts[-1] == name)


def _ambiguous_error(kind: str, name: str, candidates: tuple[ParameterPath, ...]) -> ParameterError:
    choices = ", ".join(str(path) for path in candidates)
    return ParameterError(f"Ambiguous parameter {kind} {name!r}; use one of: {choices}.")


def _resolution_from_candidates(name: str, candidates: tuple[ParameterPath, ...]) -> ParameterNameResolution:
    unique = tuple(sorted(set(candidates), key=lambda path: path.parts))
    if not unique:
        return UnknownParameterName(name)
    if len(unique) > 1:
        return AmbiguousParameterName(name, unique)
    return ResolvedParameterName(unique[0])


def _walk_parameter_models(
    model_type: type[Parameters], prefix: tuple[str, ...] = ()
) -> Iterator[tuple[type[Parameters], tuple[str, ...]]]:
    """Yield every ``Parameters`` model in the tree with its path prefix, pre-order."""
    yield model_type, prefix
    for name, field_info in model_type.model_fields.items():
        nested_type = _nested_parameters_type(field_info.annotation)
        if nested_type is not None:
            yield from _walk_parameter_models(nested_type, (*prefix, name))


def _iter_parameter_fields(model_type: type[Parameters]) -> tuple[ParameterField, ...]:
    fields: list[ParameterField] = []
    for model, prefix in _walk_parameter_models(model_type):
        for name, field_info in model.model_fields.items():
            path = ParameterPath((*prefix, name))
            fields.append(ParameterField(path, classify_parameter_annotation(field_info.annotation)))
    return tuple(fields)


def _iter_parameter_aliases(model_type: type[Parameters]) -> tuple[ParameterAlias, ...]:
    aliases: list[ParameterAlias] = []
    for model, prefix in _walk_parameter_models(model_type):
        for name, target in model.parameter_aliases.items():
            canonical_path = ParameterPath((*prefix, *split_parameter_path(target).parts))
            aliases.append(ParameterAlias(name, canonical_path))
            if prefix:
                aliases.append(ParameterAlias(format_parameter_path((*prefix, name)), canonical_path))
    return tuple(aliases)


def _set_parameter_value(target: dict[str, object], path: ParameterPath, value: object) -> None:
    current = target
    for part in path.parts[:-1]:
        match current.get(part):
            case BaseModel() as branch:
                nested = branch.model_dump(exclude_unset=True)
            case Mapping() as branch:
                nested = dict(branch)
            case _:
                nested = {}
        current[part] = nested
        current = nested
    current[path.parts[-1]] = value


def split_parameter_path(name: str, separator: str = PARAMETER_PATH_SEPARATOR) -> ParameterPath:
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
            prefix = format_parameter_path(path.parts[: index + 1])
            raise ValueError(f"Conflicting override paths for {str(path)!r}: {prefix!r} already has a parent value.")
        nested: dict[str, object] = {}
        current[part] = nested
        current = nested
    leaf = path.parts[-1]
    if leaf in current:
        raise ValueError(f"Conflicting override paths for {str(path)!r}: nested values already exist below this path.")
    current[leaf] = value
