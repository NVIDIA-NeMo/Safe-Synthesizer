# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private schema-aware configuration patch primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from pydantic import BaseModel

from ..configurator.parameter_paths import ParameterPath, format_parameter_path
from ..configurator.pydantic_compat import nested_model_type
from ..errors import ParameterError
from .unknown_fields import UnknownFieldBehavior, raise_if_removed_legacy_fields

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class PatchAssignment:
    """One canonical configuration assignment with source precedence."""

    path: ParameterPath
    value: object
    origin: str
    precedence: int


@dataclass(frozen=True, slots=True)
class CompiledConfigPatch(Generic[ModelT]):
    """A configuration patch compiled for one exact Pydantic model type."""

    target_model: type[ModelT]
    assignments: tuple[PatchAssignment, ...]

    @staticmethod
    def from_paths(target_model: type[ModelT], assignments: Iterable[PatchAssignment]) -> CompiledConfigPatch[ModelT]:
        _require_model_type(target_model)
        copied = tuple(
            PatchAssignment(item.path, deepcopy(item.value), item.origin, item.precedence) for item in assignments
        )
        for assignment in copied:
            _field_model_at_path(target_model, assignment.path)
        _validate_conflicts(target_model, copied)
        return CompiledConfigPatch(target_model=target_model, assignments=copied)

    @staticmethod
    def from_mapping(
        target_model: type[ModelT],
        source: Mapping[str, object],
        *,
        origin: str,
        precedence: int,
        unknown_fields: UnknownFieldBehavior,
    ) -> CompiledConfigPatch[ModelT]:
        """Compile a mapping with an explicit policy for traversable unknown fields.

        Collections are atomic patch leaves. Pydantic validates their contents
        when the patch is applied.
        """
        _require_model_type(target_model)
        if unknown_fields not in ("ignore", "reject"):
            raise ValueError(f"Unsupported unknown-field behavior {unknown_fields!r}.")
        compatible_source = _mapping_without_extras(target_model, source) if unknown_fields == "ignore" else source
        assignments = _mapping_assignments(target_model, compatible_source, origin=origin, precedence=precedence)
        return CompiledConfigPatch.from_paths(target_model, assignments)

    @staticmethod
    def from_model(
        target_model: type[ModelT], source: ModelT, *, origin: str, precedence: int
    ) -> CompiledConfigPatch[ModelT]:
        _require_exact_model(target_model, source)
        assignments = _mapping_assignments(
            target_model, _extract_set_fields(source), origin=origin, precedence=precedence
        )
        return CompiledConfigPatch.from_paths(target_model, assignments)

    def combine(self, other: CompiledConfigPatch[ModelT]) -> CompiledConfigPatch[ModelT]:
        if self.target_model is not other.target_model:
            raise TypeError(
                f"Cannot combine patches with different target models: "
                f"{self.target_model.__name__} and {other.target_model.__name__}."
            )
        return CompiledConfigPatch.from_paths(self.target_model, (*self.assignments, *other.assignments))

    def materialize(self) -> dict[str, object]:
        result: dict[str, object] = {}
        ordered = sorted(
            self.assignments,
            key=lambda item: (item.precedence, len(item.path.parts), item.path.parts, item.origin),
        )
        for assignment in ordered:
            _insert_value(result, self.target_model, assignment.path.parts, deepcopy(assignment.value))
        return result

    def apply(self, base: ModelT | None = None) -> ModelT:
        values: dict[str, object] = {}
        if base is not None:
            _require_exact_model(self.target_model, base)
            values = _extract_set_fields(base)
        _merge_model_mapping(values, self.target_model, self.materialize())
        return self.target_model.model_validate(values)

    def apply_to_full_model(self, base: ModelT) -> ModelT:
        """Apply to full current values while retaining sparse field presence."""
        _require_exact_model(self.target_model, base)
        patch_values = self.materialize()
        values = base.model_dump()
        _merge_model_mapping(values, self.target_model, patch_values)
        result = self.target_model.model_validate(values)
        _restore_model_fields_set(result, base, patch_values)
        return result


def _require_model_type(model_type: type[BaseModel]) -> None:
    if not isinstance(model_type, type) or not issubclass(model_type, BaseModel):
        raise TypeError(f"Expected a Pydantic target model, received {model_type!r}.")


def _require_exact_model(model_type: type[ModelT], value: BaseModel) -> None:
    if type(value) is not model_type:
        raise TypeError(f"Patch target model is {model_type.__name__}; received {type(value).__name__}.")


def _field_model_at_path(model_type: type[BaseModel], path: ParameterPath) -> type[BaseModel] | None:
    current = model_type
    for index, part in enumerate(path.parts):
        if (field := current.model_fields.get(part)) is None:
            raise ParameterError(f"Unknown configuration path {str(path)!r}.")
        nested = nested_model_type(field.annotation, BaseModel)
        if index == len(path.parts) - 1:
            return nested
        if nested is None:
            prefix = format_parameter_path(path.parts[: index + 1])
            raise ParameterError(f"Configuration path {str(path)!r} descends through atomic field {prefix!r}.")
        current = nested
    raise AssertionError("ParameterPath guarantees at least one path segment.")


def _mapping_assignments(
    model_type: type[BaseModel],
    source: Mapping[str, object],
    *,
    origin: str,
    precedence: int,
    prefix: tuple[str, ...] = (),
) -> tuple[PatchAssignment, ...]:
    assignments: list[PatchAssignment] = []
    for name, value in source.items():
        if name not in model_type.model_fields:
            raise_if_removed_legacy_fields(model_type, (name,), path=prefix)
            path = format_parameter_path((*prefix, name))
            raise ParameterError(f"Unknown configuration path {path!r}.")
        path = ParameterPath((*prefix, name))
        nested = _field_model_at_path(model_type, ParameterPath((name,)))
        nested_source = _branch_mapping(nested, value)
        if nested is None or nested_source is None or not nested_source:
            assignments.append(PatchAssignment(path, value, origin, precedence))
            continue
        nested_assignments = _mapping_assignments(
            nested,
            nested_source,
            origin=origin,
            precedence=precedence,
            prefix=path.parts,
        )
        if nested_assignments:
            assignments.extend(nested_assignments)
        else:
            # Pydantic keeps a known model branch explicit even when all of its
            # provided children are ignored extras.
            assignments.append(PatchAssignment(path, {}, origin, precedence))
    return tuple(assignments)


def _mapping_without_extras(
    model_type: type[BaseModel],
    source: Mapping[str, object],
    *,
    prefix: tuple[str, ...] = (),
) -> dict[str, object]:
    """Adapt a raw mapping to Pydantic's recursive extra-ignore behavior."""
    adapted: dict[str, object] = {}
    for name, value in source.items():
        if (field := model_type.model_fields.get(name)) is None:
            raise_if_removed_legacy_fields(model_type, (name,), path=prefix)
            continue
        nested_model = nested_model_type(field.annotation, BaseModel)
        nested_source = _branch_mapping(nested_model, value)
        adapted[name] = (
            _mapping_without_extras(nested_model, nested_source, prefix=(*prefix, name))
            if nested_model is not None and nested_source is not None
            else value
        )
    return adapted


def _branch_mapping(nested_model: type[BaseModel] | None, value: object) -> Mapping[str, object] | None:
    if nested_model is None:
        return None
    if isinstance(value, Mapping):
        return cast(Mapping[str, object], value)
    if type(value) is nested_model:
        return _extract_set_fields(value)
    return None


def _extract_set_fields(model: BaseModel) -> dict[str, object]:
    extracted: dict[str, object] = {}
    for name in type(model).model_fields:
        value = _stored_field_value(model, name)
        if isinstance(value, BaseModel):
            nested = _extract_set_fields(value)
            if nested or name in model.model_fields_set:
                extracted[name] = nested
            continue
        if name in model.model_fields_set:
            extracted[name] = deepcopy(value)
    return extracted


def _stored_field_value(model: BaseModel, name: str) -> object:
    """Read a validated field without triggering Pydantic access warnings."""
    return vars(model)[name]


def _restore_model_fields_set(result: BaseModel, base: BaseModel, patch: Mapping[str, object]) -> None:
    """Restore recursive base presence and add fields supplied by ``patch``."""
    _replace_model_fields_set(result, (*base.model_fields_set, *patch))
    for name in type(result).model_fields:
        result_value = _stored_field_value(result, name)
        if not isinstance(result_value, BaseModel):
            continue
        base_value = _stored_field_value(base, name)
        nested_patch = _branch_mapping(type(result_value), patch.get(name)) or {}
        if isinstance(base_value, BaseModel):
            _restore_model_fields_set(result_value, base_value, nested_patch)
        else:
            _set_model_fields_from_patch(result_value, nested_patch)


def _set_model_fields_from_patch(model: BaseModel, patch: Mapping[str, object]) -> None:
    _replace_model_fields_set(model, patch)
    for name in type(model).model_fields:
        value = _stored_field_value(model, name)
        if not isinstance(value, BaseModel):
            continue
        nested_patch = _branch_mapping(type(value), patch.get(name)) or {}
        _set_model_fields_from_patch(value, nested_patch)


def _replace_model_fields_set(model: BaseModel, fields: Iterable[str]) -> None:
    model.model_fields_set.clear()
    model.model_fields_set.update(fields)


def _validate_conflicts(model_type: type[BaseModel], assignments: tuple[PatchAssignment, ...]) -> None:
    ordered = sorted(assignments, key=lambda item: (item.path.parts, item.precedence, item.origin))
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.path == right.path and left.precedence == right.precedence:
                raise ParameterError(
                    f"Duplicate parameter path {str(left.path)!r} from origins {left.origin!r} and {right.origin!r}."
                )
            ancestor, descendant = _ancestor_pair(left, right)
            if ancestor is None or ancestor.precedence != descendant.precedence:
                continue
            if _branch_mapping(_field_model_at_path(model_type, ancestor.path), ancestor.value) is None:
                raise ParameterError(
                    f"Cannot assign nested parameter path {str(descendant.path)!r}; "
                    f"{str(ancestor.path)!r} is already set. This parent/child conflict is between "
                    f"origins {descendant.origin!r} and {ancestor.origin!r}."
                )


def _ancestor_pair(left: PatchAssignment, right: PatchAssignment) -> tuple[PatchAssignment | None, PatchAssignment]:
    if len(left.path.parts) < len(right.path.parts) and right.path.parts[: len(left.path.parts)] == left.path.parts:
        return left, right
    if len(right.path.parts) < len(left.path.parts) and left.path.parts[: len(right.path.parts)] == right.path.parts:
        return right, left
    return None, right


def _ensure_branch(target: dict[str, object], name: str) -> dict[str, object]:
    """Return ``target[name]`` as a nested dict, replacing any non-dict value."""
    branch = target.get(name)
    nested = cast(dict[str, object], branch) if isinstance(branch, dict) else {}
    target[name] = nested
    return nested


def _insert_value(
    target: dict[str, object], model_type: type[BaseModel], parts: tuple[str, ...], value: object
) -> None:
    name, *tail = parts
    field = model_type.model_fields[name]
    nested_model = nested_model_type(field.annotation, BaseModel)
    if tail:
        if nested_model is None:
            raise AssertionError("Validated paths cannot descend through atomic fields.")
        _insert_value(_ensure_branch(target, name), nested_model, tuple(tail), value)
        return
    branch_source = _branch_mapping(nested_model, value)
    if branch_source is None:
        target[name] = value
        return
    if nested_model is None:
        raise AssertionError("A branch mapping must have a nested model type.")
    _merge_model_mapping(_ensure_branch(target, name), nested_model, branch_source)


def _merge_model_mapping(target: dict[str, object], model_type: type[BaseModel], source: Mapping[str, object]) -> None:
    for name, value in source.items():
        if name not in model_type.model_fields:
            raise ParameterError(f"Unknown configuration path {name!r} for {model_type.__name__}.")
        _insert_value(target, model_type, (name,), value)
