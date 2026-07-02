# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate Click CLI options from a Pydantic model.

Used by ``cli/run.py`` and ``cli/config.py`` to expose every
``SafeSynthesizerParameters`` field as a ``--field_name`` CLI option.
Nested ``BaseModel`` fields are flattened with a separator
(e.g. ``--data__holdout``).  Fields typed as ``SomeModel | None`` also
get a ``--no-<field>`` is-flag that sets the field to ``None``.

The companion ``parse_overrides()`` reverses the flattening at runtime,
converting Click's flat ``{key: value}`` dict back into the nested structure
Pydantic expects.  The ``field_sep`` argument to ``parse_overrides`` must
match the ``field_separator`` passed to ``pydantic_options``; otherwise
nested keys like ``data__holdout`` will not be reconstructed correctly.
"""

from __future__ import annotations

import inspect
import types
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import click
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from typing_extensions import TypeIs

from ..config.types import AUTO_STR
from .parameter_paths import insert_parameter_value, split_parameter_path

__all__ = ["pydantic_options", "parse_overrides", "AutoParamType"]

_NEGATION_PREFIX = "no_"
"""Prefix marking a generated disable flag for a nullable sub-config field."""

_LEGACY_CLI_OPTION_PATHS: dict[str, tuple[str, ...]] = {
    "generation.structured_generation.enabled": ("generation.use_structured_generation",),
    "generation.structured_generation.backend": ("generation.structured_generation_backend",),
    "generation.structured_generation.schema_method": ("generation.structured_generation_schema_method",),
    "generation.structured_generation.use_single_sequence": ("generation.structured_generation_use_single_sequence",),
}
"""Hidden compatibility aliases for renamed generated CLI options."""


def parse_overrides(values: dict[str, Any] | None = None, field_sep: str = "__") -> dict[str, Any]:
    """Parse Click kwargs into a nested override dict.

    ``no_<field>=True`` injects ``{field: None}`` to disable a nullable-model
    field.  ``no_<field>=False`` (unset is-flag) is silently dropped.
    ``None`` values (unset regular options) are also dropped.

    Args:
        values: Flat dictionary of command line arguments from Click. (``None``-valued keys are dropped).
        field_sep: Separator used to reconstruct nesting.  For example, ``{"data__holdout": 0.1}`` becomes ``{"data": {"holdout": 0.1}}``.

    Returns:
        A nested dictionary suitable for schema-aware config patching or direct
        model validation.

    Raises:
        ValueError: If a key contains empty segments (e.g. consecutive
            separators like ``a____b``), or parent and child overrides target
            incompatible values.
    """
    if not values:
        return {}
    overrides: dict[str, Any] = {}
    for k, v in values.items():
        if k.startswith(_NEGATION_PREFIX) and isinstance(v, bool):
            if v:
                insert_parameter_value(
                    overrides, split_parameter_path(k.removeprefix(_NEGATION_PREFIX), field_sep), None
                )
            continue
        if v is None:
            continue
        try:
            path = split_parameter_path(k, field_sep)
        except ValueError as error:
            raise ValueError(f"Invalid override key: {k!r}") from error
        insert_parameter_value(overrides, path, v)
    return overrides


# ---------------------------------------------------------------------------
# Param variants
# ---------------------------------------------------------------------------


@dataclass
class LeafParam:
    """A scalar CLI option backed by a Pydantic FieldInfo."""

    name: str
    field: FieldInfo


@dataclass
class FlagParam:
    """A ``--no-<field>`` is-flag that sets the named field to ``None``."""

    name: str  # internal key for sort ordering, e.g. "no_replace_pii" (CLI: --no-replace-pii)
    field_name: str  # field being disabled, e.g. "replace_pii"


ClickParam = LeafParam | FlagParam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_basemodel(t: Any) -> TypeIs[type[BaseModel]]:
    return inspect.isclass(t) and issubclass(t, BaseModel)


def _nullable_model_arg(union_args: tuple) -> type[BaseModel] | None:
    """Return the BaseModel member of a ``SomeModel | None`` union, or ``None``."""
    return next((a for a in union_args if a is not type(None) and _is_basemodel(a)), None)


# Click types ordered from widest to narrowest acceptance. When a union
# contains multiple scalar types (e.g. ``str | int``), the widest type
# that won't reject valid input is chosen. Click validates *before*
# Pydantic, so a narrow Click type (INT) would reject values that
# Pydantic could accept (a date string for ``str | int``).
_CLICK_TYPE_PRIORITY: list[tuple[type, click.ParamType]] = [
    (str, click.STRING),
    (float, click.FLOAT),
    (int, click.INT),
    (bool, click.BOOL),
]


class AutoParamType(click.ParamType):
    """A Click type that accepts the sentinel ``AUTO_STR`` or a base numeric/bool value.

    Used for ``Auto*Param`` fields (``AutoIntParam``, ``AutoFloatParam``,
    ``AutoBoolParam``) so that ``--flag auto`` and ``--flag 2`` both work.
    The ``--help`` display shows ``integer|auto``, ``float|auto``, or
    ``boolean|auto`` instead of the generic ``TEXT`` label.

    Args:
        base_type: The underlying Click type (``click.INT``, ``click.FLOAT``,
            or ``click.BOOL``) used to parse and validate non-``AUTO_STR`` values.
    """

    def __init__(self, base_type: click.ParamType) -> None:
        self.base_type = base_type
        self.name = f"{base_type.name}|{AUTO_STR}"

    def convert(
        self,
        value: str,
        param: click.Parameter | None,
        ctx: click.Context | None,
    ) -> str | int | float | bool:
        """Convert the raw CLI value to ``AUTO_STR`` or the base numeric/bool type.

        The ``value`` parameter is typed ``str`` to match the parent
        ``click.ParamType.convert`` stub signature; in practice Click may also
        pass through default values of any type, but the equality check and
        delegated ``base_type.convert`` both handle that correctly.

        Args:
            value: Raw value from the CLI or the option default.
            param: The Click parameter object (passed through to the base type).
            ctx: The Click context (passed through to the base type).

        Returns:
            ``AUTO_STR`` if ``value`` equals it, otherwise the result of
            delegating to ``self.base_type.convert()``.
        """
        if value == AUTO_STR:
            return AUTO_STR
        return self.base_type.convert(value, param, ctx)


def _has_string_literal(args: set) -> bool:
    """Check if any member is a ``Literal`` containing a string value."""
    return any(get_origin(a) is Literal and any(isinstance(v, str) for v in get_args(a)) for a in args)


def _is_auto_only_literal_union(args: set) -> bool:
    """Check that the union's string-valued ``Literal`` members are exactly ``{AUTO_STR}``.

    Returns ``True`` only if every string-valued Literal member contributes the
    single value ``AUTO_STR`` -- e.g. ``Literal["auto"] | int``. Returns
    ``False`` for unions like ``Literal["disabled"] | int`` or
    ``Literal["auto", "manual"] | int``, where ``AutoParamType`` would
    silently reject the non-``"auto"`` sentinels at parse time.
    """
    string_values: set[str] = set()
    for a in args:
        if get_origin(a) is Literal:
            for v in get_args(a):
                if isinstance(v, str):
                    string_values.add(v)
    return string_values == {AUTO_STR}


def _literal_value_types(args: set) -> set:
    """Replace non-string ``Literal`` annotations with their value types."""
    normalized: set = set()
    for arg in args:
        if get_origin(arg) is Literal:
            normalized.update(type(value) for value in get_args(arg))
        else:
            normalized.add(arg)
    return normalized


def _click_type(annotation: Any) -> click.ParamType:
    """Map a Pydantic field annotation to a Click type.

    Unwraps ``Annotated[T, ...]`` and ``T | None`` unions, then returns the
    widest Click type that covers any member of the union. ``Auto*Param``
    fields (``Literal["auto"] | <numeric|bool>``) get an ``AutoParamType``
    wrapping the numeric/bool base so Click can validate non-sentinel values
    while still accepting the ``"auto"`` sentinel. Other string-valued
    ``Literal`` members fall through to ``click.STRING`` so Click won't
    reject the sentinel before Pydantic validates it. Falls back to
    ``click.STRING`` for unrecognized types.
    """
    t = annotation
    if get_origin(t) is Annotated:
        t = get_args(t)[0]
    args = set(get_args(t)) if get_origin(t) in (Union, types.UnionType) else {t}
    args.discard(type(None))
    if _has_string_literal(args):
        # Auto*Param: Literal["auto"] | <numeric|bool> -- wrap in AutoParamType so
        # --help shows "integer|auto" instead of "TEXT" and Click validates the
        # numeric side before handing the value to Pydantic. The detection is
        # tightened to only fire when the literal values are exactly {AUTO_STR};
        # other sentinels (e.g. Literal["disabled"] | int) fall through to STRING
        # so AutoParamType doesn't reject them with a confusing error.
        if _is_auto_only_literal_union(args):
            for py_type, click_type in _CLICK_TYPE_PRIORITY:
                if py_type is str:
                    continue
                if py_type in args:
                    return AutoParamType(click_type)
        return click.STRING
    args = _literal_value_types(args)
    for py_type, click_type in _CLICK_TYPE_PRIORITY:
        if py_type in args:
            return click_type
    return click.STRING


def _option_names(name: str, field_separator: str) -> tuple[str, ...]:
    """Build the Click ``*names`` tuple for a given logical field name."""
    cli = f"--{name.replace('.', field_separator)}"
    if field_separator == ".":
        return cli, name.replace(".", "_")
    return (cli,)


def _collect_params(cls: type[BaseModel], prefix: str = "") -> list[ClickParam]:
    """Recursively collect CLI params from a Pydantic model.

    Returns an unsorted list -- callers are responsible for sorting.
    """
    params: list[ClickParam] = []
    for name, field in cls.model_fields.items():
        full = f"{prefix}{name}"
        ft = field.annotation

        # Unwrap Annotated[T, ...] to its inner type.
        inner = get_args(ft)[0] if get_origin(ft) is Annotated else ft

        match inner:
            case t if _is_basemodel(t):
                params.extend(_collect_params(t, f"{full}."))
            case t if get_origin(t) is types.UnionType:
                model_arg = _nullable_model_arg(get_args(t))
                if model_arg is not None:
                    params.extend(_collect_params(model_arg, f"{full}."))
                    params.append(FlagParam(f"{_NEGATION_PREFIX}{full}", full))
                else:
                    params.append(LeafParam(full, field))
            case _:
                params.append(LeafParam(full, field))

    return params


# ---------------------------------------------------------------------------
# Public decorator
# ---------------------------------------------------------------------------


def pydantic_options(model_class: type[BaseModel], field_separator: str = "__"):
    """Decorate a Click command with options derived from a Pydantic model.

    Recurses into nested sub-models, flattening their fields into top-level
    CLI options separated by ``field_separator``.  Fields typed as
    ``SomeModel | None`` also get a ``--no-<field>`` is-flag that sets the
    field to ``None`` when passed.  Field types are mapped to Click types
    via ``_CLICK_TYPE_PRIORITY``; help text is pulled from
    ``Field(description=...)``.

    Args:
        model_class: The Pydantic model to generate options from
            (typically ``SafeSynthesizerParameters``).
        field_separator: String used to join parent and child field names
            in the CLI option (default ``"__"``).

    Returns:
        A Click decorator that attaches the generated options to a command.
    """

    def apply_leaf_option(f, name: str, field: FieldInfo, *, hidden: bool = False):
        names = _option_names(name, field_separator)
        return click.option(
            *names,
            type=_click_type(field.annotation),
            help=field.description or "",
            hidden=hidden,
        )(f)

    def decorator(f):
        for param in sorted(_collect_params(model_class), key=lambda p: p.name):
            match param:
                case FlagParam(field_name=field_name):
                    # Flags use standard CLI dashes (--no-replace-pii) while
                    # LeafParam options preserve underscores (--training__learning_rate)
                    # because the separator/field structure is autogenerated from Pydantic.
                    nested_name = field_name.replace(".", field_separator)
                    if field_separator != ".":
                        parts = nested_name.split(field_separator)
                        nested_name = field_separator.join(p.replace("_", "-") for p in parts)
                    else:
                        nested_name = nested_name.replace("_", "-")
                    flag_cli = f"--no-{nested_name}"
                    f = click.option(
                        flag_cli,
                        is_flag=True,
                        default=False,
                        help=f"Disable {field_name.replace('_', '-')} entirely.",
                    )(f)
                case LeafParam(field=field):
                    f = apply_leaf_option(f, param.name, field)
                    for legacy_name in _LEGACY_CLI_OPTION_PATHS.get(param.name, ()):
                        f = apply_leaf_option(f, legacy_name, field, hidden=True)
        return f

    return decorator
