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
from typing import Annotated, Any, Union, get_args, get_origin

import click
from pydantic import BaseModel
from pydantic.fields import FieldInfo

__all__ = ["pydantic_options", "parse_overrides"]


def parse_overrides(values: dict[str, Any] | None = None, field_sep: str = "__") -> dict[str, Any]:
    """Parse Click kwargs into a nested override dict.

    ``no_<field>=True`` injects ``{field: None}`` to disable a nullable-model
    field.  ``no_<field>=False`` (unset is-flag) is silently dropped.
    ``None`` values (unset regular options) are also dropped.

    Args:
        values: Flat dictionary of command line arguments from Click. (``None``-valued keys are dropped).
        field_sep: Separator used to reconstruct nesting.  For example, ``{"data__holdout": 0.1}`` becomes ``{"data": {"holdout": 0.1}}``.

    Returns:
        A nested dict suitable for ``model_validate()`` or for merging
        with a loaded config via ``merge_dicts()``.

    Raises:
        ValueError: If a key contains empty segments (e.g. consecutive
            separators like ``a____b``).
    """
    if not values:
        return {}
    overrides: dict[str, Any] = {}
    for k, v in values.items():
        if k.startswith("no_") and isinstance(v, bool):
            if v:
                overrides[k[3:]] = None
            continue
        if v is None:
            continue
        match k.split(field_sep):
            case [key]:
                overrides[key] = v
            case [first, *rest, last] if all(rest) and last:
                target = overrides
                target = target.setdefault(first, {})
                for part in rest:
                    target = target.setdefault(part, {})
                target[last] = v
            case _:
                raise ValueError(f"Invalid override key: {k!r}")
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

    name: str  # CLI param name, e.g. "no_replace_pii"
    field_name: str  # field being disabled, e.g. "replace_pii"


ClickParam = LeafParam | FlagParam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_basemodel(t: Any) -> bool:
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


def _click_type(annotation: Any) -> click.ParamType:
    """Map a Pydantic field annotation to a Click type.

    Unwraps ``Annotated[T, ...]`` and ``T | None`` unions, then returns the
    widest Click type that covers any member of the union. Falls back to
    ``click.STRING`` for unrecognized types.
    """
    t = annotation
    if get_origin(t) is Annotated:
        t = get_args(t)[0]
    args = set(get_args(t)) if get_origin(t) in (Union, types.UnionType) else {t}
    args.discard(type(None))
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
                    params.append(FlagParam(f"no_{full}", full))
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

    def decorator(f):
        for param in sorted(_collect_params(model_class), key=lambda p: p.name):
            names = _option_names(param.name, field_separator)
            match param:
                case FlagParam(field_name=field_name):
                    f = click.option(
                        *names,
                        is_flag=True,
                        default=False,
                        help=f"Disable {field_name.replace('_', '-')} entirely.",
                    )(f)
                case LeafParam(field=field):
                    f = click.option(
                        *names,
                        type=_click_type(field.annotation),
                        help=field.description or "",
                    )(f)
        return f

    return decorator
