# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Generate Click CLI options from a Pydantic model.

Used by ``cli/run.py`` and ``cli/config.py`` to expose every
``SafeSynthesizerParameters`` field as a ``--field_name`` CLI option.
Nested ``BaseModel`` fields are flattened with a separator
(e.g. ``--data__holdout``).

The companion ``parse_overrides()`` reverses the flattening at runtime,
converting Click's flat ``{key: value}`` dict back into the nested structure
Pydantic expects.  The ``field_sep`` argument to ``parse_overrides`` must
match the ``field_separator`` passed to ``pydantic_options``; otherwise
nested keys like ``data__holdout`` will not be reconstructed correctly.
"""

import inspect
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import click
from pydantic import BaseModel

__all__ = ["pydantic_options", "parse_overrides"]


def parse_overrides(values: dict[str, Any] | None = None, field_sep: str = "__") -> dict[str, Any]:
    """Convert flat CLI overrides into a nested dict suitable for Pydantic.

    Splits each key on ``field_sep`` to reconstruct nesting.  For example,
    ``{"data__holdout": 0.1}`` becomes ``{"data": {"holdout": 0.1}}``.

    Args:
        values: Flat dict of CLI arguments from Click (``None``-valued keys
            are skipped).
        field_sep: Separator used by ``pydantic_options`` to flatten nested
            field names.

    Returns:
        A nested dict ready to be passed to ``Parameters.from_yaml_or_overrides``
        or ``model_copy(update=...)``.

    Raises:
        ValueError: If a key contains more than one separator (only one level
            of nesting is supported).
    """
    if values is None:
        return {}
    overrides = {}
    for k, v in values.items():
        if v is not None:
            match k.split(field_sep):
                # e.g., --enable_synthesis - top level value with no nesting
                case [k]:
                    overrides[k] = v
                # e.g., --enable_replace_pii or --data__group_training_examples_by
                # would have
                case [key, suffix]:
                    # we don't want to overwrite existing nested dicts
                    if key in overrides:
                        overrides[key][suffix] = v
                    else:
                        overrides[key] = {suffix: v}
                case _:
                    raise ValueError(f"Invalid override: {k}")

    return overrides


def pydantic_options(model_class: type[BaseModel], field_separator: str = "__"):
    """Click decorator that adds a ``--option`` for every field in ``model_class``.

    Recursively walks ``model_class.model_fields``, flattening nested
    ``BaseModel`` subfields with ``field_separator``.  Field types are
    mapped to Click types (``INT``, ``FLOAT``, ``BOOL``, or ``str``).
    Help text is pulled from ``Field(description=...)``.

    Args:
        model_class: The Pydantic model to generate options from
            (typically ``SafeSynthesizerParameters``).
        field_separator: String used to join parent and child field names
            in the CLI option (default ``"__"``).

    Returns:
        A Click decorator that attaches the generated options to a command.
    """

    def get_fields(cls: type[BaseModel], prefix=""):
        fields = []
        for name, field in cls.model_fields.items():
            field_type = field.annotation
            full_name = f"{prefix}{name}" if prefix else name

            # Handle nested BaseModel
            if inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                fields.extend(get_fields(field_type, f"{full_name}."))

            elif get_origin(field_type) is Annotated:
                base_type = get_args(field_type)[0]

                if inspect.isclass(base_type) and issubclass(base_type, BaseModel):
                    fields.extend(get_fields(base_type, f"{full_name}."))

                else:
                    fields.append((full_name, field))

            # union types are strange
            elif get_origin(field_type) is types.UnionType:
                union_args = get_args(field_type)

                for arg in union_args:
                    # Skip None type in the union (for Optional fields)
                    if arg is type(None):
                        continue

                    if inspect.isclass(arg) and issubclass(arg, BaseModel):
                        fields.extend(get_fields(arg, f"{full_name}."))

                else:
                    # If no BaseModel was found in the union, treat it as a regular field
                    fields.append((full_name, field))

            else:
                fields.append((full_name, field))
        return sorted(fields, key=lambda x: x[0])

    def decorator(f):
        for name, field in get_fields(model_class):
            param_type = field.annotation
            if get_origin(param_type) is Annotated:
                param_type = get_args(param_type)[0]
            elif get_origin(param_type) is Union:
                param_type = get_args(param_type)[0]
            else:
                param_type = param_type

            if param_type in (int, Literal["auto"] | int):
                click_type = click.INT
            elif param_type in (float, Literal["auto"] | float):
                click_type = click.FLOAT
            elif param_type is bool or param_type == Literal["auto"] | bool:
                click_type = click.BOOL
            else:
                click_type = str

            option_name = f"--{name.replace('.', field_separator)}"
            # click tries to assign the passed value to a variable with the same name, so we need to rename
            # it if it has dots in the name.
            # the name and option name are passed as *args to click, so we pack either into a tuple to unpack correctly.
            if field_separator == ".":
                option_name = option_name, name.replace(".", "_")
            else:
                option_name = tuple([option_name])
            help_text = field.description if hasattr(field, "description") else ""
            f = click.option(*option_name, type=click_type, help=help_text)(f)

        return f

    return decorator
