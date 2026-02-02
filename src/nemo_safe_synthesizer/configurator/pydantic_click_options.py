# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Generate Click options from a Pydantic model.
This will recursively add options for nested models to the top level command
"""

import inspect
import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import click
from pydantic import BaseModel

__all__ = ["pydantic_options", "parse_overrides"]


def parse_overrides(values: dict[str, Any] | None = None, field_sep: str = "__") -> dict[str, Any]:
    """Parse Click command line overrides into a nested dictionary.
    Args:
        values: Dictionary of command line arguments from Click.
        field_sep: Separator used in command line arguments to denote nesting.
    Returns:
        A nested dictionary of overrides.
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
    """Generate Click options from a Pydantic model."""

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
