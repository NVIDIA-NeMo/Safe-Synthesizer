# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import (
    GetCoreSchemaHandler,
    ValidationInfo,
)
from pydantic_core import core_schema

from ..observability import get_logger
from .parameter import Parameter

__all__ = ["DependsOnValidator", "ValueValidator", "AutoParamRangeValidator"]


logger = get_logger(__name__)


# The frozen=True specification makes DependsOnValidator hashable.
# Without this, a union on the custom type such as X | None will raise an error.
@dataclass(frozen=True)
class DependsOnValidator:
    """
    Validator for creating conditional field dependencies in Pydantic models.

    This validator allows you to define fields that are only valid when another
    field meets specific conditions. Useful for creating configuration schemas
    where certain options are only available based on other settings.

    Attributes:
        depends_on: Name of the field this validation depends on
        depends_on_func: Function that validates the dependency field's value
        value_func: Optional function to validate the current field's value

    Example:
        >>> validator = DependsOnValidator(
        ...     depends_on="enabled",  # a field
        ...     depends_on_func=lambda x: x is True,
        ...     value_func=lambda x: isinstance(x, bool)
        ... )
    """

    depends_on: str
    depends_on_func: Callable[[Any], bool]
    value_func: Callable[[Any], bool] | None

    def validate(self, value, info: ValidationInfo):
        """
        Validate the field value based on dependency conditions. This is a pydantic construction.

        Args:
            value: The value being validated
            info: Pydantic validation context containing field information

        Returns:
            The validated value if all conditions pass

        Raises:
            ValueError: If dependency field is missing or conditions aren't met
        """
        if self.depends_on not in info.data:
            raise ValueError(f"{info.field_name} is only allowed in model with {self.depends_on}")

        vf = self.value_func if self.value_func is not None else lambda x: x

        if vf(value):
            if self.depends_on_func(info.data.get(self.depends_on)):
                return value
            else:
                raise ValueError(
                    f"{info.field_name} is only allowed when {self.depends_on} pass condition \
                    `{inspect.getsource(self.depends_on_func)}`"
                )

        return value

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """
        Generate Pydantic core schema for this validator.

        Args:
            source_type: The source type annotation
            handler: Pydantic's schema generation handler

        Returns:
            A core schema that applies the dependency validation
        """
        return core_schema.with_info_after_validator_function(self.validate, handler(source_type))


@dataclass(frozen=True)
class ValueValidator:
    """
    Custom validator for applying validation functions to Parameter values.

    This validator allows you to define custom validation logic for Parameter
    instances by providing a function that examines the parameter's value.

    Attributes:
        value_func: Function that takes a Parameter value and returns True if valid

    Example:
        >>> # Validate that a numeric parameter is positive
        >>> validator = ValueValidator(
        ...     value_func=lambda x: x > 0 if isinstance(x, (int, float)) else True
        ... )
    """

    value_func: Callable[[Parameter[Any]], bool]

    def validate(self, value, info: ValidationInfo):
        """
        Validate a Parameter using the provided validation function.
        """
        if (v := getattr(value, "value", None)) is None:
            real_val = value
        else:
            real_val = v
        logger.debug(f"Parameter {info.field_name}, {value}, passed validation: {real_val}")
        if self.value_func(real_val):
            return value
        else:
            try:
                # Sometimes the inspect.getsource() function raises an OSError
                # for lambda functions defined inline, hence the try/except.
                src = inspect.getsource(self.value_func)
                msg = f"Parameter {info.field_name}, {real_val}, did not pass validation: {src}"
            except OSError:
                msg = f"Parameter {info.field_name}, {value}, did not pass validation"
            raise ValueError(msg)

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """
        Generate Pydantic core schema for this validator.

        Args:
            source_type: The source type annotation
            handler: Pydantic's schema generation handler

        Returns:
            A core schema that applies the value validation
        """
        return core_schema.with_info_after_validator_function(self.validate, handler(source_type))


def range_validator(value: int | float, func: Callable) -> bool:
    """
    Validate that a numeric parameter falls within a specified range.

    This utility function provides range validation for Parameter instances
    containing numeric values, with graceful handling of non-numeric types.

    Args:
        value: The Parameter instance to validate
        func: A function that takes a numeric value and returns True if valid

    Returns:
        True if validation passes or if the value is not numeric
    """
    return True if value == "auto" else func(value)


AutoParamRangeValidator = ValueValidator(lambda p: range_validator(p, lambda v: v >= 0))
