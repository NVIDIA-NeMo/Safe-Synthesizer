# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter wrapper classes for Pydantic-based configuration systems.

This module provides generic parameter wrappers and mixins that enable strongly-typed
configuration parameters with automatic validation, serialization, and schema generation
for Pydantic v2 models.
"""

import operator
import typing
from dataclasses import dataclass
from typing import Any, Callable, Generic, Sequence, Type, TypeVar, get_args

from pydantic import BaseModel, GetCoreSchemaHandler, model_serializer
from pydantic_core import core_schema

from ..observability import get_logger

DataT = TypeVar(
    "DataT", bound=(int | float | str | bytes | bool | None | Sequence[int | float | str | bytes | bool | BaseModel])
)

logger = get_logger(__name__)

ParameterT = TypeVar("ParameterT", bound="Parameter")


@dataclass(eq=False, order=False)
class Parameter(Generic[DataT]):
    """
    Generic wrapper for configuration parameters with Pydantic integration.

    This class wraps configuration values to provide type safety, serialization,
    and schema generation for Pydantic models. It supports both single values
    and sequences, with automatic comparison operations.

    Attributes:
        name: Optional name identifier for the parameter
        value: The parameter value, can be a single value or sequence

    Example:
        >>> param = Parameter[int](name="max_size", value=100)
        >>> param.value
        100
        >>> param == 100
        True
    """

    name: str | None = None
    value: DataT | Sequence[DataT] | None = None

    @model_serializer
    def ser_model(self) -> dict[str, DataT] | DataT | Sequence[DataT] | None:
        """
        Serialize the parameter for Pydantic model serialization.

        Returns:
            The parameter value if it exists, otherwise the parameter instance itself.
        """
        if hasattr(self, "value"):
            return self.value
        else:
            return self

    def __str__(self):
        """Return string representation of the parameter."""
        return self.__repr__()

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """
        Generate Pydantic core schema for this parameter type.

        This method enables Pydantic to properly validate and serialize Parameter
        instances by creating a union schema that accepts both Parameter instances
        and raw values that can be converted to Parameters.

        Args:
            source: The source type annotation
            handler: Pydantic's schema generation handler

        Returns:
            A core schema that handles both Parameter instances and convertible values
        """
        instance_schema = core_schema.is_instance_schema(cls)

        args = get_args(source)
        if args:
            # replace the type and rely on Pydantic to generate the right schema
            # for our data types
            sequence_t_schema = handler.generate_schema(DataT[args[0]])  # type: ignore
        else:
            sequence_t_schema = handler.generate_schema(DataT)

        non_instance_schema = core_schema.no_info_before_validator_function(cls, sequence_t_schema)
        return core_schema.union_schema([instance_schema, non_instance_schema])

    def _comp_helper(self, other: object, op: Callable[[Any, Any], bool]) -> bool:
        """
        Helper method for comparison operations between parameters and values.

        Args:
            other: Another Parameter instance or raw value to compare against
            op: The comparison operator function to apply

        Returns:
            Result of the comparison operation, or NotImplemented for unsupported types
        """
        match other:
            case Parameter(value=y) if isinstance(self.value, type(y)):
                return op(self.value, y)
            case y if isinstance(self.value, type(y)):
                return op(self.value, y)
            case _:
                return NotImplemented

    def __ge__(self, other: "Parameter[DataT] | DataT"):
        self._comp_helper(other, operator.__ge__)

    def __le__(self, other: "Parameter[DataT] | DataT"):
        self._comp_helper(other, operator.__le__)

    def __gt__(self, other: "Parameter[DataT] | DataT") -> bool:
        return self._comp_helper(other, operator.__gt__)

    def __lt__(self, other: "Parameter[DataT] | DataT") -> bool:
        return self._comp_helper(other, operator.__lt__)

    def __eq__(self, other: object) -> bool:
        return self._comp_helper(other, operator.__eq__)


class AutoMixin(Generic[DataT]):
    """
    Mixin providing automatic type-safe value updates for parameters.

    This mixin adds functionality to safely update parameter values while
    maintaining type consistency and providing warnings for potentially
    problematic type conversions.
    """

    def autoupdate(self, new_val: DataT) -> None:
        """
        Update the parameter value with type checking and conversion logic.

        This method performs safe type updates, allowing compatible conversions
        (like int to float) while preventing incompatible type changes.
        Normally, this is used when a parameter is set to "auto", and otherwise will perform checks before updating.


        Args:
            new_val: The new value to assign to the parameter

        Raises:
            TypeError: If the new value type is incompatible with the current type
            ValueError: If the value types don't match the expected pattern
            AttributeError: If the object doesn't have a 'value' attribute

        Example:
            >>> param = AutoParam[int](value=10)
            >>> param.autoupdate(20)  # OK
            >>> param.autoupdate(20.5)  # OK with warning (int to float)
        """
        str_float = (str, float)
        if hasattr(self, "value"):
            match new_val:
                case Parameter(value=y) if isinstance(self.value, type(y)):
                    self.value = y
                case y if isinstance(self.value, type(y)):
                    self.value = y
                case y if isinstance(self.value, str_float) and isinstance(y, str_float):
                    self.value = y
                case None:
                    self.value = None
                case _:
                    return NotImplemented

            self.updated_ = True
        else:
            raise AttributeError("AutoParameter must have a 'value' attribute to autoupdate.")

    def autoupdate_pipe(self, f: Callable[[Any], DataT]) -> None:
        """
        Update the parameter value by applying a transformation function.

        Args:
            f: Function that takes the current value and returns a new value

        Example:
            >>> param = AutoParam(value=10)
            >>> param.autoupdate_pipe(lambda x: x * 2)  # value becomes 20
        """
        self.autoupdate(f(getattr(self, "value")))


@dataclass(eq=False, order=False)
class AutoParam(AutoMixin, Parameter, Generic[DataT]):
    """
    Parameter class with automatic update capabilities.

    Attributes:
        updated_: Flag indicating whether the parameter has been updated
    """

    updated_: bool = False


class UnsetMixin(Generic[DataT]):
    """
    Mixin for parameters that can represent unset or default states.

    This mixin is used to mark parameters that haven't been explicitly set
    and may use default values or require special handling.
    """

    ...


@dataclass(eq=False, order=False)
class UnsetParam(UnsetMixin, Parameter, Generic[DataT]):
    """
    Parameter variant for handling unset or default values.

    This class is used when a parameter hasn't been explicitly set and
    should be treated differently from parameters with None values.

    Attributes:
        name: Optional name identifier for the parameter
        value: The parameter value, can include AutoParam instances
    """

    name: str | None = None
    value: DataT | AutoParam[DataT] | None = None


class ValidNoneMixin:
    """
    Mixin for parameters that explicitly allow None as a valid value.

    This distinguishes between unset parameters and parameters that are
    intentionally set to None.
    """

    name: str | None = None
    value: None = None

    def __eq__(self, other: Any) -> bool:
        if other is None:
            return True
        elif isinstance(other, ValidNoneParam):
            return True
        else:
            return False


@dataclass(eq=False, order=False)
class ValidNoneParam(ValidNoneMixin, Parameter, Generic[DataT]):
    """
    Parameter variant that explicitly allows None as a valid value.

    Use this when None is a meaningful value for your parameter,
    distinct from an unset parameter.
    """

    name: str | None = None
    value: None = None


ValidNoneType = TypeVar("ValidNoneType", ValidNoneParam, None)


def _convert_val_type_to_param(
    value: Any,
    type_: Type,
    name: str | None = None,
) -> Parameter[DataT]:
    """
    Convert a raw value to an appropriate Parameter instance based on type and value.

    This utility function handles the conversion of various value types into
    the appropriate Parameter subclass, with special handling for unset values,
    None values, and different data types.

    Args:
        value: The raw value to convert
        type_: The target Parameter type to create
        name: Optional name for the parameter

    Returns:
        An appropriate Parameter instance wrapping the value

    Example:
        >>> param = _convert_val_type_to_param(42, AutoParam, "max_size")
        >>> isinstance(param, AutoParam)
        True
    """
    logger.debug(f"Converting value: {value} of type {type_} for field {name}")
    match value, type_:
        case "_unset_", _:
            msg = f"_unset_, {type_}: UnsetParam[{type_}]"
            value = UnsetParam(name=name)

        case None, AutoParam():
            msg = f"None, AutoParam(): {type}(None)"
            value = type_(value=None)

        case None | ValidNoneParam(), ValidNoneParam():
            msg = "None | ValidNoneParam(), ValidNoneParam(): ValidNoneParam(None)"
            value = ValidNoneParam(name=name)

        case None, _:
            msg = "None, _: AutoParam(None)"
            value = AutoParam(name=name, value=None)

        case list() as list_val, t:
            # Check if list contains only primitives (DataT types)
            if list_val and all(isinstance(item, (int, float, str, bytes, bool, type(None))) for item in list_val):
                msg = "list() with primitives, _: type_(name=name, value=list_val)"
                value = type_(name=name, value=list_val)
            else:
                # For nested/complex objects, pass through without conversion
                msg = "list() with complex objects, _: pass through"
                value = list_val

        case str() as s, t if s == "auto":
            return AutoParam(name=name, value=value)

        case str() | int() | float() | bool(), t:
            # Handle Union types by finding the first compatible type
            if getattr(t, "__origin__", None) is typing.Union:
                # first of  the union, e.g., Parameter[int] | AutoParam[int], this is the first
                for arg in get_args(type_):
                    # this is the subtype of the arg, the generic
                    logger.debug(f"Checking arg: {arg} for value: {value} of type {type(value)}")
                    for subtype in get_args(arg):
                        if isinstance(value, subtype):
                            msg = f"Converting value: {value} to type: {arg} for field {name}"
                            return arg(value=value, name=name)
            else:
                return type_(value=value, name=name)

        case None, t:
            msg = f"DataT, {type}(value)"
            if getattr(t, "__origin__", None) is typing.Union:
                if ValidNoneParam in get_args(type_):
                    msg = f"DataT, Union: ValidNoneParam({value})"
                    return ValidNoneParam(name=name)
                else:
                    msg = f"DataT, Union: AutoParam({value})"
                    return Parameter(name=name, value=None)

        case (_, AutoParam()):
            msg = f"_, AutoParam(): AutoParam({value.value})"
            value = AutoParam(name=name, value=value.value)

        case (AutoParam(), _):
            msg = f"AutoParam(), _: AutoParam({value.value})"
            value = value

        case Parameter(), _:
            msg = f"Parameter(), _: {type}(value)"
            value = value

        case _, _:
            msg = "_, _: No change"
            pass

    logger.debug(msg)

    if hasattr(value, "name") and (value.name is None) and (name is not None):
        logger.debug(f"setting name of parameter {value} to {name}")
        value.name = name

    return value
