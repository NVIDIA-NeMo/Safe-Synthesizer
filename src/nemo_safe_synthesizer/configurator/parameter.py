# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic parameter wrapper for Pydantic-based configuration.

Provides ``Parameter`` -- a generic dataclass wrapper that integrates with
Pydantic v2 core schemas so that configuration values carry metadata
(e.g. ``name``) while remaining transparent to serialization and comparison.
"""

import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast, get_args

from pydantic import BaseModel, GetCoreSchemaHandler, model_serializer
from pydantic_core import core_schema

DataT = TypeVar(
    "DataT", bound=(int | float | str | bytes | bool | None | Sequence[int | float | str | bytes | bool | BaseModel])
)

ParameterT = TypeVar("ParameterT", bound="Parameter")


@dataclass(eq=False, order=False)
class Parameter(Generic[DataT]):
    """Generic wrapper for a single configuration value.

    Wraps a primitive or sequence value so it can carry a ``name``, participate
    in Pydantic validation/serialization, and support rich comparisons against
    both raw values and other ``Parameter`` instances.

    Args:
        name: Identifier used for logging and lookup (e.g. ``"holdout"``).
        value: The wrapped configuration value.

    Example::

        >>> param = Parameter[int](name="max_size", value=100)
        >>> param.value
        100
        >>> param == 100
        True
    """

    name: str | None = None
    value: DataT | Sequence[DataT] | None = None

    @model_serializer
    def ser_model(self) -> "dict[str, DataT] | DataT | Sequence[DataT] | Parameter[DataT] | None":
        """Serialize to the bare value for Pydantic ``model_dump`` / ``model_dump_json``."""
        if hasattr(self, "value"):
            return self.value
        else:
            return self

    def __str__(self):
        return self.__repr__()

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Build a Pydantic core schema that accepts both ``Parameter`` instances and raw values.

        The resulting union schema lets Pydantic validate a field annotated as
        ``Parameter[int]`` from either an already-constructed ``Parameter`` or a
        plain ``int`` (which is wrapped automatically).

        Args:
            source: The source type annotation (e.g. ``Parameter[int]``).
            handler: Pydantic's schema generation handler.

        Returns:
            A union core schema covering instance and raw-value branches.
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

    def _comp_helper(self, other: "Parameter[DataT] | DataT", op: Callable[[Any, Any], bool]) -> bool | None:
        """Apply a comparison operator ``op`` to ``self.value`` and the unwrapped value of ``other``."""
        match other:
            case Parameter(value=y) if isinstance(self.value, type(y)):
                return op(self.value, y)
            case y if isinstance(self.value, type(y)):
                return op(self.value, y)
            case _:
                return NotImplemented

    def __ge__(self, other: "Parameter[DataT] | DataT") -> bool | None:
        return self._comp_helper(other, operator.__ge__)

    def __le__(self, other: "Parameter[DataT] | DataT") -> bool | None:
        return self._comp_helper(other, operator.__le__)

    def __gt__(self, other: "Parameter[DataT] | DataT") -> bool | None:
        return self._comp_helper(other, operator.__gt__)

    def __lt__(self, other: "Parameter[DataT] | DataT") -> bool | None:
        return self._comp_helper(other, operator.__lt__)

    def __eq__(self, other: object) -> bool:
        result = self._comp_helper(cast("Parameter[DataT] | DataT", other), operator.__eq__)
        return cast(bool, result)
