# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic-integrated field validators for parameter models.

Provides two validator types used as ``Annotated`` metadata on fields in
``config/`` models:

- ``DependsOnValidator`` -- enforces conditional dependencies between fields
  (e.g. ``order_training_examples_by`` requires ``group_training_examples_by``).
- ``ValueValidator`` -- applies an arbitrary predicate to the field value
  (e.g. "learning rate must be in ``(0, 1)``").

Both implement ``__get_pydantic_core_schema__`` so Pydantic runs them as
after-validators during model construction.
"""

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

__all__ = ["DependsOnValidator", "ValueValidator", "AutoParamRangeValidator"]


logger = get_logger(__name__)


# frozen=True makes DependsOnValidator hashable, which is required when
# the annotated type is a union (e.g. X | None).
@dataclass(frozen=True)
class DependsOnValidator:
    """Conditional dependency validator for Pydantic fields.

    Rejects a field value when a related field does not satisfy a precondition.
    Attach via ``Annotated`` metadata on the dependent field.

    Args:
        depends_on: Name of the field that this field depends on.
        depends_on_func: Predicate applied to the dependency field's value.
            Must return ``True`` for the dependent field to be accepted.
        value_func: Optional predicate on the current field's own value.
            When ``None`` (or when it returns falsy), the dependency check
            is skipped and the value passes through unchanged.

    Example::

        order_training_examples_by: Annotated[
            str | None,
            DependsOnValidator(
                depends_on="group_training_examples_by",
                depends_on_func=lambda v: v is not None,
                value_func=lambda v: v is not None,
            ),
            Field(...),
        ] = None
    """

    depends_on: str
    depends_on_func: Callable[[Any], bool]
    value_func: Callable[[Any], bool] | None

    def validate(self, value, info: ValidationInfo):
        """Run the dependency check during Pydantic validation.

        Args:
            value: The value being validated.
            info: Pydantic validation context containing sibling field data.

        Returns:
            The validated value if all conditions pass.

        Raises:
            ValueError: If the dependency field is absent from the model or
                its value does not satisfy ``depends_on_func``.
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
        """Register this validator as a Pydantic after-validator on the annotated field."""
        return core_schema.with_info_after_validator_function(self.validate, handler(source_type))


@dataclass(frozen=True)
class ValueValidator:
    """Predicate-based validator for field values.

    Wraps an arbitrary boolean function and raises ``ValueError`` when it
    returns ``False``.  Commonly used inline:
    ``ValueValidator(value_func=lambda v: v > 0)``.

    Args:
        value_func: Predicate applied to the field value.  Returns True if valid.
    """

    value_func: Callable[[Any], bool]

    def validate(self, value, info: ValidationInfo):
        """Apply ``value_func`` to the field value during Pydantic validation.

        Args:
            value: The raw field value.
            info: Pydantic validation context.

        Returns:
            The original ``value`` if the predicate passes.

        Raises:
            ValueError: If the predicate returns ``False``.
        """
        logger.debug(f"Field {info.field_name}, value={value}, validating")
        if self.value_func(value):
            return value
        try:
            # Sometimes inspect.getsource() raises an OSError for lambda
            # functions defined inline, hence the try/except.
            src = inspect.getsource(self.value_func)
            msg = f"Field {info.field_name}, value={value}, did not pass validation: {src}"
        except OSError:
            msg = f"Field {info.field_name}, value={value}, did not pass validation"
        raise ValueError(msg)

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """Register this validator as a Pydantic after-validator on the annotated field."""
        return core_schema.with_info_after_validator_function(self.validate, handler(source_type))


def range_validator(value: int | float, func: Callable) -> bool:
    """Check a numeric value against ``func``, passing ``"auto"`` sentinels unconditionally.

    ``func`` is typically a predicate that asserts a numeric range (e.g.
    non-negative, within ``(0, 1)``).

    Args:
        value: The value to validate -- may be a number or the string ``"auto"``.
        func: Predicate applied when ``value`` is numeric (e.g. ``lambda v: v >= 0``).

    Returns:
        ``True`` if ``value`` is ``"auto"`` or ``func(value)`` is truthy.
    """
    return True if value == "auto" else func(value)


AutoParamRangeValidator = ValueValidator(lambda p: range_validator(p, lambda v: v >= 0))
"""Pre-built ``ValueValidator`` that accepts ``"auto"`` or any non-negative number."""
