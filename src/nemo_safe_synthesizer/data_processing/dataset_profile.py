# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-wide column profiling for generation constraints and evaluation."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_object_dtype
from pydantic import BaseModel, Field, model_validator

from ..artifacts.base.fields import (
    HIGHLY_UNIQUE_FIELD_TYPES,
    NOMINAL_FIELD_TYPES,
    NUMERIC_FIELD_TYPES,
    TABULAR_FIELD_TYPES,
    FieldType,
)
from ..errors import DataError

JsonScalar = None | int | float | bool | str
TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD = 2

# Range and length keywords the structured-generation grammar ignores once a
# property becomes a JSON Schema type list; see ``_make_nullable``.
_UNENFORCEABLE_BOUND_KEYS = frozenset({"minimum", "maximum", "minLength", "maxLength"})

_NESTED_VALUE_TYPES = (list, dict, set, tuple, np.ndarray)

# Schema / enum detection parameters shared by discovery and JSON Schema export.
SCHEMA_ENUM_MAX_DISTINCT_EXP = 1 / 2
SCHEMA_ENUM_MAX_SINGLETONS_EXP = 1 / 3
STRING_LENGTH_MULTIPLE = 1.5


class IntegerConstraints(BaseModel):
    """Exact integer bounds for generation."""

    kind: Literal[FieldType.INTEGER] = FieldType.INTEGER
    min_value: int = Field(description="Exact observed integer minimum.")
    max_value: int = Field(description="Exact observed integer maximum.")


class FloatConstraints(BaseModel):
    """Exact floating-point bounds for generation."""

    kind: Literal[FieldType.FLOAT] = FieldType.FLOAT
    min_value: float = Field(description="Exact observed float minimum.")
    max_value: float = Field(description="Exact observed float maximum.")


class BinaryConstraints(BaseModel):
    """Exactly two observed values for a binary column."""

    kind: Literal[FieldType.BINARY] = FieldType.BINARY
    enum_values: list[JsonScalar] = Field(
        min_length=2,
        max_length=2,
        description="The two observed JSON-safe values.",
    )


class CategoricalConstraints(BaseModel):
    """Non-empty observed enum values for a categorical column."""

    kind: Literal[FieldType.CATEGORICAL] = FieldType.CATEGORICAL
    enum_values: list[JsonScalar] = Field(
        min_length=1,
        description="Observed JSON-safe enum values.",
    )


class StringLengthConstraints(BaseModel):
    """String length bounds for text or other non-enum string columns."""

    kind: Literal[FieldType.TEXT, FieldType.OTHER] = Field(description="Text vs other string column.")
    min_str_length: int = Field(ge=0, description="Minimum string length among non-null values.")
    max_str_length: int = Field(ge=0, description="Maximum string length among non-null values.")


class EmptyConstraints(BaseModel):
    """Constraints for an all-null column."""

    kind: Literal[FieldType.EMPTY] = FieldType.EMPTY


ColumnConstraints = Annotated[
    IntegerConstraints
    | FloatConstraints
    | BinaryConstraints
    | CategoricalConstraints
    | StringLengthConstraints
    | EmptyConstraints,
    Field(discriminator="kind"),
]


class ColumnProfile(BaseModel):
    """Type-tagged generation constraints for one modeled dataframe column."""

    name: str = Field(description="Column name in the modeled dataframe.")
    nullable: bool = Field(description="Whether the column contains missing values.")
    constraints: ColumnConstraints = Field(description="Type-specific generation constraints.")

    @property
    def field_type(self) -> FieldType:
        """Column type implied by ``constraints.kind``."""
        return FieldType(self.constraints.kind)

    @property
    def is_numeric(self) -> bool:
        """Whether this is an integer or floating-point column."""
        return self.field_type in NUMERIC_FIELD_TYPES

    @property
    def is_nominal(self) -> bool:
        """Whether this is a binary or categorical column."""
        return self.field_type in NOMINAL_FIELD_TYPES

    @property
    def is_tabular(self) -> bool:
        """Whether this column participates in tabular metrics."""
        return self.field_type in TABULAR_FIELD_TYPES

    @property
    def is_text(self) -> bool:
        """Whether this is a free-text column."""
        return self.field_type == FieldType.TEXT

    @property
    def is_highly_unique(self) -> bool:
        """Whether distribution metrics should skip this column."""
        return self.field_type in HIGHLY_UNIQUE_FIELD_TYPES


class DatasetProfile(BaseModel):
    """Per-run profile of every modeled training dataframe column."""

    columns: dict[str, ColumnProfile] = Field(description="Profiles in dataframe-column order.")

    @model_validator(mode="after")
    def validate_column_names(self) -> DatasetProfile:
        """Keep dictionary keys and embedded column names consistent."""
        mismatched = [name for name, profile in self.columns.items() if profile.name != name]
        if mismatched:
            raise ValueError(f"Profile column names do not match keys: {mismatched!r}")
        return self

    def get_columns_of_type(self, types: Iterable[FieldType]) -> list[str]:
        """Return names whose inferred type belongs to ``types``."""
        type_set = set(types)
        return [name for name, profile in self.columns.items() if profile.field_type in type_set]

    def tabular_columns(self) -> list[str]:
        """Return binary, categorical, integer, and floating-point columns."""
        return self.get_columns_of_type(TABULAR_FIELD_TYPES)

    def nominal_columns(self) -> list[str]:
        """Return binary and categorical columns."""
        return self.get_columns_of_type(NOMINAL_FIELD_TYPES)

    def text_columns(self) -> list[str]:
        """Return free-text columns."""
        return self.get_columns_of_type({FieldType.TEXT})

    def numeric_columns(self) -> list[str]:
        """Return integer and floating-point columns."""
        return self.get_columns_of_type(NUMERIC_FIELD_TYPES)

    def validate_against_dataframe(self, df: pd.DataFrame) -> None:
        """Raise when this profile does not describe exactly ``df``'s columns."""
        if list(self.columns) != list(df.columns):
            raise ValueError(
                "Dataset profile columns must exactly match the modeled dataframe columns. "
                f"Expected {list(df.columns)!r}, got {list(self.columns)!r}."
            )

    def to_json_schema(self, string_length_multiple: float = STRING_LENGTH_MULTIPLE) -> dict:
        """Export generation JSON Schema from the inferred profile."""
        properties: dict[str, dict] = {}
        required: list[str] = []
        for name, column in self.columns.items():
            property_schema = _column_to_json_schema(column, string_length_multiple)
            if column.nullable:
                property_schema = _make_nullable(property_schema)
            else:
                required.append(name)
            properties[name] = property_schema
        return {"type": "object", "properties": properties, "required": required}


@dataclass(frozen=True)
class _ColumnStats:
    """Observed column facts used to classify a profile (no pandas dependency)."""

    row_count: int
    non_null_count: int
    unique_count: int
    missing_count: int
    singleton_count: int
    min_str_length: int
    max_str_length: int
    avg_space_count: float | None
    is_float: bool
    is_integer: bool
    min_value: int | float | None
    max_value: int | float | None
    enum_values: tuple[JsonScalar, ...]


def discover_dataset_profile(df: pd.DataFrame) -> DatasetProfile:
    """Infer a complete profile for the modeled training dataframe.

    Args:
        df: Modeled training dataframe of flat, scalar-valued columns.

    Returns:
        A profile describing every column in ``df``.

    Raises:
        DataError: If any column holds nested values.
    """
    nested = _nested_columns(df)
    if nested:
        raise DataError(
            "Dataset profile discovery models flat records of scalar values, but these columns "
            f"contain nested values: {nested!r}. Flatten them into scalar columns or serialize "
            "them to strings before training."
        )
    return DatasetProfile(
        columns={name: _discover_column_profile(name, _compute_column_stats(df[name])) for name in df.columns}
    )


def _nested_columns(df: pd.DataFrame) -> list[str]:
    """Return names of columns holding list, dict, set, or array values."""
    return [
        str(name)
        for name in df.columns
        if is_object_dtype(df[name].dtype) and any(isinstance(value, _NESTED_VALUE_TYPES) for value in df[name])
    ]


def _compute_column_stats(series: pd.Series) -> _ColumnStats:
    """Extract classification inputs from one dataframe column."""
    non_null = series.dropna()
    non_null_count = len(non_null)
    row_count = len(series)
    lengths = [len(str(value)) for value in non_null]
    is_float = non_null_count > 0 and is_float_dtype(non_null.dtype)
    is_integer = non_null_count > 0 and is_integer_dtype(non_null.dtype)
    if is_float:
        min_value: int | float | None = float(non_null.min())
        max_value: int | float | None = float(non_null.max())
    elif is_integer:
        min_value = int(non_null.min())
        max_value = int(non_null.max())
    else:
        min_value = None
        max_value = None
    value_counts = series.value_counts(dropna=False)
    return _ColumnStats(
        row_count=row_count,
        non_null_count=non_null_count,
        unique_count=len(non_null.unique()) if non_null_count else 0,
        missing_count=row_count - non_null_count,
        singleton_count=int((value_counts == 1).sum()),
        min_str_length=min(lengths) if lengths else 0,
        max_str_length=max(lengths) if lengths else 0,
        avg_space_count=(sum(str(value).count(" ") for value in non_null) / non_null_count if non_null_count else None),
        is_float=is_float,
        is_integer=is_integer,
        min_value=min_value,
        max_value=max_value,
        enum_values=tuple(_enum_values(non_null)) if non_null_count else (),
    )


def _discover_column_profile(name: str, stats: _ColumnStats) -> ColumnProfile:
    """Classify one column from precomputed stats into a typed profile."""
    nullable = stats.missing_count > 0

    if stats.non_null_count == 0:
        return ColumnProfile(name=name, nullable=nullable, constraints=EmptyConstraints())
    if stats.unique_count == 2:
        return ColumnProfile(
            name=name,
            nullable=nullable,
            constraints=BinaryConstraints(enum_values=list(stats.enum_values)),
        )
    if _is_enum(stats):
        return ColumnProfile(
            name=name,
            nullable=nullable,
            constraints=CategoricalConstraints(enum_values=list(stats.enum_values)),
        )
    if stats.is_float:
        assert stats.min_value is not None and stats.max_value is not None
        return ColumnProfile(
            name=name,
            nullable=nullable,
            constraints=FloatConstraints(min_value=float(stats.min_value), max_value=float(stats.max_value)),
        )
    if stats.is_integer:
        assert stats.min_value is not None and stats.max_value is not None
        minimum = int(stats.min_value)
        if stats.unique_count <= 10 and minimum >= 0:
            return ColumnProfile(
                name=name,
                nullable=nullable,
                constraints=CategoricalConstraints(enum_values=list(stats.enum_values)),
            )
        return ColumnProfile(
            name=name,
            nullable=nullable,
            constraints=IntegerConstraints(min_value=minimum, max_value=int(stats.max_value)),
        )
    if stats.avg_space_count is not None and stats.avg_space_count > TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD:
        return ColumnProfile(
            name=name,
            nullable=nullable,
            constraints=StringLengthConstraints(
                kind=FieldType.TEXT,
                min_str_length=stats.min_str_length,
                max_str_length=stats.max_str_length,
            ),
        )
    return ColumnProfile(
        name=name,
        nullable=nullable,
        constraints=StringLengthConstraints(
            kind=FieldType.OTHER,
            min_str_length=stats.min_str_length,
            max_str_length=stats.max_str_length,
        ),
    )


def _is_enum(stats: _ColumnStats) -> bool:
    """Apply the legacy distinct-count and singleton-count enum heuristic."""
    distinct_with_nulls = stats.unique_count + (1 if stats.missing_count > 0 else 0)
    return (
        distinct_with_nulls <= stats.row_count**SCHEMA_ENUM_MAX_DISTINCT_EXP
        and stats.singleton_count <= stats.row_count**SCHEMA_ENUM_MAX_SINGLETONS_EXP
    )


def _enum_values(series: pd.Series) -> list[JsonScalar]:
    """Return non-null enum values in the legacy schema's sorted order."""
    return [_handle_enum_value(value) for value in series.value_counts().index.sort_values().tolist()]


def _column_to_json_schema(column: ColumnProfile, string_length_multiple: float) -> dict:
    """Build a non-null JSON Schema property for one profile."""
    match column.constraints:
        case IntegerConstraints(min_value=minimum, max_value=maximum):
            return {"type": "integer", "minimum": minimum, "maximum": maximum}
        case FloatConstraints(min_value=minimum, max_value=maximum):
            return {"type": "number", "minimum": minimum, "maximum": maximum}
        case BinaryConstraints(enum_values=enum_values) | CategoricalConstraints(enum_values=enum_values):
            return {"enum": enum_values}
        case StringLengthConstraints(min_str_length=min_length, max_str_length=max_length):
            return {
                "type": "string",
                "minLength": round(min_length / string_length_multiple),
                "maxLength": round(string_length_multiple * max_length),
            }
        case EmptyConstraints():
            return {"type": "null"}
        case _:
            raise ValueError(f"Unsupported column constraints: {column.constraints!r}")


def _make_nullable(property_schema: dict) -> dict:
    """Allow null for a property, dropping bounds that only reject records.

    A nullable non-enum property is exported as a JSON Schema type list
    (``["number", "null"]``). Neither structured-generation backend constrains a
    type list -- the grammar emits an unconstrained union of the listed types --
    so any retained ``minimum``/``maximum``/``minLength``/``maxLength`` acts
    purely as a post-generation validation gate that rejects records the model
    was never steered away from producing. Enum properties keep their full value
    set because the grammar does enforce those.
    """
    if "enum" in property_schema:
        return {**property_schema, "enum": [*property_schema["enum"], None]}
    if property_schema.get("type") == "null":
        return property_schema
    property_type = property_schema.get("type")
    unbounded = {key: value for key, value in property_schema.items() if key not in _UNENFORCEABLE_BOUND_KEYS}
    return {**unbounded, "type": [property_type, "null"]}


def _handle_enum_value(v: object) -> None | int | float | bool | str:
    """Convert a value to a JSON-safe Python scalar for enum schema entries.

    NumPy scalars and other non-builtin types are narrowed to the most
    precise builtin equivalent (``bool`` > ``int`` > ``float`` > ``str``).
    Returns None for NA/NaN values.
    """
    if pd.isna(v):  # ty: ignore[no-matching-overload] -- third-party stub mismatch
        return None

    if isinstance(v, (float, int, bool, str)):
        return v

    # Anything except builtin python types may cause crashes when we work with
    # the schema later, so we convert to bool, float, int, or str.
    if isinstance(v, np.bool_):
        return bool(v)

    with suppress(TypeError, ValueError, OverflowError):
        # Convert to python int if possible, but np.float32 and other float
        # types will be truncated by int(v), so check equality to make sure
        # we haven't lost precision.
        t = int(v)  # ty: ignore[invalid-argument-type] -- third-party stub mismatch
        if t == v:
            return t

    try:
        # Convert to python float if possible
        return float(v)  # ty: ignore[invalid-argument-type] -- third-party stub mismatch
    except (TypeError, ValueError, OverflowError):
        # Otherwise, ensure we're using a python str to avoid json encoding errors.
        return str(v)


def relax_numeric_bounds(schema: dict) -> dict:
    """Return a copy of ``schema`` with float (``number``) range bounds removed.

    ``DatasetProfile.to_json_schema`` records each numeric column's exact
    observed ``minimum``/``maximum``. Neither structured-generation backend can
    enforce a floating-point range (XGrammar and the regex builder bound
    integers and enums but emit an unconstrained token stream for ``number``),
    so those bounds act purely as a post-generation validation gate. On wide
    float tables the per-field rejections compound and can reject nearly every
    record even though the values are otherwise well-formed. Dropping the
    ``number`` bounds turns that hard rejection into acceptance while leaving
    integer and enum constraints -- which the grammar does enforce -- untouched.

    Only ``number``-typed properties are relaxed; a property whose type list
    includes ``integer`` keeps its (grammar-enforced, whole-number) bounds so the
    XGrammar constraint stays valid.

    Args:
        schema: A JSON schema as produced by ``DatasetProfile.to_json_schema``.

    Returns:
        A deep-ish copy of ``schema`` with ``minimum``/``maximum`` stripped from
        pure ``number`` properties. The input is not mutated.
    """
    relaxed = dict(schema)
    properties = relaxed.get("properties")
    if not isinstance(properties, dict):
        return relaxed

    new_properties: dict = {}
    for name, prop in properties.items():
        if isinstance(prop, dict):
            types = prop.get("type")
            types = types if isinstance(types, list) else [types]
            # Relax only pure floats; keep integer bounds (grammar-enforced and
            # required to be whole numbers) and enum constraints intact.
            if "number" in types and "integer" not in types:
                prop = {k: v for k, v in prop.items() if k not in ("minimum", "maximum")}
        new_properties[name] = prop
    relaxed["properties"] = new_properties
    return relaxed
