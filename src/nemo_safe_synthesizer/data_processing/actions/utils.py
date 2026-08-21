# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the data actions framework.

Provides ``ActionCtx`` (execution context with state and dependency injection),
helper types (``MetadataColumns``), data sources, and subclass-discovery functions.
"""

from __future__ import annotations

import inspect
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from enum import StrEnum
from typing import (
    Annotated,
    Any,
    Literal,
    Self,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
)

from .dates import parse_date

T = TypeVar("T")


def type_alias_fn(field_name: str) -> str:
    """Pydantic alias generator that maps ``type_`` to ``type`` for YAML compatibility."""
    if field_name == "type_":
        return "type"

    return field_name


class MetadataColumns(StrEnum):
    """Internal column names injected during validation phases."""

    INDEX = "__nss__idx"
    """Temporary index for mapping back to pre-transformed records."""

    REJECT_REASON = "__nss_reject_reason"
    """Reason a row was rejected during batch validation."""


def remove_metadata_columns_from_df(df: pd.DataFrame):
    """Drop all ``MetadataColumns`` from the DataFrame in-place."""
    metadata_cols = [col.value for col in MetadataColumns]

    columns_to_drop = [col for col in metadata_cols if col in df.columns]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)

    return df


def remove_metadata_columns_from_records(records: list[dict]) -> list[dict]:
    """Return a copy of each record dict with ``MetadataColumns`` keys removed."""
    metadata_cols = [col.value for col in MetadataColumns]

    new_records: list[dict] = []
    for record in records:
        new_records.append({k: v for k, v in record.items() if k not in metadata_cols})

    return new_records


class DataSource(BaseModel, ABC):
    """Abstract base for pluggable data sources used by ``GenDataSource`` actions.

    Subclasses implement ``generate_data`` to populate a column in an existing
    DataFrame. ``generate_records`` is a convenience wrapper that creates an
    empty DataFrame first.
    """

    model_config = ConfigDict(alias_generator=type_alias_fn)

    _ctx: ActionCtx = PrivateAttr()

    def with_ctx(self, ctx: ActionCtx) -> Self:
        """Attach an ``ActionCtx`` and return self for chaining."""
        self._ctx = ctx
        return self

    def generate_records(self, num_records: int, col: str = "newcol") -> list[dict[Hashable, Any]]:
        """Generate records as a list of dicts without an existing DataFrame."""
        df = pd.DataFrame(index=range(num_records))
        return self.generate_data(df, col).to_dict("records")

    @abstractmethod
    def generate_data(self, df: pd.DataFrame, col: str = "newcol") -> pd.DataFrame: ...


class UniqueIdSource(DataSource):
    type_: Literal["uuid"] = "uuid"

    id_type: Literal["uuid4"] = "uuid4"

    def generate_data(self, df: pd.DataFrame, col: str = "newcol") -> pd.DataFrame:
        id_fn: Callable[[Any], Any] = {
            "uuid4": lambda _: str(uuid.uuid4()),
        }[self.id_type]
        df[col] = df.apply(lambda batch: id_fn(batch), axis=1)
        return df


DataSourceT = Annotated[DataSource, Field(discriminator="type_")]
DataSourceT.__origin__ = Union[tuple(DataSource.__subclasses__())]  # type: ignore  # noqa: UP007 -- runtime Union needed for dynamic tuple()


def is_abstract(c: Any) -> bool:
    """Return True if the class has abstract methods or directly inherits ``ABC``."""
    return inspect.isabstract(c) or ABC in c.__bases__


def all_subclasses(klass: type[T]) -> set[type[T]]:
    """Recursively collect all subclasses of ``klass``."""
    subclasses: set[type[T]] = set()
    subclass_queue = [klass]
    while subclass_queue:
        parent = subclass_queue.pop()
        for subclass in parent.__subclasses__():
            if subclass not in subclasses:
                subclasses.add(subclass)
                subclass_queue.append(subclass)
    return subclasses


def concrete_subclasses(klass: type[T]) -> set[type[T]]:
    """Return all non-abstract recursive subclasses of ``klass``.

    Used by pydantic discriminated unions (e.g., ``ActionT``) to
    auto-discover instantiable action types for validation and schema
    generation.
    """
    return set(c for c in all_subclasses(klass) if not is_abstract(c))


def guess_datetime_format(datetime_str: str) -> str | None:
    """Infer a ``strftime``-compatible format string from a date string, or None."""
    # TODO: use `pandas.tseries.api.guess_datetime_format` in the future?
    format = parse_date(datetime_str)
    if format is None:
        return None
    return format.fmt_str


class ActionCtx(BaseModel):
    """Execution context shared across all action invocations.

    Provides a random seed and a state dictionary for cross-phase communication.
    """

    seed: int | None = Field(default=None, description="Seed used for all random generation tasks.")

    state: dict[str, str] = Field(
        default={}, description="Per-action state persisted across phases (keyed by BaseAction.hash())."
    )

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        np.random.seed(seed=self.seed)
