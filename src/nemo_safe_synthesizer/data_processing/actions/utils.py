# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for the data actions framework.

Provides ``ActionCtx`` (execution context with state and dependency injection),
``TransformsUtil`` (wrapper around the transforms_v2 engine), helper types
(``MetadataColumns``, ``TransformsUpdate``), and subclass-discovery functions.
"""

from __future__ import annotations

import inspect
import uuid
from abc import ABC, abstractmethod
from enum import StrEnum
from functools import cached_property
from typing import (
    Annotated,
    Any,
    Callable,
    Hashable,
    Literal,
    Optional,
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
from typing_extensions import Self

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


class TransformsUpdate(BaseModel):
    """Typed wrapper for a single transforms_v2 update step."""

    name: str = Field(description="Target column name for the update.")
    value: str = Field(description="Jinja expression evaluated by the transforms_v2 engine.")
    position: Optional[int] = Field(default=None, description="Column insertion index when adding a new column.")


class TransformsUtil:
    """Wrapper around a transforms_v2 ``Environment`` for executing column updates and drop conditions.

    Args:
        seed: Random seed passed to the underlying ``Environment``.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        from ...pii_replacer.data_editor.edit import (
            Environment,
        )

        self.env = Environment(locales=None, seed=seed, globals_config={}, entity_extractor=None)

    def execute_col_updates(self, col: str, df: pd.DataFrame, updates: list[str]) -> pd.DataFrame:
        from ...pii_replacer.data_editor.edit import (
            ProgressLog,
            Step,
        )

        columns = {"add": []}
        rows = {"update": []}

        if col not in df.columns:
            columns["add"].append({"name": col})

        for step in updates:
            rows["update"].append({"name": col, "value": step})

        step = {"columns": columns, "rows": rows}
        return Step.execute(df, {}, {}, step, self.env, ProgressLog(30), None)

    def execute_updates(self, df: pd.DataFrame, updates: list[TransformsUpdate]) -> pd.DataFrame:
        from ...pii_replacer.data_editor.edit import (
            ProgressLog,
            Step,
        )

        columns = {"add": []}
        rows = {"update": []}

        for step in updates:
            col_name = step.name
            if col_name not in df.columns:
                position_dict = {}
                if (position := step.position) is not None:
                    position_dict["position"] = position
                columns["add"].append({"name": col_name, **position_dict})

            rows["update"].append({"name": col_name, "value": step.value})

        step = {"columns": columns, "rows": rows}
        return Step.execute(df, {}, {}, step, self.env, ProgressLog(30), None)

    def execute_drop_condition(self, batch: pd.DataFrame, conditions: list) -> pd.DataFrame:
        from ...pii_replacer.data_editor.edit import (
            ProgressLog,
            Step,
        )

        conditions_list = [{"condition": c} for c in conditions]
        step = {"rows": {"drop": conditions_list}}
        return Step.execute(batch, {}, {}, step, self.env, ProgressLog(30), None)


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


class ExpressionSource(DataSource):
    type_: Literal["expression"] = "expression"

    expression: str

    def generate_data(self, df: pd.DataFrame, col: str = "newcol") -> pd.DataFrame:
        return self._ctx.transforms_util.execute_col_updates(col, df, [self.expression])


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


def guess_datetime_format(datetime_str: str) -> Optional[str]:
    """Infer a ``strftime``-compatible format string from a date string, or None."""
    # TODO: use `pandas.tseries.api.guess_datetime_format` in the future?
    format = parse_date(datetime_str)
    if format is None:
        return None
    return format.fmt_str


class ActionCtx(BaseModel):
    """Execution context shared across all action invocations.

    Provides a random seed, a state dictionary for cross-phase communication,
    and a lazily-initialized ``TransformsUtil`` for expression evaluation.
    """

    seed: Optional[int] = Field(default=None, description="Seed used for all random generation tasks.")

    state: dict[str, str] = Field(
        default={}, description="Per-action state persisted across phases (keyed by BaseAction.hash())."
    )

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        np.random.seed(seed=self.seed)

    @cached_property
    def transforms_util(self) -> TransformsUtil:
        return TransformsUtil(seed=self.seed)
