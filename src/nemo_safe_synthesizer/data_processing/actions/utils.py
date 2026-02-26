# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    """
    This alias fn allows `type_` to be parsed as `type` from config yaml. We use `type_`
    in the actual python objects so it doesn't conflict with the python builtin `type()`.
    """
    if field_name == "type_":
        return "type"

    return field_name


class MetadataColumns(StrEnum):
    INDEX = "__gretel__idx"  # used in validation to maintain a mapping to pre-transformed records
    REJECT_REASON = (
        "__gretel_reject_reason"  # used in validation to attach model_metadata about why the row was rejected
    )


def remove_metadata_columns_from_df(df: pd.DataFrame):
    metadata_cols = [col.value for col in MetadataColumns]

    columns_to_drop = [col for col in metadata_cols if col in df.columns]
    if columns_to_drop:
        df.drop(columns=columns_to_drop, inplace=True)

    return df


def remove_metadata_columns_from_records(records: list[dict]) -> list[dict]:
    metadata_cols = [col.value for col in MetadataColumns]

    new_records: list[dict] = []
    for record in records:
        new_records.append({k: v for k, v in record.items() if k not in metadata_cols})

    return new_records


class TransformsUpdate(BaseModel):
    """
    `transforms_v2` takes in untyped `dicts`, but this model adds a little
    bit of structure for better validation.
    """

    name: str
    value: str
    position: Optional[int] = None


class TransformsUtil:
    """
    Simple helper class to manage an instance of a TV2 `Environment` and some methods
    to run `Step`s on input data.
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
    model_config = ConfigDict(alias_generator=type_alias_fn)

    _ctx: ActionCtx = PrivateAttr()

    def with_ctx(self, ctx: ActionCtx) -> Self:
        self._ctx = ctx
        return self

    def generate_records(self, num_records: int, col: str = "newcol") -> list[dict[Hashable, Any]]:
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
    """
    This checks the two common ways that classes indicate themselves
    as abstract; they either have `@abstractmethod`s, or they explicitly
    inherit from `ABC` (or the metaclass). This checks both of these.
    """
    return inspect.isabstract(c) or ABC in c.__bases__


def all_subclasses(klass: type[T]) -> set[type[T]]:
    """
    Grab all of the recursive subclasses of `klass`.
    """
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
    """
    Find all the subclasses of `klass`, then filter out the abstract
    subclasses.

    This is useful for passing in a very abstract parent class
    like `BaseAction`, and finding all of the potential children
    of that `klass`. Some of these children themselves might be abstract,
    so we should filter those out.

    This function is likely used to feed information to `pydantic` about
    which potential concrete classes exist for purposes of validation and
    schema generation.
    """
    return set(c for c in all_subclasses(klass) if not is_abstract(c))


def guess_datetime_format(datetime_str: str) -> Optional[str]:
    # TODO: use `pandas.tseries.api.guess_datetime_format` in the future?
    format = parse_date(datetime_str)
    if format is None:
        return None
    return format.fmt_str


class ActionCtx(BaseModel):
    """
    Context available during all action execution. This object
    can be used for some state specific to the execution,
    as well as dependency injection for external services in the future.
    """

    seed: Optional[int] = None
    """
    Seed used for all random generation tasks
    """

    state: dict[str, str] = {}
    """
    Used for tracking state across multiple action invocations.
    This is important for actions which might have multiple functions
    which need to remember information in latter invocations. For example,
    a `postprocessing` function might benefit from information persisted
    inside a `preprocessing` function.
    """

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        np.random.seed(seed=self.seed)

    @cached_property
    def transforms_util(self) -> TransformsUtil:
        return TransformsUtil(seed=self.seed)
