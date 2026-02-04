# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from types import FunctionType
from typing import (
    Annotated,
    Any,
    Callable,
    Hashable,
    Literal,
    Optional,
    Protocol,
    Type,
    TypeVar,
    Union,
    cast,
)

import pandas as pd
from dateutil import parser
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    model_validator,
)

from ... import utils
from ...observability import get_logger
from .distributions import (
    DatetimeDistributionT,
    DistributionT,
)
from .utils import (
    ActionCtx,
    DataSourceT,
    MetadataColumns,
    TransformsUpdate,
    UniqueIdSource,
    concrete_subclasses,
    guess_datetime_format,
    type_alias_fn,
)

logger = get_logger(__name__)


class ProcessFn(Protocol):
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame: ...


class ValidateBatchFn(Protocol):
    def __call__(self, batch: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class ProcessPhase(str, Enum):
    PREPROCESS = "preprocess"
    POSTPROCESS = "postprocess"
    GENERATE = "generate"


class ValidateBatchPhase(str, Enum):
    VALIDATE_BATCH = "validate_batch"


FunctionPhase = Union[ProcessPhase, ValidateBatchPhase]


BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


def fn_to_action_type(fn: Callable) -> str:
    instance = getattr(fn, "__self__", None)
    if instance and isinstance(instance, BaseAction):
        return instance.get_type()
    return "unknown_action_type"


@dataclass
class Functions:
    preprocess: ProcessFn | None = None
    postprocess: ProcessFn | None = None
    validate_batch: ValidateBatchFn | None = None
    generate: ProcessFn | None = None


class BaseAction(BaseModel, ABC):
    """
    `BaseAction` is the key abstract parent class in the `data_actions` module.
    This class isn't meant to be instantiated directly, but to be extended
    in children classes which implement the below methods.
    """

    # allows `type` field in yaml to parse into the `type_` field
    model_config = ConfigDict(alias_generator=type_alias_fn)
    _ctx: ActionCtx = PrivateAttr()

    """
    These are all the functions that a `BaseAction` might implement, with
    provided default implementations. However, an abstract subclass of `BaseAction` might
    have requirements that a specific method is implemented, like how `GenerateAction`
    requires that `generate()` is implemented.
    """

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process/mutate the input dataset, before being sent into training. This
        is useful for modifying the shape or contents of the data, which might help
        if a model is going to use this data to train or finetune.
        """
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process/mutate the dataset, often used as "cleanup" step from whatever transformations
        occurred during the `preprocess` function.
        """
        return df

    def validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Validate the rows in a `batch` and return the rows that have passed validation, and those rejected.
        In this instance, the `batch` is some newly generated data that may or may not pass
        validation, and `df` is some context about the relevant dataset we're trying to validate into.
        """
        batch_copy = batch.copy()
        valid_mask = self._validate_batch(batch, df)
        return batch_copy[valid_mask], batch_copy[~valid_mask]

    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """
        Validate the rows in a `batch` and return a boolean mask Series indicating
        valid rows.
        """
        # By default, all rows are valid
        return pd.Series(True, index=batch.index)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new data to mutate into the dataframe. This can reference
        existing data inside the `df`, but in general the spirit of this method is
        that it is creating net-new data.
        """
        return df

    def functions(self) -> Functions:
        """
        Return all of the functions that this action emits.

        If the subclass didn't bother overwriting the method,
        don't include it in the list of returned `Functions`.
        This helps make debugging easier by seeing which actions
        are actually running functions at certain phases, versus having
        every action run a bunch of noop functions.
        """

        # We use FunctionType annotation for method rather than Callable
        # because Callable does not guarantee a __name__ attribute, so type
        # checking with ty will fail. See
        # https://docs.astral.sh/ty/reference/typing-faq/#why-does-ty-say-callable-has-no-attribute-__name__
        def _method_if_overridden(method: FunctionType) -> FunctionType | None:
            method_fn = getattr(method, "__func__", None)
            class_fn = getattr(BaseAction, method.__name__)
            if method_fn is not class_fn:
                return method
            else:
                return None

        return Functions(
            preprocess=_method_if_overridden(self.preprocess),
            postprocess=_method_if_overridden(self.postprocess),
            validate_batch=(self.validate_batch if _method_if_overridden(self._validate_batch) else None),
            generate=_method_if_overridden(self.generate),
        )

    @model_validator(mode="after")
    def add_ctx(self, info: ValidationInfo) -> "BaseAction":
        """
        When pydantic is constructing objects, it can optionally be provided
        a `context=` kwarg which will be available during deserialization.
        We use this mechanism to parse out the `action_ctx`
        """
        self._ctx = DEFAULT_ACTION_CTX
        if pydantic_ctx := info.context:
            if action_ctx := pydantic_ctx.get("action_ctx"):
                self._ctx = action_ctx

        return self

    def with_ctx(self, ctx: ActionCtx) -> "BaseAction":
        self._ctx = ctx
        return self

    def get_type(self) -> str:
        """
        This is a simple helper that should just return `type_`.
        Ideally, we'd be able to make `type_` an abstractproperty on the
        `BaseAction`, but that doesn't play nice with how pydantic expects
        discriminators to work, so we don't even define `type_` on the `BaseAction`
        at all.

        This function is our workaround to ensure callers can still figure out
        the type of the `BaseAction` that is being called, most likely only for debug
        purposes.
        """
        return getattr(self, "type_", "unknown")

    """
    A few small helpers used to mutate the state of the action's context.
    This can be useful for actions with multiple functions, where state needs to
    be remembered across different functions.
    """

    def hash(self) -> str:
        """
        Useful as a key in the `ActionCtx`'s state dictionary to store
        information for later functions.
        """
        return str(tuple(sorted(self.model_dump().items())))

    def set_state(self, state_obj: BaseModel) -> None:
        self._ctx.state[self.hash()] = state_obj.model_dump_json()

    def get_state(self, state_obj_type: Type[BaseModelT]) -> BaseModelT:
        state_obj_json = self._ctx.state[self.hash()]
        return state_obj_type.model_validate(json.loads(state_obj_json))


DEFAULT_ACTION_CTX = ActionCtx()


class GenerateAction(BaseAction, ABC):
    """
    `GenerateAction`s are simple actions that take in a dataframe
    and mutate it to include newly generated data. They generally operate
    during the `generate` phase, but can be configured via the
    `phase` instance variable available on all subclasses.

    You might want to create a new GenerateAction if you need to:
    - synthesize a new column based upon other columns
    - fill a dataframe with some faker data

    Technically any `BaseAction` can implement `generate` if it wants to, but this subclass
    is specially denoted so other consumers of `data_actions` (like jarvis tasks) can explicitly
    allow only tasks that certainly implement this method.
    """

    phase: ProcessPhase = ProcessPhase.GENERATE

    def functions(self) -> Functions:
        fns = Functions()
        if self.phase == ProcessPhase.PREPROCESS:
            fns.preprocess = self.generate
        elif self.phase == ProcessPhase.POSTPROCESS:
            fns.postprocess = self.generate
        elif self.phase == ProcessPhase.GENERATE:
            fns.generate = self.generate

        return fns

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate new data based on the existing data in the dataframe.
        """
        ...

    def generate_records(self, num_records: int) -> list[dict[Hashable, Any]]:
        """
        If you don't have an existing df you want to mutate, this will simply
        generate you records.
        """
        df = pd.DataFrame(index=range(num_records))
        return self.generate(df).to_dict("records")


class GenExpression(GenerateAction):
    type_: Literal["gen_expression"] = "gen_expression"

    col: str

    expression: Optional[str] = None
    """
    A jinja transforms_v2 expression that specifies the value of the column.
    """

    expressions: Optional[list[str]] = None
    """
    Similar to `expression`, but allows you to specify multiple statements
    that'll be processed in sequence to transforms_v2. This might be useful
    if you have a more complex set of expressions.
    """

    dtype: Optional[str] = None
    """
    If specified, the column will be cast as this `dtype` after generation.
    """

    _expressions: list[str] = PrivateAttr()

    @model_validator(mode="after")
    def validate_model(self) -> "BaseAction":
        if (self.expression is None and self.expressions is None) or (
            self.expression is not None and self.expressions is not None
        ):
            raise ValueError("Specify one and only one of `expression` or `expressions`.")

        if self.expression is not None:
            # TV2 expects a list of updates, so coerce a singular update into a list
            self._expressions = [self.expression]

        if self.expressions is not None:
            self._expressions = self.expressions

        return self

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ctx.transforms_util.execute_col_updates(self.col, df, self._expressions)
        if self.dtype is not None:
            df[self.col] = df[self.col].astype(self.dtype)  # type: ignore

        return df


class GenRawExpression(GenerateAction):
    """
    A "lower-level" action for utilizing transforms_v2 expressions.
    This action lets you pass in the whole `update` step payload, as opposed
    to the above action which restricts you into modifications on a singular column.

    Chances are you'd be better off using the `gen_expression` above, unless you want to
    pass more complex payloads into the transforms_v2 engine.
    """

    type_: Literal["gen_raw_expression"] = "gen_raw_expression"
    expressions: list[TransformsUpdate] = []

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._ctx.transforms_util.execute_updates(df, self.expressions)


class GenDistribution(GenerateAction):
    type_: Literal["gen_distribution"] = "gen_distribution"
    col: str
    distribution: DistributionT

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.col] = self.distribution.sample(num_records=len(df))
        return df


class GenDataSource(GenerateAction):
    type_: Literal["gen_datasource"] = "gen_datasource"
    col: str
    data_source: DataSourceT

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        self.data_source = self.data_source.with_ctx(self._ctx)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.data_source.generate_data(col=self.col, df=df)


class ReplaceDataSource(BaseAction):
    type_: Literal["replace_datasource"] = "replace_datasource"

    col: str
    data_source: DataSourceT

    class State(BaseModel):
        column_index: Optional[int]
        """
        The index in which `col` was before preprocessing dropped it. If `None`,
        then that means `col` was not in the original df.
        """

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        self.data_source = self.data_source.with_ctx(self._ctx)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.col in df.columns:
            column_index = df.columns.get_loc(self.col)
        else:
            column_index = None
        self.set_state(self.State(column_index=column_index))

        return df.drop(columns=[self.col], errors="ignore")

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        column_index = self.get_state(self.State).column_index
        df = self.data_source.generate_data(col=self.col, df=df)

        # If this column existed during preprocessing, ensure that we
        # place it into the same column index it was before
        if column_index is not None:
            col_data = df[self.col]
            df = df.drop(columns=[self.col])
            df.insert(column_index, self.col, col_data)

        return df


class GenDatetimeDistribution(GenerateAction):
    """
    Generate a datetime from a provided datetime distribution.
    """

    type_: Literal["gen_datetime_distribution"] = "gen_datetime_distribution"
    col: str
    distribution: DatetimeDistributionT

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        df[self.col] = self.distribution.sample(num_records=len(df))
        return df


class GenUniqueId(GenerateAction):
    type_: Literal["gen_unique_id"] = "gen_unique_id"
    col: str
    id_type: Literal["uuid4"] = "uuid4"

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)
        self._action = GenDataSource(col=self.col, data_source=UniqueIdSource(id_type=self.id_type)).with_ctx(self._ctx)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._action.generate(df)


class GenFaker(GenerateAction):
    type_: Literal["gen_faker"] = "gen_faker"
    col: str
    faker_fn: str

    @model_validator(mode="after")
    def validate_faker_fn(self) -> "BaseAction":
        transforms_util = self._ctx.transforms_util
        fn = getattr(transforms_util.env._fake, self.faker_fn, None)
        if fn is None:
            raise ValidationError(f"unknown `faker_fn`: [{self.faker_fn}]")

        return self

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        fn = getattr(self._ctx.transforms_util.env._fake, self.faker_fn)
        df[self.col] = df.apply(lambda _: fn(), axis=1)
        return df


class ValidationAction(BaseAction, ABC):
    @abstractmethod
    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series: ...


class DropExpression(ValidationAction):
    type_: Literal["expression_drop"] = "expression_drop"
    conditions: list[str] = []

    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        batch[MetadataColumns.INDEX] = range(len(batch))
        batch_copy = batch.copy()

        batch_copy = self._ctx.transforms_util.execute_drop_condition(batch=batch_copy, conditions=self.conditions)

        # Create a mask based on which temp IDs survived
        remaining_ids = set(batch_copy[MetadataColumns.INDEX])
        return pd.Series([i in remaining_ids for i in range(len(batch))], index=batch.index)


class DropDuplicates(ValidationAction):
    type_: Literal["drop_duplicates"] = "drop_duplicates"

    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        return ~batch.isin(df).all(axis=1)


class DateConstraint(BaseAction):
    type_: Literal["date_constraint"] = "date_constraint"
    colA: str
    colB: str
    operator: Literal["gt", "ge", "lt", "le"]

    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """
        Filter out all rows where the operator isn't true. The type of the
        actual cell could be a pandas datetime or a string at this point,
        so we first ensure `parser.parse` can coerce it into a datetime.
        """
        op = {
            "gt": operator.gt,
            "ge": operator.ge,
            "le": operator.le,
            "lt": operator.lt,
        }[self.operator]
        return op(
            batch[self.colA].apply(lambda x: parser.parse(str(x))),
            batch[self.colB].apply(lambda x: parser.parse(str(x))),
        )


class ColAction(BaseAction, ABC):
    """
    `ColAction`s are actions that act on one singular column. These are useful
    in describing seralization/deserialization rules, such as adjusting the format
    of the data before being sent into model training.
    """

    name: str


class DatetimeCol(ColAction):
    type_: Literal["datetime"] = "datetime"

    format: Optional[str] = None
    """
    Human-readable format of the datetime (see `strftime` in stdlib).
    If not specified, we will attempt to autodetect.
    """

    class State(BaseModel):
        dt_format: Optional[str]

    def _infer_col_dt_format(self, col: pd.Series) -> Optional[str]:
        dt_formats = col.apply(lambda x: guess_datetime_format(x))
        if len(dt_formats.unique()) > 1:
            logger.warning("Multiple time formats found: %s", dt_formats.unique())

        modes = dt_formats.mode()
        if len(modes) > 1:
            logger.warning("Multiple equally common formats found: %s", modes.tolist())

        return modes.iloc[0]

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        # Retrieve the datetime format, either from the user-config or by inferring it from the column
        dt_format = self.format
        if dt_format is None:
            dt_format = self._infer_col_dt_format(df[self.name])
            # Because the user didn't explicitly provide the dt_format,
            # we must remember our inferred format via `State`
            self.set_state(self.State(dt_format=dt_format))

        # Convert the datetime column to the specified format, coercing errors
        dates = pd.to_datetime(df[self.name], errors="coerce")
        # ty is having trouble inferring the type of dates
        dates = cast(pd.Series[pd.Timestamp], dates)
        df[self.name] = dates.apply(lambda x: x.strftime(dt_format) if pd.notna(x) else x)

        # Log a warning if any invalid datetime values are detected
        if df[self.name].isna().any():
            logger.warning("Invalid datetime value(s) detected.")

        # Remove rows with invalid datetime values (NaT)
        df = df[df[self.name].notna()]  # ty: ignore[invalid-assignment]

        return df

    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        # Retrieve datetime format, either from the instance or from the state
        dt_format = self.format or self.get_state(self.State).dt_format

        # Convert the datetime column, coercing errors, and apply the format
        dates = pd.to_datetime(batch[self.name], errors="coerce")
        # ty is having trouble inferring the type of dates
        dates = cast(pd.Series[pd.Timestamp], dates)
        batch[self.name] = dates.apply(lambda x: x.strftime(dt_format) if pd.notna(x) else x)

        return batch[self.name].notna()


class CategoricalCol(ColAction):
    type_: Literal["categorical"] = "categorical"
    values: list[Union[str, int, float]]

    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        return batch[self.name].isin(self.values)


"""
This is a fun little hack to ensure the typing works both statically
and dynamically. The definition of `ActionT` below ensures that the static
analysis (mypy/pyright) views `ActionT` effectively as a `BaseAction`; this is
what the `Annotated` does in python's `typing`.

However, the `discriminator` logic in pydantic uses the first type argument
of `Annotated` to determine all of the allowable types. To give pydantic what
it wants, we can override at runtime the `__origin__` (which is effectively
the first argument of `Annotated`) to include all subclasses of `BaseAction`.

Doing this gives us a few advantages:
- Registers all new subclasses of `BaseAction` without having to explicitly
    add to a Union list. This circumvents the common rigamarole of
    setting up a new Registry class, __init__subclass__, etc.
- Excludes subclasses which are abstract, ensuring intermediate
    subclasses aren't instantiable or suggested by the pydantic schema.
- Generates a proper `oneOf` json schema for any consuming `BaseModel`s.

It doesn't solve the issue about dynamically importing; we must first import all
the actions before they register with `__subclasses__`. Currently all the actions
are in this module, however, so that isn't an issue.
"""
ActionT = Annotated[BaseAction, Field(discriminator="type_")]
ActionT.__origin__ = Union[tuple(concrete_subclasses(BaseAction))]  # type: ignore

GenerateActionT = Annotated[GenerateAction, Field(discriminator="type_")]
GenerateActionT.__origin__ = Union[tuple(concrete_subclasses(GenerateAction))]  # type: ignore

ColT = Annotated[ColAction, Field(discriminator="type_")]
ColT.__origin__ = Union[tuple(concrete_subclasses(ColAction))]  # type: ignore


class ActionExecutor(BaseModel):
    """
    This provides an executor for running through user-provided actions.

    How this class is used will likely differ based upon the context in which we're running:
    - NavFT might be interested in running all steps at different phases,
        transforming the data before finetuning (preprocess), reverting the format back after generation (postprocess),
        validating the rows fit some constraints (validation), and then adding additional model_metadata (generation)

    """

    actions: list[ActionT]
    ctx: Optional[ActionCtx] = None

    _phase_to_functions: dict[FunctionPhase, list[Callable]] = PrivateAttr()

    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)

        if self.ctx is None:
            self.ctx = ActionCtx()

        # Rebuild the actions with the `ctx` properly attached. This ensures that each
        # action (and it's corresponding functions) properly reference the same
        # `ctx` during runtime.
        self.actions = [a.with_ctx(self.ctx) for a in self.actions]
        self._phase_to_functions = defaultdict(list)

        for action in self.actions:
            fns = action.functions()
            if fn := fns.preprocess:
                self._phase_to_functions[ProcessPhase.PREPROCESS].append(fn)
            if fn := fns.postprocess:
                # postprocess should go in reverse order to properly unravel preprocess
                self._phase_to_functions[ProcessPhase.POSTPROCESS].insert(0, fn)
            if fn := fns.validate_batch:
                self._phase_to_functions[ValidateBatchPhase.VALIDATE_BATCH].append(fn)
            if fn := fns.generate:
                self._phase_to_functions[ProcessPhase.GENERATE].append(fn)

    def _process(self, df: pd.DataFrame, process_fns: list[ProcessFn]) -> pd.DataFrame:
        df = df.copy()
        for fn in process_fns:
            if fn is not None:
                df = fn(df)
        return df

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._process(df, self._phase_to_functions[ProcessPhase.PREPROCESS])

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._process(df, self._phase_to_functions[ProcessPhase.POSTPROCESS])

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._process(df, self._phase_to_functions[ProcessPhase.GENERATE])

    def validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Start with all rows as valid and rejected rows as empty. By applying
        # the `validate_fn`s over time we'll incrementally shift some records from
        # valid -> rejected.
        valid_rows = batch.copy()
        rejected_rows = pd.DataFrame(columns=batch.columns)

        for validate_fn in self._phase_to_functions[ValidateBatchPhase.VALIDATE_BATCH]:
            valid_rows, rejected_rows_fn = validate_fn(valid_rows, df)
            rejected_rows_fn[MetadataColumns.REJECT_REASON] = fn_to_action_type(validate_fn)
            rejected_rows = pd.concat([rejected_rows, rejected_rows_fn])

        return valid_rows, rejected_rows


def data_actions_fn(
    action_executor: ActionExecutor,
) -> utils.DataActionsFn:
    """
    applies an action executor to a dataframe.
    """

    def fn(batch: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("Applying data_config postprocessing")

        logger.debug(f"Before postprocess: {utils.debug_fmt(batch)}")
        batch = action_executor.postprocess(batch)
        logger.debug(f"After postprocess: {utils.debug_fmt(batch)}")
        logger.info(
            f"Applying data_config validation on output batch of size [{len(batch)}]",
        )
        valid_df, rejected_df = action_executor.validate_batch(batch, df)
        logger.debug(f"valid_df after validate: {utils.debug_fmt(valid_df)}")
        logger.debug(f"rejected_df after validate: {utils.debug_fmt(rejected_df)}")

        logger.info(
            f"After data_config validation, output batch size is [{len(valid_df)}]",
        )
        return valid_df, rejected_df

    return fn
