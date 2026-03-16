# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extensible data action framework for pre/post-processing, validation, and generation.

Defines ``BaseAction`` and its subclasses (``GenerateAction``, ``ColAction``,
``ValidationAction``) which encapsulate data transformations applied at
different pipeline phases. ``ActionExecutor`` orchestrates running the
registered actions in order.
"""

from __future__ import annotations

import json
import operator
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Hashable
from dataclasses import dataclass
from enum import Enum
from types import MethodType
from typing import (
    Annotated,
    Any,
    Literal,
    Optional,
    Protocol,
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
    """Callable that transforms a DataFrame in-place during a processing phase."""

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame: ...


class ValidateBatchFn(Protocol):
    """Callable that splits a batch into valid and rejected DataFrames."""

    def __call__(self, batch: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: ...


class ProcessPhase(str, Enum):
    """Pipeline phases that apply DataFrame-to-DataFrame transformations."""

    PREPROCESS = "preprocess"
    POSTPROCESS = "postprocess"
    GENERATE = "generate"


class ValidateBatchPhase(str, Enum):
    """Pipeline phase for batch validation."""

    VALIDATE_BATCH = "validate_batch"


FunctionPhase = ProcessPhase | ValidateBatchPhase


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
    """Abstract base class for all data actions in the pipeline.

    Subclasses implement one or more phase methods (``preprocess``,
    ``postprocess``, ``validate_batch``, ``generate``) to transform data at
    the corresponding pipeline stage. The ``functions`` method introspects
    which methods were actually overridden, so only non-default actions run.

    State can be shared across phases via ``set_state`` / ``get_state``,
    which persist to the ``ActionCtx.state`` dictionary keyed by the
    action's ``hash``.
    """

    model_config = ConfigDict(alias_generator=type_alias_fn)
    _ctx: ActionCtx = PrivateAttr()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the input dataset before training.

        Override to modify the shape or contents of the data (e.g., encoding
        datetimes, dropping columns). The default implementation is a no-op.
        """
        return df

    def postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform generated data after generation, often reverting preprocessing.

        The default implementation is a no-op.
        """
        return df

    def validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a generated batch into valid and rejected rows.

        Args:
            batch: Newly generated data to validate.
            df: Reference dataset providing context for validation.

        Returns:
            A tuple of (valid_rows, rejected_rows) DataFrames.
        """
        batch_copy = batch.copy()
        valid_mask = self._validate_batch(batch, df)
        return batch_copy[valid_mask], batch_copy[~valid_mask]

    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        """Return a boolean mask indicating valid rows in ``batch``.

        The default implementation marks all rows as valid. Override in
        subclasses to implement actual validation logic.
        """
        return pd.Series(True, index=batch.index)

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate new data and merge it into the DataFrame.

        Override to create net-new columns or rows. The default implementation
        is a no-op.
        """
        return df

    def functions(self) -> Functions:
        """Return a ``Functions`` bundle containing only the overridden phase methods.

        Methods that were not overridden from ``BaseAction`` are excluded so
        that only actions with real work appear during debugging.
        """

        # We use FunctionType annotation for method rather than Callable
        # because Callable does not guarantee a __name__ attribute, so type
        # checking with ty will fail. See
        # https://docs.astral.sh/ty/reference/typing-faq/#why-does-ty-say-callable-has-no-attribute-__name__
        def _method_if_overridden(method: MethodType) -> MethodType | None:
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
        """Inject ``ActionCtx`` from pydantic's validation context, if provided."""
        self._ctx = DEFAULT_ACTION_CTX
        if pydantic_ctx := info.context:
            if action_ctx := pydantic_ctx.get("action_ctx"):
                self._ctx = action_ctx

        return self

    def with_ctx(self, ctx: ActionCtx) -> "BaseAction":
        """Attach an ``ActionCtx`` and return self for chaining."""
        self._ctx = ctx
        return self

    def get_type(self) -> str:
        """Return the discriminator ``type_`` value, or ``"unknown"`` if unset.

        Works around the fact that ``type_`` cannot be an abstract property
        on ``BaseAction`` due to pydantic discriminator constraints.
        """
        return getattr(self, "type_", "unknown")

    def hash(self) -> str:
        """Deterministic key for storing per-action state in ``ActionCtx.state``."""
        return str(tuple(sorted(self.model_dump().items())))

    def set_state(self, state_obj: BaseModel) -> None:
        """Persist a Pydantic model as JSON in ``ActionCtx.state``."""
        self._ctx.state[self.hash()] = state_obj.model_dump_json()

    def get_state(self, state_obj_type: type[BaseModelT]) -> BaseModelT:
        """Retrieve and deserialize a previously persisted state object."""
        state_obj_json = self._ctx.state[self.hash()]
        return state_obj_type.model_validate(json.loads(state_obj_json))


DEFAULT_ACTION_CTX = ActionCtx()


class GenerateAction(BaseAction, ABC):
    """Action that generates net-new data for the DataFrame.

    ``GenerateAction`` subclasses must implement ``generate``. The ``phase``
    field controls when ``generate`` runs:

    - ``GENERATE`` (default) -- after training, during synthetic data creation.
    - ``PREPROCESS`` -- before training.
    - ``POSTPROCESS`` -- after generation, for cleanup.

    Create a new ``GenerateAction`` when you need to synthesize a column
    based on other columns, fill in faker data, etc.
    """

    phase: ProcessPhase = ProcessPhase.GENERATE

    def functions(self) -> Functions:
        """Route ``generate`` to the correct phase slot based on ``self.phase``."""
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
        """Generate new data based on the existing data in the DataFrame."""
        ...

    def generate_records(self, num_records: int) -> list[dict[Hashable, Any]]:
        """Generate records without an existing DataFrame.

        Creates an empty DataFrame with ``num_records`` rows, runs ``generate``,
        and returns the result as a list of dicts.
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
    """Low-level action that passes raw transforms_v2 update payloads.

    Unlike ``GenExpression`` which targets a single column, this action
    accepts a full list of ``TransformsUpdate`` steps. Prefer
    ``GenExpression`` for simpler use cases.
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
            column_index = cast(int, df.columns.get_loc(self.col))
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
    """Generate a datetime from a provided datetime distribution."""

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
    """Action that operates on a single named column.

    Useful for defining serialization/deserialization rules (e.g., datetime
    formatting, categorical validation) applied before training or after
    generation.
    """

    name: str


class DatetimeCol(ColAction):
    type_: Literal["datetime"] = "datetime"

    format: Optional[str] = None
    """
    Human-readable format of the datetime (see ``strftime`` in stdlib).
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
        batch[self.name] = dates.apply(lambda x: x.strftime(dt_format) if pd.notna(x) else x)

        return batch[self.name].notna()


class CategoricalCol(ColAction):
    type_: Literal["categorical"] = "categorical"
    values: list[str | int | float]

    def _validate_batch(self, batch: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
        return batch[self.name].isin(self.values)


# ActionT / GenerateActionT / ColT use a hack to make typing work both
# statically and dynamically. The Annotated wrapper lets static analysis
# (mypy/pyright/ty) view ActionT as a BaseAction. However, pydantic's
# discriminator logic uses the first type argument of Annotated to determine
# all allowable types. We override __origin__ at runtime to include all
# concrete subclasses of BaseAction, which gives us:
#
# - Auto-registration of new BaseAction subclasses without a manual Union
#   list or a Registry class / __init_subclass__ pattern.
# - Exclusion of abstract subclasses, so intermediate ABCs aren't
#   instantiable or suggested by the pydantic schema.
# - A proper oneOf JSON schema for any consuming BaseModel.
#
# This does not solve dynamic importing -- all actions must be imported
# before they register with __subclasses__. Currently all actions live in
# this module, so that isn't an issue.
ActionT = Annotated[BaseAction, Field(discriminator="type_")]
ActionT.__origin__ = Union[tuple(concrete_subclasses(BaseAction))]  # type: ignore  # noqa: UP007 -- runtime Union needed for dynamic tuple()

GenerateActionT = Annotated[GenerateAction, Field(discriminator="type_")]
GenerateActionT.__origin__ = Union[tuple(concrete_subclasses(GenerateAction))]  # type: ignore  # noqa: UP007 -- runtime Union needed for dynamic tuple()

ColT = Annotated[ColAction, Field(discriminator="type_")]
ColT.__origin__ = Union[tuple(concrete_subclasses(ColAction))]  # type: ignore  # noqa: UP007 -- runtime Union needed for dynamic tuple()


class ActionExecutor(BaseModel):
    """Orchestrate a sequence of ``BaseAction`` instances across pipeline phases.

    Groups each action's overridden methods by phase (preprocess, postprocess,
    validate_batch, generate) and runs them in order. Postprocess functions
    run in reverse order to properly unwind preprocessing transformations.
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
    """Applies an action executor to a dataframe."""

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
