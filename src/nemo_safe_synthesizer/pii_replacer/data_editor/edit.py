# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from time import monotonic
from typing import Any

import pandas as pd
import yaml
from jinja2 import Template
from jinja2.exceptions import TemplateError, TemplateSyntaxError

from ...observability import get_logger
from .detect import DEFAULT_ENTITIES, EntityExtractor
from .environment import (
    Environment,
    SafeSynthesizerFakerMethodNotFound,
)

logger = get_logger(__name__)

CHUNKSIZE = 1000000

Rule = dict[str, Any]


class TransformFnAccounting:
    included_fns: set[str]
    column_fns: dict[str, set[str]]

    def __init__(self, included_fns: list[str]):
        """included_fns - list of function / filter names that should be included in accounting"""
        self.included_fns = set(included_fns)
        self.column_fns = defaultdict(set)

    def update(self, column_names: str | Iterable[str], fns: str | set[str]):
        """
        Records the functions and filters that are being applied to columns.

        column_names - iterable of column names to record filters / functions on. An individual string
          column name is also accepted.
        fns - set of names of filters / functions
        """
        if isinstance(fns, str):
            fns = set([fns])
        fns &= self.included_fns
        if not fns:
            fns = {"jinja"}
        if isinstance(column_names, str):
            column_names = [column_names]
        for column_name in column_names:
            self.column_fns[column_name] |= fns


@dataclass
class ProgressStatus:
    step_n: int = 0
    step_n_total: int = 0
    update_rule_n: int = 0
    update_rule_n_total: int = 0
    update_rule_description: str = ""
    row_n: int = 0
    row_n_total: int = 0
    column_n: int = 0
    column_n_total: int = 0
    column_name: str = ""


class ProgressLog:
    status: ProgressStatus
    start_time: float
    last_log: float
    log_duration: float  # Minimum number of seconds between logs

    def __init__(self, log_duration: float):
        self.status = ProgressStatus()
        self.start_time = monotonic()
        self.last_log = monotonic()
        self.log_duration = log_duration

    def log_throttled(self, force: bool = False):
        """
        Log if sufficient time has passed since previous log.

        If `force` is `True`, log regardless of time passed.
        """
        if force or monotonic() - self.last_log > self.log_duration:
            duration = monotonic() - self.start_time
            rows_per_second = 0 if duration == 0 else (self.status.row_n) / duration
            speed_emoji = "🐇" if rows_per_second >= 10 else "🐢"
            column_string = (
                f""""{self.status.column_name}", #{self.status.column_n + 1} of {self.status.column_n_total}"""
                if self.status.column_name
                else ""
            )
            duration_string = f"{duration:.1f} seconds" if duration < 120 else f"{duration / 60:.1f} minutes"
            update_rule_description = ""
            if self.status.update_rule_description:
                update_rule_description = f'"{self.status.update_rule_description}"'
            row_n_conditional_s = "s" if self.status.row_n != 1 else ""
            progress_data = {
                "transform_time": duration_string,
                "step": f"{self.status.step_n + 1} of {self.status.step_n_total}",
                "rule": f"{self.status.update_rule_n + 1} of {self.status.update_rule_n_total} {update_rule_description}",
                "column": column_string,
                "progress": f"{self.status.row_n} row{row_n_conditional_s} out of {self.status.row_n_total} transformed",
                "speed": f"{speed_emoji} {rows_per_second:.1f} rows per second.",
            }
            logger.user.info(
                "",
                extra={
                    "ctx": {
                        "render_table": True,
                        "tabular_data": progress_data,
                        "title": "Transformation Progress",
                    }
                },
            )

            self.last_log = monotonic()


class Step:
    _env: Environment
    _vars: dict[str, str | dict | list]

    def do_make_template(self, template_str: str) -> Template:
        return self._env.make_template(template_str)

    def make_template(self, template_str: str) -> Template:
        try:
            return self.do_make_template(template_str)
        except TemplateError as e:
            raise Exception(
                f"Error building jinja template '{template_str}': {e}",
                error_id="param",
            )

    def template_to_fnames(self, template_str) -> set[str]:
        retval = set()
        try:
            retval = self._env.template_to_fnames(template_str)
        except TemplateError:
            # Let other template functions raise the error
            pass
        return retval

    def _render_column(
        self,
        template: Template,
        column: pd.Series,
        **kwargs,
    ) -> str:
        """Render the full column into a single string from template"""
        self._env.maybe_seed(column)
        for k, v in kwargs.items():
            setattr(column, k, v)
        return template.render(column=column, vars=self._vars, **kwargs)

    def _render_cell(
        self,
        row: pd.Series,
        template: Template,
        column: dict[str, Any],
        fallback_template: Template | None = None,
        foreach: Template | None = None,
        progress: ProgressLog | None = None,
        fn_names: list[str] | None = None,
        fallback_fn_names: list[str] | None = None,
        fnreport: TransformFnAccounting | None = None,
        **kwargs,
    ) -> str:
        if progress is not None:
            progress.log_throttled()

        this = row[column["name"]]
        self._env.maybe_seed(this)
        self._env.entity_extractor.current_column = column["name"]
        if foreach:
            foreach_str = foreach.render(
                row=row,
                index=row.name,
                column=column,
                this=this,
                vars=self._vars,
                **kwargs,
            )

            foreach_itr = None

            try:
                foreach_itr = ast.literal_eval(foreach_str)
            except (ValueError, TypeError, SyntaxError):
                pass

            try:
                foreach_itr = json.loads(foreach_str)
            except (json.JSONDecodeError, TypeError):
                pass

            try:
                iter(foreach_itr)
            except TypeError:
                return f"[Error] '{foreach_str}' is not iterable built-in python type or JSON blob."

        else:
            foreach_itr = [None]

        try:
            cell = this
            for foreach_item in foreach_itr:
                cell = template.render(
                    row=row,
                    index=row.name,
                    column=column,
                    this=cell,
                    items=foreach_itr,
                    item=foreach_item,
                    vars=self._vars,
                    **kwargs,
                )
                if fnreport:
                    fnreport.update(column["name"], fn_names)
            if progress is not None:
                progress.status.row_n += 1
                progress.log_throttled()
            return cell
        except SafeSynthesizerFakerMethodNotFound:
            raise
        except Exception as e:
            if fallback_template:
                if fnreport:
                    fnreport.update(column["name"], fallback_fn_names)
                cell = self._render_cell(
                    row,
                    fallback_template,
                    column,
                    foreach=foreach,
                    progress=None,
                    **kwargs,
                )
            else:
                cell = f"[Error] {e}"
            if progress is not None:
                progress.status.row_n += 1
                progress.log_throttled()
            return cell

    def _render_row(self, template: Template, row, **kwargs) -> str:
        return template.render(row=row, index=row.name, vars=self._vars, **kwargs)

    def _add_columns(self, df: pd.DataFrame, rules: list[Rule]):
        for col in rules:
            name = col["name"]
            position = col.get("position")
            if position is None:
                position = len(df.columns)
            df.insert(position, name, None)

    def _rename_columns(self, df: pd.DataFrame, rules: list[Rule]):
        # Keeping it simple for now, renaming columns by name only (not condition).
        locked = set(self._env._env.globals["globals"].get("lock_columns") or [])
        column_names = {
            col["name"]: col["value"]
            for col in rules
            if "name" in col and col["name"] not in locked and col["value"] not in locked
        }
        df.rename(columns=column_names, inplace=True)

    def _drop_rows(self, df: pd.DataFrame, rules: list[Rule]):
        conditions = [self.make_template(rule["condition"]) for rule in rules]
        for condition in conditions:
            row_filter = df.apply(lambda row: self._render_row(condition, row), axis=1)
            df.drop(index=df.index[row_filter == "True"], inplace=True)

    def _parse_drop_columns_rule(
        self,
        rule: Rule,
        df: pd.DataFrame,
        entities: dict[str, str | None],
        column_types: dict[str, str | None],
    ):
        column_name = rule.get("name")
        rule_entities = rule.get("entity")
        position = rule.get("position")
        condition = rule.get("condition")
        rule_coltypes = rule.get("type")
        if column_name:
            if isinstance(column_name, list):
                columns = column_name
            else:
                columns = [column_name]
        elif rule_entities:
            if not isinstance(rule_entities, list):
                rule_entities = [rule_entities]
            columns = [col for col in df.columns if entities.get(col) in rule_entities]
        elif rule_coltypes:
            if not isinstance(rule_coltypes, list):
                rule_coltypes = [rule_coltypes]
            columns = [col for col in df.columns if column_types.get(col) in rule_coltypes]
        elif position is not None:
            if isinstance(position, list):
                position_list = position
            else:
                position_list = [position]
            columns = [df.columns[pos] for pos in position_list]
        elif condition:
            columns = df.columns
            condition = self.make_template(condition)
        else:
            raise Exception(
                f"column drop rule must contain one of name, entity, position, or condition. {rule}",
                error_id="param",
            )
        return columns, condition

    def _drop_columns(
        self,
        df: pd.DataFrame,
        rules: list[Rule],
        entities: dict[str, str | None],
        column_types: dict[str, str | None],
        fnreport: TransformFnAccounting | None = None,
    ):
        locked = set(self._env._env.globals["globals"].get("lock_columns") or [])
        try:
            for rule in rules:
                columns, condition_tmpl = self._parse_drop_columns_rule(rule, df, entities, column_types)
                to_drop = []
                if condition_tmpl:
                    column_properties = {}
                    for position, column_name in enumerate(columns):
                        column_properties[column_name] = {
                            "name": column_name,
                            "entity": entities.get(column_name),
                            "position": position,
                            "type": column_types.get(column_name),
                            # Note - dtype included as Series attribute
                        }
                    colfilter = df.apply(
                        lambda column: self._render_column(condition_tmpl, column, **column_properties[column.name]),
                        axis=0,
                    )
                    colfilter[locked & set(columns)] = "False"
                    to_drop = df.columns[colfilter == "True"]
                    df.drop(columns=to_drop, inplace=True)
                else:
                    to_drop = list(set(columns) - locked)
                    df.drop(columns=to_drop, inplace=True)
                if fnreport:
                    fnreport.update(to_drop, "drop")
        except KeyError as keyerr:
            raise Exception(
                f"Attempting to drop nonexistent column: {keyerr}",
                error_id="param",
            )

    def update_ner_cache(self, texts: pd.Series, entities: set[str] | None = None):
        self._env.entity_extractor.batch_update_cache([str(s) for s in texts], entities)

    def _parse_update_rows_rule(
        self,
        rule: Rule,
        df: pd.DataFrame,
        entities: dict[str, str | None],
        column_types: dict[str, str | None],
    ):
        column_name = rule.get("name")
        rule_entities = rule.get("entity")
        condition = rule.get("condition")
        rule_coltypes = rule.get("type")
        if column_name:
            if isinstance(column_name, list):
                columns = column_name
            else:
                columns = [column_name]
        elif rule_entities:
            if not isinstance(rule_entities, list):
                rule_entities = [rule_entities]
            columns = [col for col in df.columns if entities.get(col) in rule_entities]
        elif rule_coltypes:
            if not isinstance(rule_coltypes, list):
                rule_coltypes = [rule_coltypes]
            columns = [col for col in df.columns if column_types.get(col) in rule_coltypes]
        elif condition:
            columns = df.columns
            condition = self.make_template(condition)
        else:
            raise Exception(
                f"row update rule must contain one of name, entity, or condition. {rule}",
                error_id="param",
            )
        for column in columns:
            if column not in df.columns:
                raise Exception(
                    f"The column '{column}' was not found. If you are adding a column and wish to access it, be sure to place the column.add rule in a step prior to the step accessing the column.",
                    error_id="param",
                )
        return columns, condition

    def _update_rows(
        self,
        df: pd.DataFrame,
        rules: list[Rule],
        entities: dict[str, str | None],
        column_types: dict[str, str | None],
        progress: ProgressLog,
        fnreport: TransformFnAccounting,
    ):
        locked = set(self._env._env.globals["globals"].get("lock_columns") or [])
        progress.status.update_rule_n_total = len(rules)
        for rule_n, rule in enumerate(rules):
            progress.status.update_rule_n = rule_n
            progress.status.update_rule_description = rule.get("description")
            columns, condition = self._parse_update_rows_rule(rule, df, entities, column_types)
            columns = [col for col in columns if col not in locked]

            foreach = rule.get("foreach")
            if foreach:
                foreach = self.make_template(foreach)

            fns = self.template_to_fnames(rule["value"])
            value = self.make_template(rule["value"])
            fallback_value = rule.get("fallback_value")
            fallback_fns = None
            if fallback_value:
                fallback_fns = self.template_to_fnames(fallback_value)
                fallback_value = self.make_template(fallback_value)
            progress.status.column_n_total = len(columns)
            for position, column_name in enumerate(columns):
                column_properties = {
                    "dtype": df.dtypes[column_name],
                    "name": column_name,
                    "entity": entities.get(column_name),
                    "type": column_types.get(column_name),
                    "position": position,
                }
                if condition:
                    row_filter = df.apply(
                        lambda row: self._render_cell(row, condition, column=column_properties),
                        axis=1,
                    )
                    rows = df.index[row_filter == "True"]
                else:
                    rows = df.index

                progress.status.column_name = column_name
                progress.status.column_n = position
                progress.status.row_n_total = len(rows)
                progress.status.row_n = 0
                if rows.empty:
                    continue
                if fns & self._env.ner_cacheable_filters:
                    self.update_ner_cache(df.loc[rows, column_name])
                df.loc[rows, column_name] = df.loc[rows].apply(
                    self._render_cell,
                    args=(
                        value,
                        column_properties,
                        fallback_value,
                        foreach,
                        progress,
                        fns,
                        fallback_fns,
                        fnreport,
                    ),
                    axis=1,
                )
                progress.log_throttled(force=True)

    @classmethod
    def execute(
        cls,
        df: pd.DataFrame,
        entities: dict[str, str | None],
        column_types: dict[str, str | None],
        step_config: dict[str, dict],
        env: Environment,
        progress: ProgressLog,
        fnreport: TransformFnAccounting | None,
    ):
        step = cls()
        step._env = env
        step._vars = {}
        vars_config = step_config.get("vars") or {}
        for var_name, var_value in vars_config.items():
            step._vars[var_name] = instantiate_vars(var_name, var_value, step, df)
        columns_config = step_config.get("columns") or {}
        for action_name, action_config in columns_config.items():
            if action_name == "add" and action_config is not None:
                step._add_columns(df, action_config)
            elif action_name == "drop" and action_config is not None:
                step._drop_columns(df, action_config, entities, column_types, fnreport)
            elif action_name == "rename" and action_config is not None:
                step._rename_columns(df, action_config)
        rows_config = step_config.get("rows") or {}
        for action_name, action_config in rows_config.items():
            if action_name == "drop" and action_config is not None:
                step._drop_rows(df, action_config)
            elif action_name == "update" and action_config is not None:
                step._update_rows(df, action_config, entities, column_types, progress, fnreport)
        df = df.reset_index(drop=True)
        return df


def instantiate_vars(var_name: str, var_value: dict | list | str, step: Step, df: pd.DataFrame) -> Any:
    """
    Traverse the var value, render any template strings and attempt to eval
    as more specific types.

    This is recursive, but users are not expected to add large, deeply
    nested structures to the vars list.

    A template may reference running vars list, but a given var may or may not be instantiated
    in time for it to be available.
    """
    if isinstance(var_value, str):
        try:
            var_value = step.do_make_template(var_value).render(data=df, vars=step._vars)
        except TemplateSyntaxError:
            # If it cannot be rendered as template, take the literal string.
            pass
        except TemplateError as e:
            # If it's valid jinja syntax but some other error occurred, assume user error.
            raise Exception(
                f"Error building jinja template for var '{var_name}': '{var_value}': {e}",
                error_id="param",
            )

        try:
            var_value = ast.literal_eval(var_value)
        except (ValueError, TypeError, SyntaxError):
            # Assume just a regular string. Can also raise MemoryError and RecursionError,
            # but that would likely mean the user did not intend it to be used as a string
            # and should throw an error.
            pass
    elif isinstance(var_value, list):
        var_value = [instantiate_vars(var_name, elm, step, df) for elm in var_value]
    elif isinstance(var_value, dict):
        var_value = {k: instantiate_vars(var_name, v, step, df) for k, v in var_value.items()}

    return var_value


class Editor:
    def _config_globals(self, entity_extractor: EntityExtractor | None):
        globals_config = self.config.get("globals") or {}
        locales: list[str] | None = globals_config.get("locales")
        seed = globals_config.get("seed")
        if seed is None:
            seed = datetime.now().timestamp()
        random.seed(seed)
        self._chunksize = globals_config.setdefault("chunksize", CHUNKSIZE)
        self._entities = set(globals_config.setdefault("entities", DEFAULT_ENTITIES))
        self._env = Environment(
            locales,
            seed,
            globals_config=globals_config,
            entity_extractor=entity_extractor,
        )

    def __init__(self, config: dict[str, dict], entity_extractor: EntityExtractor | None):
        self.config = config
        self._config_globals(entity_extractor)

    @classmethod
    def load_yaml(cls, yaml_str: str) -> Editor:
        return cls(yaml.safe_load(yaml_str))

    def _process_df(
        self,
        df: pd.DataFrame,
        entities: dict[str, str | None],
        column_types: dict[str, str | None],
        fnreport: TransformFnAccounting | None = None,
    ) -> pd.DataFrame:
        progress = ProgressLog(30)
        progress.status.step_n_total = len(self.config["steps"])
        for stepn, step_config in enumerate(self.config["steps"]):
            progress.status.step_n = stepn
            logger.user.info(
                f"Executing transform step : {stepn} / {len(self.config['steps'])}",
            )
            df = Step.execute(df, entities, column_types, step_config, self._env, progress, fnreport)
        return df

    def process_df(
        self,
        df: pd.DataFrame,
        entities: dict[str, str | None],
        column_types: dict[str, str | None],
        fnreport: TransformFnAccounting | None = None,
    ) -> pd.DataFrame:
        df_copy = df.copy(deep=True)
        return self._process_df(df_copy, entities, column_types, fnreport)
