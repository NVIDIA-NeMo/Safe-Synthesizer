# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Processed-record storage and selection for base-model prompting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .. import utils
from ..defaults import DEFAULT_EXCLUDE_COLUMNS
from ..errors import ParameterError
from .record_utils import records_to_jsonl

if TYPE_CHECKING:
    from ..config import SafeSynthesizerParameters
    from ..llm.metadata import ModelMetadata


def _stable_offset(value: object) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


class PromptRecordPool:
    """A persisted pool of post-preprocessing records used for prompt examples."""

    def __init__(self, records: pd.DataFrame, config: SafeSynthesizerParameters):
        self.records = records.reset_index(drop=True)
        self.config = config
        self._sample_once: pd.DataFrame | None = None
        self._prompt_config: Any | None = None

    @classmethod
    def load(cls, path: Path, config: SafeSynthesizerParameters) -> PromptRecordPool:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(pd.DataFrame(rows), config)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(records_to_jsonl(self.records), encoding="utf-8")

    @property
    def visible_columns(self) -> list[str]:
        excluded = set(DEFAULT_EXCLUDE_COLUMNS)
        return [str(column) for column in self.records.columns if column not in excluded]

    @property
    def count(self) -> int:
        return self.config.generation.num_in_context_records

    def _ordered(self, records: pd.DataFrame) -> pd.DataFrame:
        columns = [
            column
            for column in (
                self.config.data.group_training_examples_by,
                self.config.data.order_training_examples_by,
            )
            if column is not None and column in records.columns
        ]
        if not columns:
            return records.sort_index(kind="stable")
        return records.sort_values(columns, kind="stable")

    def _seed(self, prompt_index: int = 0, group_id: object | None = None) -> int:
        base = int(self.config.data.random_state or 0)
        strategy = self.config.generation.in_context_record_selection
        prompt_offset = prompt_index if strategy == "sample_per_prompt" else 0
        group_offset = _stable_offset(group_id) if group_id is not None else 0
        return (base + prompt_offset + group_offset) % (2**32)

    def _require_count(self, records: pd.DataFrame, context: str) -> None:
        if len(records) < self.count:
            raise ParameterError(
                f"generation.num_in_context_records={self.count} requires at least "
                f"{self.count} available records for {context}; found {len(records)}."
            )

    def select(self, prompt_index: int = 0, group_id: object | None = None) -> pd.DataFrame:
        """Select exactly the configured number of records for one prompt."""
        if self.count == 0:
            return self.records.iloc[0:0].copy()

        records = self.records
        group_column = self.config.data.group_training_examples_by
        if group_id is not None and group_column is not None and group_column in records.columns:
            records = records[records[group_column].astype(str) == str(group_id)]

        context = f"group {group_id!r}" if group_id is not None else "the prompt record pool"
        self._require_count(records, context)
        ordered = self._ordered(records)
        strategy = self.config.generation.in_context_record_selection

        if strategy == "first":
            return ordered.iloc[: self.count].copy()

        if self.config.time_series.is_timeseries:
            rng = np.random.default_rng(self._seed(prompt_index, group_id))
            start = int(rng.integers(0, len(ordered) - self.count + 1))
            return ordered.iloc[start : start + self.count].copy()

        if strategy == "sample_once" and group_id is None and self._sample_once is not None:
            return self._sample_once.copy()

        selected = records.sample(n=self.count, replace=False, random_state=self._seed(prompt_index, group_id))
        selected = self._ordered(selected)
        if strategy == "sample_once" and group_id is None:
            self._sample_once = selected.copy()
        return selected

    def _visible_records(self, records: pd.DataFrame) -> pd.DataFrame:
        return records.drop(columns=list(DEFAULT_EXCLUDE_COLUMNS), errors="ignore")

    def render(self, records: pd.DataFrame) -> str:
        """Render selected records in the completion shape for the active mode."""
        if records.empty:
            return ""

        visible = self._visible_records(records)
        group_column = self.config.data.group_training_examples_by
        if group_column is not None and not self.config.time_series.is_timeseries:
            prompt_config = self._prompt_config
            if prompt_config is None:
                raise RuntimeError("PromptRecordPool must be bound to model metadata before rendering grouped records")
            if group_column not in visible.columns:
                return f"{prompt_config.bos_token}{records_to_jsonl(visible)}{prompt_config.eos_token}"
            chunks: list[str] = []
            for _, group in visible.groupby(group_column, sort=False, dropna=False):
                chunks.append(f"{prompt_config.bos_token}{records_to_jsonl(group)}{prompt_config.eos_token}")
            return "".join(chunks)

        serialized = records_to_jsonl(visible)
        return f" {serialized}" if self.config.time_series.is_timeseries else serialized

    def bind_metadata(self, metadata: ModelMetadata) -> PromptRecordPool:
        self._prompt_config = metadata.prompt_config
        return self

    def prompt(self, metadata: ModelMetadata, prompt_index: int = 0) -> str:
        self.bind_metadata(metadata)
        prefill = self.render(self.select(prompt_index=prompt_index))
        return utils.create_schema_prompt(
            self.visible_columns,
            instruction=metadata.instruction,
            prompt_template=metadata.prompt_config.template,
            prefill=prefill,
        )

    def longest_candidate_prompt(self, metadata: ModelMetadata) -> str:
        """Build a conservative prompt candidate for context-window budgeting."""
        if self.count == 0:
            return self.prompt(metadata)
        visible = self._visible_records(self.records)
        lengths = visible.apply(lambda row: len(records_to_jsonl([row.to_dict()])), axis=1)
        longest = self.records.loc[lengths.nlargest(self.count).index]
        self.bind_metadata(metadata)
        prefill = self.render(self._ordered(longest))
        return utils.create_schema_prompt(
            self.visible_columns,
            instruction=metadata.instruction,
            prompt_template=metadata.prompt_config.template,
            prefill=prefill,
        )

    def max_prompt_token_count(self, metadata: ModelMetadata, tokenizer: Any) -> int:
        """Return a conservative upper bound for any selected non-time-series prompt."""
        base_prompt = utils.create_schema_prompt(
            self.visible_columns,
            instruction=metadata.instruction,
            prompt_template=metadata.prompt_config.template,
        )
        base_tokens = len(tokenizer.encode(base_prompt))
        if self.count == 0:
            return base_tokens

        visible = self._visible_records(self.records)
        record_token_counts = [
            len(tokenizer.encode(records_to_jsonl([row.to_dict()]))) for _, row in visible.iterrows()
        ]
        selected_tokens = sum(sorted(record_token_counts, reverse=True)[: self.count])

        delimiter_tokens = 0
        group_column = self.config.data.group_training_examples_by
        if group_column is not None and not self.config.time_series.is_timeseries:
            delimiter_tokens = self.count * (
                len(tokenizer.encode(metadata.prompt_config.bos_token))
                + len(tokenizer.encode(metadata.prompt_config.eos_token))
            )

        # BPE tokenization at concatenation boundaries is not strictly additive.
        return base_tokens + selected_tokens + delimiter_tokens + (2 * self.count)

    def timeseries_prefills(self, metadata: ModelMetadata) -> dict[str, str]:
        """Build one initial prefill per time-series group."""
        self.bind_metadata(metadata)
        group_column = self.config.data.group_training_examples_by
        if group_column is None or group_column not in self.records.columns:
            return {"0": self.render(self.select(group_id=None))}

        prefills: dict[str, str] = {}
        group_values = self._ordered(self.records)[group_column].drop_duplicates().tolist()
        for prompt_index, group_value in enumerate(group_values):
            prefills[str(group_value)] = self.render(self.select(prompt_index=prompt_index, group_id=group_value))
        return prefills
