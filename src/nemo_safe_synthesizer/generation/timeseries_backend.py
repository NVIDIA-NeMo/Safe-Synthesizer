# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time-series generation backend with chronological validation."""

from __future__ import annotations

import calendar
import json
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Self

import pandas as pd
from vllm.inputs.llm import TokensPrompt
from vllm.sampling_params import SamplingParams

from .. import utils
from ..config import SafeSynthesizerParameters
from ..data_processing.record_utils import ParsedRecord, _parse_timestamp_to_seconds
from ..defaults import FIXED_RUNTIME_GENERATE_ARGS, LOG_DASHES, PSEUDO_GROUP_COLUMN
from ..errors import GenerationError
from ..generation.batch import Batch
from ..generation.results import GenerateJobResults, GenerationBatches, GenerationStatus
from ..generation.timeseries_prompting import (
    build_partial_record_prefix,
    build_record_history,
    build_training_compatible_prompt_token_ids,
)
from ..generation.vllm_backend import VllmBackend
from ..llm.metadata import ModelMetadata, TimeSeriesGroupValue, TimeseriesMetadata
from ..observability import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class _ResolvedTimeseriesSettings:
    """Generation-required values produced by time-series preprocessing."""

    timestamp_column: str
    group_column: str
    start_timestamp: str | int
    stop_timestamp: str | int | None
    timestamp_interval_seconds: int | None
    timestamp_format: str
    timeseries_metadata: TimeseriesMetadata | None

    @classmethod
    def from_config(cls, config: SafeSynthesizerParameters, metadata: ModelMetadata) -> Self:
        """Validate and extract the time-series values required by generation."""
        timestamp_column = config.time_series.timestamp_column
        group_column = config.data.group_training_examples_by
        start_timestamp = config.time_series.start_timestamp
        if timestamp_column is None or group_column is None or start_timestamp is None:
            missing = [
                name
                for name, value in (
                    ("timestamp column", timestamp_column),
                    ("group column", group_column),
                    ("start timestamp", start_timestamp),
                )
                if value is None
            ]
            raise GenerationError(
                f"Time-series generation requires resolved {', '.join(missing)}. "
                "Run time-series preprocessing before generation."
            )
        return cls(
            timestamp_column=timestamp_column,
            group_column=group_column,
            start_timestamp=start_timestamp,
            stop_timestamp=config.time_series.stop_timestamp,
            timestamp_interval_seconds=config.time_series.timestamp_interval_seconds,
            timestamp_format=config.time_series.timestamp_format or "",
            timeseries_metadata=metadata.timeseries_metadata,
        )


@dataclass
class ProgressSnapshot:
    """Snapshot configuration for saving partial generation results at progress milestones."""

    label: str
    """Human-readable label for the milestone (e.g. ``"50"``)."""

    threshold: int
    """Record or group count that triggers this snapshot."""

    path: Path
    """File path where the snapshot CSV will be written."""

    saved: bool = field(default=False)
    """Whether this snapshot has already been written to disk."""


@dataclass
class RecordPromptState:
    """Mutable prefix/history state used to build record prompts."""

    prefix: str
    """Incomplete first record used until the first generated record is accepted."""

    history: list[ParsedRecord] = field(default_factory=list)
    """Exact accepted record text used as rolling prompt history."""

    using_prefix: bool = True
    """Whether generation is still completing the initial record prefix."""

    @property
    def prompt_segments(self) -> str | Sequence[str]:
        """Return the prefix or individually encoded history records."""
        if self.using_prefix:
            return self.prefix
        return [f"{record.text}\n" for record in self.history]

    @property
    def completion_prefix(self) -> str:
        """Return bytes prepended to completions during the prefix phase."""
        return self.prefix if self.using_prefix else ""

    @property
    def history_text(self) -> str:
        """Return history as newline-terminated training-compatible JSONL."""
        return build_record_history(self.history)

    def add_history(self, records: Sequence[ParsedRecord], *, max_records: int) -> None:
        """Accept records, switch from prefix to history, and clamp the window."""
        if not records:
            return
        self.using_prefix = False
        self.history.extend(records)
        if len(self.history) > max_records:
            self.history = self.history[-max_records:]


@dataclass
class GroupState:
    """Mutable state for tracking a single group during parallel generation.

    Each group maintains its own sliding-window context, timestamp cursor,
    and retry counters so that multiple groups can be generated in parallel
    while tracking progress independently.
    """

    group_id: TimeSeriesGroupValue
    """Unique identifier for this group (e.g., device ID, customer ID)."""

    group_ordinal: int
    """Stable one-based position in the saved group registry, used in diagnostics."""

    prompt_state: RecordPromptState
    """Prefix/history container for building prompts and parsing completions."""

    expected_records: int = 0
    """Target record count, calculated from ``(stop_timestamp - start_timestamp) / interval_seconds``."""

    last_timestamp_seconds: int | None = None
    """Timestamp (in seconds) of the most recently generated record, used for chronological validation."""

    last_source_timestamp_seconds: int | None = None
    """Flexible mode: source timestamp (in seconds) of the most recently accepted record."""

    batches: list[Batch] = field(default_factory=list, repr=False)
    """Batches whose accepted rows belong to this group; invalidated if the group fails."""

    low_valid_fraction_count: int = 0
    """Consecutive batches with high invalid fraction.  Triggers group failure after ``patience`` is exceeded."""

    completed: bool = False
    """Whether this group has reached the stop timestamp."""

    failed: bool = False
    """Whether this group failed (e.g., too many retries without progress)."""

    total_valid_records: int = 0
    """Cumulative count of valid records generated for this group."""

    total_invalid_records: int = 0
    """Cumulative count of invalid records generated for this group."""

    no_progress_count: int = 0
    """Consecutive batches that did not advance the accepted timestamp."""

    termination_reason: str | None = None
    """Final flexible time-series outcome: ``terminal``, ``cap``, or ``failure``."""

    total_prompts: int = 0
    """Number of generation prompts attempted for this group."""

    total_prompt_tokens: int = 0
    """Prompt tokens submitted once per generation request, including retries."""

    total_completion_tokens: int = 0
    """Completion tokens attempted for this group, including discarded rows."""


class GroupProcessingResult(Enum):
    """Result of processing a generation batch for a single group.

    Used by ``_process_group_result`` to signal whether a group should
    remain active, be marked complete, or be removed due to failure.
    """

    IN_PROGRESS = auto()
    """Group continues; batch should be added to the accumulator."""

    COMPLETED = auto()
    """Group reached the stop timestamp; remove from active processing."""

    FAILED = auto()
    """Group failed after exhausting a consecutive retry condition."""


class TimeseriesBackend(VllmBackend):
    """Time-series aware generator that enforces chronological constraints.

    This backend extends VllmBackend to generate synthetic time-series data with
    strict chronological ordering. It uses a sliding window approach where recently
    generated records are used as history for subsequent generation,
    ensuring temporal continuity.

    Key Concepts:
        - Deterministic Time-Range Generation: For fixed-shape training data, the number of records generated is
          determined by the configured time range and interval, not by a target
          count. Specifically: (stop_timestamp - start_timestamp) / interval_seconds.
          The ``config.generation.num_records`` parameter is used only for progress
          tracking, not to limit output.
        - Flexible Marker Generation: For automatically routed training data,
          generated rows must have contiguous sequence indices. A final-row marker
          completes the group, while the largest source-group length provides a
          safety cap. Internal control columns are removed from final output.
        - Sliding Window: The backend maintains a window of recent records
          (controlled by ``_history_window_size``) that are included in each prompt
          to provide context for the LLM, ensuring generated records follow the
          established patterns and timestamps.
        - Parallel Group Generation: Multiple time-series groups (e.g., different
          devices, customers) are processed in parallel batches for efficiency.
          Even single-sequence data uses this path (treated as 1 group via a
          pseudo-group column added during preprocessing). Groups are the same as
          those seen during training (from ``model_metadata.timeseries_group_values``).
        - Chronological Validation: Each generated record must continue from the
          previous timestamp at the expected interval. Out-of-order records are
          marked invalid.

    Generation Flow (parallel group mode):
        1. Initialize GroupState for each group with a partial first record
        2. While groups remain pending or active:
           a. Fill active slots with pending groups (up to max_groups_per_batch)
           b. Build prompts for all active groups using their prefix or history
           c. Generate completions for all prompts in a single LLM batch call
           d. Process LLM outputs into per-group Batch objects
           e. For each group:
              - Validate chronological order against group's last timestamp
              - Retain the response with the most valid records (discard others)
              - Apply data actions and discard records rejected by post-processing
              - Update group state from accepted records (history, last_timestamp)
              - Check if an accepted record reached the stop timestamp
              - Track invalid output and timestamp progress; fail after consecutive retries
           f. Remove completed/failed groups from active list
           g. Save progress snapshots if thresholds are met
           h. Log per-group progress summary

    Stopping Conditions:
        Generation stops when all groups finish (either completed or failed). Individual
        groups and the overall generation can stop for different reasons:

        Per-Group Stopping:
            - Completion (success): A deterministic group completes at
              ``_stop_timestamp_value``. A flexible group completes at its
              final-row marker or source-derived record cap.
            - Failure (low valid fraction or no progress): A group fails after
              ``config.generation.patience`` consecutive batches where either the
              invalid record fraction remains above the configured threshold or
              no accepted timestamp advances. Failed groups are not retried, and
              rows they accepted in earlier batches are discarded from the output.

        Global Stopping:
            - Natural completion: Generation ends when both the pending groups
              queue and active groups list are empty (all groups processed).
            - Per-group retry state is authoritative; batch-level stop signals
              from `GenerationBatches` are cleared while groups remain active.

        The final status is ``COMPLETE`` only when every group completed and
        ``INCOMPLETE`` when any group failed. ``num_records`` does not affect it.

    Attributes:
        _samples_per_prompt (int): Minimum number of completion samples per
            prompt. One sample per group is kept per batch.
        _max_prompts_per_batch (int): Maximum number of prompts to include in a
            single LLM generation call. Controls parallelism. Default: 100.
        _history_window_size (int): Number of recent records to include in the
            sliding prompt history. Default: 3.
        _time_column (str): Name of the timestamp column in the data.
        _time_format (str): Format string for parsing timestamps (strptime format),
            or "elapsed_seconds" for numeric elapsed time.
        _is_elapsed_time (bool): True if timestamps are numeric elapsed seconds.
        _start_timestamp_value: Starting timestamp for generation range.
        _stop_timestamp_value: Ending timestamp for generation range. Generation
            stops when a record reaches or exceeds this timestamp.
        _timestamp_interval_seconds (int | None): Expected interval between
            consecutive timestamps. Used for chronological validation.
        _group_column (str): Column name used to group time-series data.
        _group_ordinals (dict[TimeSeriesGroupValue, int]): Mapping of group IDs
            to stable, non-sensitive positions used in diagnostics.
        _group_prefixes (dict[TimeSeriesGroupValue, str]): Mapping of group IDs to
            incomplete first records used to start generation.
        _groups (list[TimeSeriesGroupValue]): Typed group IDs to generate.
        _timeseries_metadata (TimeseriesMetadata | None): Resolved
            representation and source schema from training.
        _flexible_metadata (FlexibleTimeseriesMetadata | None): Resolved
            marker-based generation settings; ``None`` for deterministic routing.
    """

    def __init__(self, config: SafeSynthesizerParameters, model_metadata: ModelMetadata, **kwargs):
        settings = _ResolvedTimeseriesSettings.from_config(config, model_metadata)
        super().__init__(config, model_metadata, **kwargs)

        self._timeseries_metadata = settings.timeseries_metadata
        flexible_metadata = self._timeseries_metadata.flexible if self._timeseries_metadata is not None else None
        self._flexible_metadata = flexible_metadata
        self._flexible_timeseries = flexible_metadata is not None
        self._sequence_index_column = flexible_metadata.index_column if flexible_metadata is not None else ""
        self._sequence_marker_column = flexible_metadata.marker_column if flexible_metadata is not None else ""
        self._samples_per_prompt = 5
        self._max_prompts_per_batch = 100  # max prompts per batch for parallel group generation
        self._history_window_size = 3
        self._time_column = settings.timestamp_column
        self._time_format = settings.timestamp_format
        self._is_elapsed_time = self._time_format == "elapsed_seconds"
        self._start_timestamp_value = settings.start_timestamp
        self._stop_timestamp_value = settings.stop_timestamp
        self._timestamp_interval_seconds = settings.timestamp_interval_seconds
        self._group_column = settings.group_column
        self._raw_generations_path = self.workdir.generate.path / "raw_generations.jsonl"
        self._flexible_metrics_path = self.workdir.generate.path / "flexible_timeseries_metrics.json"
        self._internal_output_path = self.workdir.generate.path / "synthetic_data_internal.csv"
        self._sequence_group_states: dict[TimeSeriesGroupValue, GroupState] = {}

        group_values = self.model_metadata.timeseries_group_values
        if not group_values:
            raise GenerationError("The saved artifact has no time-series group registry. Retrain it before generating.")

        self._groups: list[TimeSeriesGroupValue] = list(group_values)
        self._group_ordinals = {group_id: ordinal for ordinal, group_id in enumerate(self._groups, start=1)}
        self._group_prefixes: dict[TimeSeriesGroupValue, str] = {
            group_id: build_partial_record_prefix(
                columns=self.columns,
                schema=self.schema,
                group_column=self._group_column,
                group_id=group_id,
                timestamp_column=self._time_column,
                start_timestamp=self._start_timestamp_value,
            )
            for group_id in self._groups
        }

    def _get_prompt_token_count(self) -> int:
        """Return the longest initial prompt length for ``SamplingParams``.

        Time-series generation prepends a per-group partial record to the
        templated prompt. This initial value sizes the base sampling parameters;
        each parallel batch applies a second clamp using its current rolling
        prompt lengths.

        Returns zero when the engine has not yet been initialized.
        """
        if self._prompt_token_count is not None:
            return self._prompt_token_count
        if self.llm is None:
            return 0
        tokenizer = self.llm.get_tokenizer()
        self._prompt_token_count = max(
            (len(self._build_prompt_token_ids(prefix)) for prefix in self._group_prefixes.values()),
            default=len(
                build_training_compatible_prompt_token_ids(
                    tokenizer=tokenizer,
                    prompt_config=self.model_metadata.prompt_config,
                    prompt=self.prompt,
                    record_context="",
                )
            ),
        )
        return self._prompt_token_count

    def _build_progress_snapshots(self, total: int, is_group_based: bool = False) -> list[ProgressSnapshot]:
        """Build progress snapshots for saving intermediate results.

        Args:
            total: Total count (number of groups if is_group_based, else num_records).
            is_group_based: If True, snapshots are based on group milestones.

        Returns:
            List of ProgressSnapshot objects.
        """
        if total <= 0:
            return []
        snapshots: list[ProgressSnapshot] = []
        seen_thresholds: set[int] = set()
        for fraction in (0.25, 0.5, 0.75):
            threshold = max(1, math.ceil(total * fraction))
            if threshold in seen_thresholds:
                continue
            seen_thresholds.add(threshold)
            label = f"{int(fraction * 100)}"
            suffix = "groups" if is_group_based else "records"
            snapshots.append(
                ProgressSnapshot(
                    label=label,
                    threshold=threshold,
                    path=self.model_metadata.adapter_path / f"generated_partial_{label}pct_{suffix}.csv",
                )
            )
        return snapshots

    def _write_progress_snapshot(
        self, batches: GenerationBatches, snapshot: ProgressSnapshot, is_group_based: bool = False
    ) -> None:
        """Write a progress snapshot to disk.

        Args:
            batches: The GenerationBatches object containing the records.
            snapshot: The snapshot configuration to save.
            is_group_based: If True, save all records (no max_num_records limit).
        """
        try:
            # For group-based snapshots, save all records generated so far
            # For record-based snapshots, limit to threshold records
            max_records = None if is_group_based else snapshot.threshold
            df = batches.to_dataframe(self.columns, max_num_records=max_records)
            # Sort by group and timestamp for consistent output
            df = self._sort_dataframe(df)
        except Exception:
            logger.exception(f"Failed to build DataFrame for {snapshot.label}% snapshot")
            return

        if df.empty:
            return

        snapshot.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_csv(snapshot.path, index=False)
        except Exception:
            logger.exception(f"Failed to save partial generation output to {snapshot.path.as_posix()}")
            return

        snapshot.saved = True
        snapshot_type = "groups" if is_group_based else "records"
        logger.info(
            f"Saved partial generation output ({snapshot.label}% {snapshot_type}) to {snapshot.path.as_posix()}",
        )

    def _maybe_save_progress_snapshots(
        self,
        batches: GenerationBatches,
        snapshots: list[ProgressSnapshot],
        current_count: int | None = None,
        is_group_based: bool = False,
    ) -> None:
        """Check and save progress snapshots if thresholds are met.

        Args:
            batches: The GenerationBatches object containing the records.
            snapshots: List of snapshots to check.
            current_count: Current progress count. If None, uses batches.num_valid_records.
            is_group_based: If True, snapshots are based on group milestones.
        """
        if not snapshots:
            return
        count = current_count if current_count is not None else batches.num_valid_records
        for snapshot in snapshots:
            if snapshot.saved or count < snapshot.threshold:
                continue
            self._write_progress_snapshot(batches, snapshot, is_group_based=is_group_based)

    def _build_prompt_token_ids(self, record_context: str | Sequence[str]) -> list[int]:
        """Build a training-compatible token prompt with record context."""
        if self.llm is None:
            raise GenerationError("The generation backend must be initialized before building prompts.")
        return build_training_compatible_prompt_token_ids(
            tokenizer=self.llm.get_tokenizer(),
            prompt_config=self.model_metadata.prompt_config,
            prompt=self.prompt,
            record_context=record_context,
        )

    def _parse_timestamp_seconds(self, timestamp_value: object) -> int | None:
        """Parse a timestamp value to seconds, returning None on failure.

        Uses the shared _parse_timestamp_to_seconds from record_utils but wraps
        exceptions to return None instead of raising.
        """
        if timestamp_value is None or self._time_format is None:
            return None

        try:
            return _parse_timestamp_to_seconds(timestamp_value, self._time_format)
        except (ValueError, TypeError):
            return None

    def _advance_expected_time(self, timestamp_seconds: int) -> int | None:
        """Return the next expected timestamp by adding the configured interval."""
        if self._timestamp_interval_seconds is None:
            return None
        return timestamp_seconds + self._timestamp_interval_seconds

    def _has_reached_stop_time(self, records: list[dict]) -> bool:
        """Return ``True`` if any record's timestamp meets or exceeds the stop timestamp."""
        if not records or self._stop_timestamp_value is None:
            return False
        stop_ts = self._parse_timestamp_seconds(self._stop_timestamp_value)
        if stop_ts is None:
            return False
        for record in records:
            ts = self._parse_timestamp_seconds(record.get(self._time_column))
            if ts is not None and ts >= stop_ts:
                return True
        return False

    def _init_group_state(self, group_id: TimeSeriesGroupValue) -> GroupState:
        """Initialize a GroupState for a given group.

        Args:
            group_id: The group identifier.

        Returns:
            A new GroupState initialized with the group's partial record.
        """
        group_ordinal = self._group_ordinals.get(group_id)
        if group_ordinal is None:
            raise GenerationError("Cannot initialize an unregistered time-series group.")
        try:
            prefix = self._group_prefixes[group_id]
        except KeyError as exc:
            raise GenerationError(f"No initial prefix was built for time-series group {group_ordinal}.") from exc

        # Calculate expected number of records: (stop - start) / interval + 1
        expected_records = self._compute_expected_records_per_group()

        state = GroupState(
            group_id=group_id,
            group_ordinal=group_ordinal,
            prompt_state=RecordPromptState(prefix=prefix),
            expected_records=expected_records,
        )
        return state

    def _compute_expected_records_per_group(self) -> int:
        """Compute expected number of records per group based on time range and interval.

        Returns:
            Expected number of records per group, or 0 if cannot be computed.
        """
        start_ts = self._parse_timestamp_seconds(self._start_timestamp_value)
        stop_ts = self._parse_timestamp_seconds(self._stop_timestamp_value)
        if start_ts is not None and start_ts == stop_ts:
            return 1
        if (
            start_ts is not None
            and stop_ts is not None
            and self._timestamp_interval_seconds
            and self._timestamp_interval_seconds > 0
        ):
            return ((stop_ts - start_ts) // self._timestamp_interval_seconds) + 1
        return 0

    def _compute_total_expected_records(self) -> int:
        """Compute total expected records across all groups.

        Returns:
            Total expected records (expected_per_group * num_groups).
        """
        expected_per_group = self._compute_expected_records_per_group()
        return expected_per_group * len(self._groups)

    def _is_chronological_for_group(self, records: list[dict], group_state: GroupState) -> bool:
        """Check if records continue from the group's last timestamp.

        Args:
            records: The records to validate.
            group_state: The state of the group.

        Returns:
            True if records continue the chronological sequence, False otherwise.
        """
        if not records:
            return False

        first_record = records[0]
        timestamp_seconds = self._parse_timestamp_seconds(first_record.get(self._time_column))
        if timestamp_seconds is None:
            return False

        if group_state.last_timestamp_seconds is not None:
            expected_ts = self._advance_expected_time(group_state.last_timestamp_seconds)
            if expected_ts is not None and timestamp_seconds != expected_ts:
                return False

        return True

    def _check_chronological_for_group(self, batch: Batch, group_state: GroupState) -> None:
        """Validate chronological ordering and demote out-of-order records.

        Responses whose first record does not continue from the group's
        last timestamp have all their valid records moved to
        ``invalid_records``.

        Args:
            batch: The batch containing responses to validate.
            group_state: Current state of the group (provides the last
                known timestamp).
        """
        error = ("Out-of-order time step", "TimeSeries")
        for response in batch._responses:
            if not response.valid_records:
                continue
            if self._is_chronological_for_group(response.valid_records, group_state):
                continue
            for record in response.records:
                if record.is_valid:
                    record.invalidate(error)

    def _trim_flexible_timeseries_records(
        self,
        state: GroupState,
        records: list[ParsedRecord],
    ) -> tuple[list[ParsedRecord], tuple[str, ParsedRecord] | None]:
        """Retain the structurally valid prefix of a flexible sequence.

        Records must match the active group and advance the generated sequence
        index contiguously. This method invalidates records after the first
        structural failure or stopping row. The stopping candidate remains
        tentative until data actions have accepted it. Source timestamps are
        checked after data actions in ``_validate_postprocessed_sequence_indices``.

        Args:
            state: Active generation state for the group.
            records: Parsed records to validate and trim in place.

        Returns:
            The retained prefix and its tentative marker- or cap-based stop.
        """
        invalidations, retained, accepted_row_stop = self._scan_flexible_prefix(state, records)
        for record, error in invalidations:
            record.invalidate(error)
        return retained, accepted_row_stop

    def _scan_flexible_prefix(
        self,
        state: GroupState,
        records: list[ParsedRecord],
    ) -> tuple[list[tuple[ParsedRecord, tuple[str, str]]], list[ParsedRecord], tuple[str, ParsedRecord] | None]:
        """Evaluate a flexible sequence prefix without modifying the records.

        Args:
            state: Active generation state for the group.
            records: Parsed records in generation order.

        Returns:
            The records to invalidate with their errors, the retained prefix,
            and its tentative marker- or cap-based stop.
        """
        metadata = self._flexible_metadata
        if metadata is None:
            return [], records, None

        invalidations: list[tuple[ParsedRecord, tuple[str, str]]] = []
        retained: list[ParsedRecord] = []
        accepted_row_stop: tuple[str, ParsedRecord] | None = None
        prefix_ended = False
        expected_index = 0 if state.last_timestamp_seconds is None else state.last_timestamp_seconds + 1
        trimmed_error = ("Generated row appears after the sequence end marker", "TimeSeries")
        group_error = ("Generated record group does not match the active time-series group", "TimeSeries")
        index_error = ("Generated sequence index is not a valid integer", "TimeSeries")

        for record in records:
            if prefix_ended:
                invalidations.append((record, trimmed_error))
                continue
            parsed = record.parsed
            if parsed is None:
                continue
            if not self._matches_group(parsed, state):
                invalidations.append((record, group_error))
                prefix_ended = True
                continue
            index_value = parsed.get(self._sequence_index_column)
            if not isinstance(index_value, int) or isinstance(index_value, bool):
                invalidations.append((record, index_error))
                prefix_ended = True
                continue
            if index_value != expected_index:
                invalidations.append(
                    (
                        record,
                        (
                            f"Generated sequence index {index_value!r} does not match expected index {expected_index}",
                            "TimeSeries",
                        ),
                    )
                )
                prefix_ended = True
                continue

            retained.append(record)
            expected_index += 1
            if parsed.get(self._sequence_marker_column) is True:
                accepted_row_stop = ("terminal", record)
                prefix_ended = True
            elif index_value >= metadata.max_records - 1:
                accepted_row_stop = ("cap", record)
                prefix_ended = True

        return invalidations, retained, accepted_row_stop

    def _select_flexible_response(self, state: GroupState, batch: Batch) -> int | None:
        """Choose which sampled response continues a flexible sequence.

        Prefers the first response whose valid prefix ends with an accepted
        final-row marker, then the first response with any valid prefix.
        Selecting by sample order rather than by the number of valid rows avoids
        favoring responses that continue past the point where the group should end.

        Args:
            state: Active generation state for the group.
            batch: The group's batch of sampled responses.

        Returns:
            The index of the selected response, or ``None`` when no response has
            a valid prefix.
        """
        first_valid_index: int | None = None
        for index, response in enumerate(batch._responses):
            valid_records = [record for record in response.records if record.is_valid and record.parsed is not None]
            _, retained, stop = self._scan_flexible_prefix(state, valid_records)
            if not retained:
                continue
            if stop is not None and stop[0] == "terminal":
                return index
            if first_valid_index is None:
                first_valid_index = index
        return first_valid_index

    def _matches_group(self, parsed: dict, state: GroupState) -> bool:
        """Whether a record belongs to the active group.

        The pseudo-group of an ungrouped dataset is never part of the schema or
        generated records, so every record of the single stream belongs to it.
        """
        if self._group_column == PSEUDO_GROUP_COLUMN:
            return True
        return parsed.get(self._group_column) == state.group_id

    def _source_timestamp_seconds(self, parsed: dict) -> int | None:
        """Parse the source timestamp of a post-processed record, or ``None`` when absent or unparseable.

        Post-processing restores the source representation, so values follow
        ``source_timestamp_format``; data actions may also return datetime objects.
        """
        metadata = self._flexible_metadata
        if metadata is None or metadata.source_timestamp_column is None or metadata.source_timestamp_format is None:
            return None
        value = parsed.get(metadata.source_timestamp_column)
        if isinstance(value, datetime):
            timestamp = pd.Timestamp(value)
            if timestamp.tzinfo is not None:
                return int(timestamp.timestamp())
            return calendar.timegm(timestamp.timetuple())
        try:
            return _parse_timestamp_to_seconds(value, metadata.source_timestamp_format)
        except (ValueError, TypeError, OverflowError):
            return None

    def _check_source_timestamp(
        self,
        parsed: dict,
        previous_seconds: int | None,
    ) -> tuple[int | None, tuple[str, str] | None]:
        """Parse a post-processed source timestamp and check it against the previous accepted row.

        Args:
            parsed: Post-processed record in the source representation.
            previous_seconds: Source timestamp of the previous accepted row in the group.

        Returns:
            The parsed timestamp in seconds, or ``None`` when the source data has
            no timestamp column, and an invalidation error when a check fails.
        """
        metadata = self._flexible_metadata
        if metadata is None or metadata.source_timestamp_column is None:
            return None, None
        seconds = self._source_timestamp_seconds(parsed)
        if seconds is None:
            return None, ("Generated source timestamp does not match the source timestamp format", "TimeSeries")
        if previous_seconds is None:
            return seconds, None
        if seconds < previous_seconds:
            return None, ("Generated source timestamp decreases within the group", "TimeSeries")
        interval = metadata.source_interval_seconds
        if interval is not None and seconds - previous_seconds != interval:
            return None, ("Generated source timestamp does not follow timestamp_interval_seconds", "TimeSeries")
        return seconds, None

    def _resolve_postprocessed_termination(
        self,
        accepted_row_stop: tuple[str, ParsedRecord] | None,
    ) -> str | None:
        """Resolve termination after data actions have accepted or rejected marker/cap rows."""
        if accepted_row_stop is None:
            return None
        reason, candidate = accepted_row_stop
        if not candidate.is_valid or candidate.parsed is None:
            return None
        if reason == "terminal":
            return reason if candidate.parsed.get(self._sequence_marker_column) is True else None
        if reason == "cap":
            metadata = self._flexible_metadata
            if metadata is None:
                return None
            index_value = candidate.parsed.get(self._sequence_index_column)
            if not isinstance(index_value, int) or isinstance(index_value, bool):
                return None
            return reason if index_value >= metadata.max_records - 1 else None
        return None

    def _validate_postprocessed_group_identity(
        self,
        state: GroupState,
        records: list[ParsedRecord],
    ) -> None:
        """Invalidate accepted rows whose postprocessed group no longer matches the active stream."""
        error = ("Postprocessed record group does not match the active time-series group", "TimeSeries")
        for record in records:
            if record.is_valid and record.parsed is not None and not self._matches_group(record.parsed, state):
                record.invalidate(error)

    def _validate_postprocessed_sequence_indices(
        self,
        state: GroupState,
        records: list[ParsedRecord],
    ) -> None:
        """Retain only the contiguous, chronological post-processed sequence prefix.

        Each accepted row must advance the sequence index by one. When the source
        data has a timestamp column, its post-processed value must also be
        non-decreasing and follow any asserted interval. Checks run after data
        actions so they apply to the values that are actually returned.
        """
        if self._flexible_metadata is None:
            return
        expected_index = 0 if state.last_timestamp_seconds is None else state.last_timestamp_seconds + 1
        previous_source_seconds = state.last_source_timestamp_seconds
        prefix_ended = False
        for record in records:
            if prefix_ended or not record.is_valid or record.parsed is None:
                prefix_ended = True
                if record.is_valid:
                    record.invalidate(("Generated row follows an invalid sequence row", "TimeSeries"))
                continue
            index_value = record.parsed.get(self._sequence_index_column)
            if not isinstance(index_value, int) or isinstance(index_value, bool) or index_value != expected_index:
                record.invalidate(
                    (
                        f"Postprocessed sequence index {index_value!r} does not match expected index {expected_index}",
                        "TimeSeries",
                    )
                )
                prefix_ended = True
                continue
            source_seconds, source_error = self._check_source_timestamp(record.parsed, previous_source_seconds)
            if source_error is not None:
                record.invalidate(source_error)
                prefix_ended = True
                continue
            expected_index += 1
            if source_seconds is not None:
                previous_source_seconds = source_seconds

    @staticmethod
    def _in_progress_path(path: Path) -> Path:
        """Return the sibling path an artifact is written to until generation finishes."""
        return path.with_name(f"{path.name}.tmp")

    @property
    def _raw_generations_tmp_path(self) -> Path:
        """In-progress raw completion log, moved to ``_raw_generations_path`` when generation finishes."""
        return self._in_progress_path(self._raw_generations_path)

    @property
    def _flexible_artifact_paths(self) -> tuple[Path, Path, Path]:
        """Final paths of the flexible artifacts that are replaced together."""
        return self._raw_generations_path, self._flexible_metrics_path, self._internal_output_path

    def _prepare_flexible_timeseries_artifacts(self) -> None:
        """Start an empty in-progress raw completion log and clear stale in-progress artifacts.

        Like other generation outputs, artifacts from a previous run in the same
        workdir are left in place until this run finishes and overwrites them.
        """
        if not self._flexible_timeseries:
            return
        for path in self._flexible_artifact_paths:
            self._in_progress_path(path).unlink(missing_ok=True)
        self._raw_generations_tmp_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_generations_tmp_path.write_text("", encoding="utf-8")

    def _finalize_flexible_timeseries_artifacts(self) -> None:
        """Replace the previous run's flexible artifacts with this run's artifacts.

        All three artifacts are written to in-progress paths first, so a run
        that fails before this point never leaves files from two runs side by side.
        """
        if not self._flexible_timeseries:
            return
        for path in self._flexible_artifact_paths:
            in_progress = self._in_progress_path(path)
            if in_progress.exists():
                in_progress.replace(path)

    def _write_raw_completion(
        self,
        state: GroupState,
        *,
        completion_text: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str,
    ) -> None:
        """Append one unparsed model completion to the flexible time-series audit log."""
        if not self._flexible_timeseries:
            return
        payload = {
            "group": state.group_id,
            "group_ordinal": state.group_ordinal,
            "prompt": state.total_prompts,
            "completion": completion_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "finish_reason": finish_reason,
        }
        with self._raw_generations_tmp_path.open("a", encoding="utf-8") as raw_file:
            raw_file.write(json.dumps(payload, ensure_ascii=False))
            raw_file.write("\n")

    def _resolve_timeseries_status(self) -> GenerationStatus:
        """Return ``COMPLETE`` only when every group finished; ``num_records`` does not apply."""
        states = self._sequence_group_states.values()
        if states and all(state.completed and not state.failed for state in states):
            return GenerationStatus.COMPLETE
        return GenerationStatus.INCOMPLETE

    def _discard_failed_group_records(self) -> None:
        """Invalidate accepted rows of failed groups so they are excluded from output."""
        error = ("Time-series group failed before completing", "TimeSeries")
        failed_states = [state for state in self._sequence_group_states.values() if state.failed]
        if failed_states:
            logger.warning(
                "Discarding rows from failed time-series groups; generation is incomplete.",
                extra={
                    "failed_groups": len(failed_states),
                    "total_groups": len(self._sequence_group_states),
                    "failed_group_ordinals": [state.group_ordinal for state in failed_states],
                },
            )
        records = (
            record
            for state in failed_states
            for batch in state.batches
            for response in batch._responses
            for record in response.records
        )
        for record in records:
            if record.is_valid:
                record.invalidate(error)

    def _write_flexible_timeseries_metrics(self) -> None:
        """Write auditable per-group and aggregate flexible time-series metrics."""
        metadata = self._flexible_metadata
        if metadata is None:
            return
        groups = [
            {
                "group": state.group_id,
                "group_ordinal": state.group_ordinal,
                "reason": state.termination_reason or "failure",
                "rows": 0 if state.failed else state.total_valid_records,
                "prompts": state.total_prompts,
                "prompt_tokens": state.total_prompt_tokens,
                "completion_tokens": state.total_completion_tokens,
                "total_tokens": state.total_prompt_tokens + state.total_completion_tokens,
            }
            for state in self._sequence_group_states.values()
        ]
        reason_counts = {
            reason: sum(group["reason"] == reason for group in groups) for reason in ("terminal", "cap", "failure")
        }
        metrics = {
            "status": self._resolve_timeseries_status().value,
            "flexible_timeseries": True,
            "sequence_max_records": metadata.max_records,
            "groups": groups,
            "aggregate": {
                "groups": len(groups),
                "rows": sum(int(group["rows"]) for group in groups),
                "prompts": sum(int(group["prompts"]) for group in groups),
                "prompt_tokens": sum(int(group["prompt_tokens"]) for group in groups),
                "completion_tokens": sum(int(group["completion_tokens"]) for group in groups),
                "total_tokens": sum(int(group["total_tokens"]) for group in groups),
                "reason_counts": reason_counts,
            },
        }
        utils.write_json(metrics, self._in_progress_path(self._flexible_metrics_path), indent=2)

    def _update_group_state(self, group_state: GroupState, records: list[ParsedRecord]) -> None:
        """Update a group's state with new valid records.

        The history reuses each record's original ``text`` (the bytes
        the model emitted, which match the training serialization) as
        newline-terminated records. The prompt builder inserts the sequence BOS
        token directly before these bytes.

        Args:
            group_state: The group state to update.
            records: The new valid records (``parsed`` is set on each).
        """
        if not records:
            return

        group_state.prompt_state.add_history(records, max_records=self._history_window_size)

        # Update last timestamp
        last_parsed = records[-1].parsed or {}
        source_seconds = self._source_timestamp_seconds(last_parsed)
        if source_seconds is not None:
            group_state.last_source_timestamp_seconds = source_seconds
        timestamp_value = last_parsed.get(self._time_column) if self._time_column is not None else None
        timestamp_seconds = self._parse_timestamp_seconds(timestamp_value)
        if timestamp_seconds is not None:
            group_state.last_timestamp_seconds = timestamp_seconds

    def _sort_internal_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort accepted rows while preserving flexible time-series control columns."""
        if df.empty:
            return df
        sort_columns = [
            column for column in (self._group_column, self._time_column) if column is not None and column in df.columns
        ]
        return df.sort_values(by=sort_columns).reset_index(drop=True) if sort_columns else df

    def _write_internal_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Persist accepted flexible time-series rows before internal-column cleanup."""
        internal_df = self._sort_internal_dataframe(df)
        if self._flexible_timeseries:
            internal_path = self._in_progress_path(self._internal_output_path)
            internal_path.parent.mkdir(parents=True, exist_ok=True)
            internal_df.to_csv(internal_path, index=False)
        return internal_df

    def _sort_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort dataframe by group column then timestamp column.

        Also removes the pseudo-group column if present (used internally for
        single-sequence time series).

        Args:
            df: The dataframe to sort.

        Returns:
            Sorted dataframe with pseudo-group column removed.
        """
        if df.empty:
            return df

        df = self._sort_internal_dataframe(df)

        if self._flexible_timeseries:
            internal_columns = [self._sequence_index_column, self._sequence_marker_column]
            df = df.drop(columns=internal_columns, errors="ignore")

        # Remove pseudo-group column from output (it's only used internally)
        if PSEUDO_GROUP_COLUMN in df.columns:
            df = df.drop(columns=[PSEUDO_GROUP_COLUMN])

        source_columns = self._timeseries_metadata.source_columns if self._timeseries_metadata is not None else ()
        if source_columns:
            restored_columns = [column for column in source_columns if column in df.columns]
            extra_columns = [column for column in df.columns if column not in restored_columns]
            df = df.loc[:, [*restored_columns, *extra_columns]]

        return df

    def _build_modified_sampling_params(
        self,
        sampling_params: SamplingParams,
        num_active: int,
        max_prompt_tokens: int | None = None,
    ) -> tuple[SamplingParams, int]:
        """Build modified sampling params with dynamic samples per prompt.

        Args:
            sampling_params: Base sampling parameters.
            num_active: Number of active groups.
            max_prompt_tokens: Longest current prompt in the batch. When
                provided, generation is clamped to the remaining context.

        Returns:
            Tuple of (modified SamplingParams, effective_samples_per_prompt).

        Raises:
            GenerationError: If the rolling prompt leaves no context for
                generation.
        """
        effective_samples_per_prompt = min(
            10,
            max(self._samples_per_prompt, self._max_prompts_per_batch // num_active),
        )
        max_tokens = sampling_params.max_tokens
        if max_prompt_tokens is not None:
            remaining_context = self.model_metadata.max_seq_length - max_prompt_tokens
            if remaining_context <= 0:
                raise GenerationError(
                    "The time-series rolling prompt fills the model context window and leaves no room for "
                    "generation. Reduce the generated record width or retrain with a larger context window."
                )
            max_tokens = remaining_context if max_tokens is None else min(max_tokens, remaining_context)

        modified_params = SamplingParams(
            n=effective_samples_per_prompt,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            min_p=sampling_params.min_p,
            max_tokens=max_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            skip_special_tokens=sampling_params.skip_special_tokens,
            include_stop_str_in_output=sampling_params.include_stop_str_in_output,
            ignore_eos=sampling_params.ignore_eos,
            stop=sampling_params.stop,
            stop_token_ids=sampling_params.stop_token_ids,
            seed=sampling_params.seed,
        )

        return modified_params, effective_samples_per_prompt

    def _process_group_result(
        self,
        state: GroupState,
        batch: Batch,
        accepted_records: list[ParsedRecord],
        invalid_fraction_threshold: float,
        termination_reason: str | None = None,
    ) -> GroupProcessingResult:
        """Update a group from records accepted after post-processing.

        Args:
            state: The group state.
            batch: The batch containing results for this group.
            accepted_records: Retained records that passed data actions.
            invalid_fraction_threshold: Threshold for invalid fraction.
            termination_reason: Explicit flexible time-series stop reason, when detected.

        Returns:
            GroupProcessingResult enum indicating the group's status.
        """
        previous_timestamp_seconds = state.last_timestamp_seconds
        reached_stop = self._has_reached_stop_time(
            [record.parsed for record in accepted_records if record.parsed is not None]
        )
        self._update_group_state(state, accepted_records)
        made_progress = state.last_timestamp_seconds is not None and (
            previous_timestamp_seconds is None or state.last_timestamp_seconds > previous_timestamp_seconds
        )

        state.total_valid_records += batch.num_valid_records
        state.total_invalid_records += batch.num_invalid_records

        if made_progress:
            state.no_progress_count = 0
        else:
            state.no_progress_count += 1

        if termination_reason is not None:
            state.completed = True
            state.termination_reason = termination_reason
            return GroupProcessingResult.COMPLETED

        if reached_stop:
            state.completed = True
            return GroupProcessingResult.COMPLETED

        patience = self.config.generation.patience

        # Check if batch has high invalid fraction
        invalid_fraction = 1.0 - batch.valid_record_fraction
        if invalid_fraction >= invalid_fraction_threshold:
            state.low_valid_fraction_count += 1

            if batch.num_valid_records == 0:
                logger.warning(
                    "Time-series group batch produced no valid records.",
                    extra={
                        "group_ordinal": state.group_ordinal,
                        "retry_count": state.low_valid_fraction_count,
                        "patience": patience,
                    },
                )
            else:
                logger.warning(
                    "Time-series group batch has a high invalid fraction.",
                    extra={
                        "group_ordinal": state.group_ordinal,
                        "invalid_fraction": invalid_fraction,
                        "invalid_fraction_threshold": invalid_fraction_threshold,
                        "retry_count": state.low_valid_fraction_count,
                        "patience": patience,
                    },
                )

            if state.low_valid_fraction_count >= patience:
                state.failed = True
                state.termination_reason = "failure"
                logger.warning(
                    "Time-series group skipped after consecutive batches with a high invalid fraction.",
                    extra={
                        "group_ordinal": state.group_ordinal,
                        "retry_count": state.low_valid_fraction_count,
                        "patience": patience,
                        "invalid_fraction_threshold": invalid_fraction_threshold,
                    },
                )
                return GroupProcessingResult.FAILED
        else:
            state.low_valid_fraction_count = 0

        if state.no_progress_count >= patience:
            state.failed = True
            state.termination_reason = "failure"
            logger.warning(
                "Time-series group skipped after consecutive batches without timestamp progress.",
                extra={
                    "group_ordinal": state.group_ordinal,
                    "no_progress_count": state.no_progress_count,
                    "patience": patience,
                },
            )
            return GroupProcessingResult.FAILED

        return GroupProcessingResult.IN_PROGRESS

    def _log_parallel_batch_summary(
        self,
        active_states: list[GroupState],
        group_batches: dict[TimeSeriesGroupValue, Batch],
        groups_completed: int,
        batches: GenerationBatches,
        duration: float,
        effective_samples_per_prompt: int,
    ) -> None:
        """Log progress summary for parallel batch.

        Args:
            active_states: Currently active group states.
            group_batches: Batches for each group.
            groups_completed: Number of completed groups.
            batches: The GenerationBatches accumulator.
            duration: Time taken for this batch.
            effective_samples_per_prompt: Samples per prompt used.
        """
        num_active = len(active_states)
        total_batch_records = sum(b.num_valid_records for b in group_batches.values())
        total_prompts_used = num_active * effective_samples_per_prompt
        records_per_second = 0 if duration == 0 else total_batch_records / duration
        duration_string = f"{duration:.1f}s" if duration < 120 else f"{duration / 60:.1f}min"

        # Build per-group progress summary
        group_progress_lines = []
        for state in active_states:
            batch = group_batches[state.group_id]
            batch_valid_rate = batch.valid_record_fraction
            progress_pct = (
                (state.total_valid_records / state.expected_records * 100) if state.expected_records > 0 else 0.0
            )
            status = "✓" if batch.num_valid_records > 0 else "✗"
            group_progress_lines.append(
                f"  {status} group {state.group_ordinal}: "
                f"+{batch.num_valid_records} ({batch_valid_rate:.0%} valid), "
                f"progress={state.total_valid_records}/{state.expected_records} ({progress_pct:.1f}%)"
            )
        group_progress_str = "\n".join(group_progress_lines)

        logger.info(
            f"Parallel batch summary:\n"
            f"{LOG_DASHES}\n"
            f"Batch time: {duration_string}\n"
            f"Speed: {records_per_second:.1f} records/sec\n"
            f"Groups: {num_active} active, {groups_completed}/{len(self._groups)} completed\n"
            f"Samples/prompt: {effective_samples_per_prompt} (total prompts: {total_prompts_used})\n"
            f"Per-group progress this batch:\n{group_progress_str}\n"
            f"Total records: {batches.num_valid_records}\n"
            f"{LOG_DASHES}",
        )

    def _generate_parallel_groups(
        self,
        batches: GenerationBatches,
        sampling_params: SamplingParams,
        progress_snapshots: list[ProgressSnapshot],
    ) -> bool:
        """Generate records for multiple groups in parallel.

        This method processes multiple groups at once by generating prompts for
        multiple groups in a single batch. The maximum number of prompts per batch
        is controlled by `_max_prompts_per_batch`.

        Args:
            batches: The GenerationBatches object to accumulate results.
            sampling_params: Sampling parameters for generation.
            progress_snapshots: Progress snapshots for saving intermediate results (record-based).

        Returns:
            True if all groups completed successfully, False otherwise.
        """
        invalid_fraction_threshold = self.config.generation.invalid_fraction_threshold
        max_groups_per_batch = max(1, self._max_prompts_per_batch // self._samples_per_prompt)

        # Initialize states for all groups
        all_group_states: dict[TimeSeriesGroupValue, GroupState] = {
            group_id: self._init_group_state(group_id) for group_id in self._groups
        }
        self._sequence_group_states = all_group_states

        pending_groups = list(self._groups)
        active_states: list[GroupState] = []
        groups_completed = 0
        all_groups_succeeded = True

        logger.info(
            f"Starting parallel generation for {len(self._groups)} groups "
            f"(max {max_groups_per_batch} groups per batch, {self._samples_per_prompt} samples per prompt)",
        )

        while pending_groups or active_states:
            # Fill active slots with pending groups
            while len(active_states) < max_groups_per_batch and pending_groups:
                next_group = pending_groups.pop(0)
                state = all_group_states[next_group]
                active_states.append(state)
                logger.debug(
                    "Activated time-series group for parallel generation.",
                    extra={"group_ordinal": state.group_ordinal},
                )

            if not active_states:
                break

            start_time = time.perf_counter()

            # Build token prompts that reproduce the training BOS/EOS boundary.
            prompt_token_ids = [
                self._build_prompt_token_ids(state.prompt_state.prompt_segments) for state in active_states
            ]
            prompts = [TokensPrompt(prompt_token_ids=token_ids) for token_ids in prompt_token_ids]
            group_batches: dict[TimeSeriesGroupValue, Batch] = {
                state.group_id: Batch(processor=self.processor) for state in active_states
            }

            modified_params, effective_samples_per_prompt = self._build_modified_sampling_params(
                sampling_params,
                len(active_states),
                max_prompt_tokens=max(len(token_ids) for token_ids in prompt_token_ids),
            )
            for state, token_ids in zip(active_states, prompt_token_ids, strict=True):
                state.total_prompts += 1
                state.total_prompt_tokens += len(token_ids)

            # Generate for all prompts at once
            if self.llm is None:
                raise GenerationError("The generation backend must be initialized before generating.")
            outputs = self.llm.generate(
                prompts=prompts,
                sampling_params=modified_params,
                lora_request=self.lora_req,
            )

            # Process LLM outputs into batches
            for prompt_idx, output in enumerate(outputs):
                group_state = active_states[prompt_idx]
                batch = group_batches[group_state.group_id]
                for completion_idx, completion in enumerate(output.outputs):
                    finish_reason = str(completion.finish_reason or "unknown")
                    completion_tokens = len(completion.token_ids)
                    batch.finish_reasons[finish_reason] += 1
                    group_state.total_completion_tokens += completion_tokens
                    self._write_raw_completion(
                        group_state,
                        completion_text=completion.text,
                        prompt_tokens=len(prompt_token_ids[prompt_idx]) if completion_idx == 0 else 0,
                        completion_tokens=completion_tokens,
                        finish_reason=finish_reason,
                    )
                    completion_text = f"{group_state.prompt_state.completion_prefix}{completion.text}"
                    batch.process(completion_idx, completion_text, completion_tokens=completion_tokens)

            duration = time.perf_counter() - start_time

            # Process results for each group
            states_to_remove = []
            for state in active_states:
                batch = group_batches[state.group_id]
                if self.config.time_series.timestamp_interval_seconds is not None:
                    self._check_chronological_for_group(batch, state)
                preferred_index = self._select_flexible_response(state, batch) if self._flexible_timeseries else None
                retained_records = self._retain_single_valid_response(batch, preferred_index=preferred_index)
                retained_records, accepted_row_stop = self._trim_flexible_timeseries_records(state, retained_records)
                batches.postprocess_batch(batch, commit_history=False)
                self._validate_postprocessed_group_identity(state, retained_records)
                self._validate_postprocessed_sequence_indices(state, retained_records)
                batches.commit_history(batch)
                termination_reason = self._resolve_postprocessed_termination(accepted_row_stop)
                accepted_records = [
                    record for record in retained_records if record.is_valid and record.parsed is not None
                ]
                if termination_reason is None:
                    result = self._process_group_result(
                        state,
                        batch,
                        accepted_records,
                        invalid_fraction_threshold,
                    )
                else:
                    result = self._process_group_result(
                        state,
                        batch,
                        accepted_records,
                        invalid_fraction_threshold,
                        termination_reason=termination_reason,
                    )
                state.batches.append(batch)
                batches.add_batch(batch, apply_data_actions=False)
                # Time-series retries are tracked per group. Since completion is
                # determined from post-processed records, a batch-level zero-record
                # status can safely be cleared while group processing continues.
                batches.status = GenerationStatus.IN_PROGRESS
                if result == GroupProcessingResult.FAILED:
                    states_to_remove.append(state)
                    groups_completed += 1
                    all_groups_succeeded = False
                elif result == GroupProcessingResult.COMPLETED:
                    states_to_remove.append(state)
                    groups_completed += 1
                    completion_message = (
                        "Time-series group completed."
                        if self._flexible_timeseries
                        else "Time-series group completed after reaching the stop timestamp."
                    )
                    logger.info(
                        completion_message,
                        extra={
                            "group_ordinal": state.group_ordinal,
                            "groups_completed": groups_completed,
                            "total_groups": len(self._groups),
                        },
                    )

            # Remove completed/failed states from active list
            for state in states_to_remove:
                active_states.remove(state)

            # Check progress snapshots
            self._maybe_save_progress_snapshots(
                batches,
                progress_snapshots,
                current_count=batches.num_valid_records,
                is_group_based=False,
            )

            # Log progress summary
            self._log_parallel_batch_summary(
                active_states,
                group_batches,
                groups_completed,
                batches,
                duration,
                effective_samples_per_prompt,
            )

        return all_groups_succeeded

    def _retain_single_valid_response(
        self,
        batch: Batch,
        preferred_index: int | None = None,
    ) -> list[ParsedRecord]:
        """Retain one response, discarding all others.

        For time-series sliding window generation, only one response can be used
        per batch to maintain chronological continuity. This method keeps
        ``preferred_index`` when given, otherwise the response with the most
        valid records, and discards all other responses (both their valid and
        invalid records are cleared, and an error note is added to track that
        they were trimmed).

        Args:
            batch: The batch to retain the response from.
            preferred_index: Index of the response to keep, if already selected.

        Returns:
            The retained response's valid ``ParsedRecord`` objects, keeping
            both the parsed dicts and the original emitted text (the latter
            feeds the next prompt context).
        """
        final_records: list[ParsedRecord] = []

        max_valid_idx = preferred_index
        if max_valid_idx is None:
            # Find the index of the response with the most valid records.
            max_valid_count = -1
            for idx, response in enumerate(batch._responses):
                count = len(response.valid_records)
                if count > max_valid_count:
                    max_valid_count = count
                    max_valid_idx = idx

        trim_error = ("Extra response trimmed for sliding window", "TimeSeries")
        for idx, response in enumerate(batch._responses):
            if idx != max_valid_idx:
                # Drop all records from the trimmed response; replace with a
                # single synthetic marker record so the trim is visible in
                # error statistics without carrying stale text/token counts.
                response.records = [ParsedRecord(text="", error=trim_error)]
            else:
                final_records.extend(r for r in response.records if r.is_valid and r.parsed is not None)

        return final_records

    def generate(
        self,
        data_actions_fn: utils.DataActionsFn | None = None,
    ) -> GenerateJobResults:
        """Generate time-series tabular data using Nemo Safe Synthesizer.

        All time series are processed as groups (single-sequence is treated as 1 group
        via a pseudo-group column added during preprocessing). Fixed-shape groups stop
        at the configured time range. Automatically routed flexible groups stop at an
        accepted final-row marker or the maximum source-group length.

        Note:
            ``config.generation.num_records`` is used for progress tracking but
            does not limit time-series output or decide completion. The status is
            ``COMPLETE`` only when every group finishes; otherwise it is
            ``INCOMPLETE`` and rows from failed groups are discarded. Groups are
            the same as those seen during training in
            ``model_metadata.timeseries_group_values``.

        Args:
            data_actions_fn: Optional function that takes a DataFrame and returns a modified DataFrame.

        Returns:
            Generation results object, which includes a DataFrame of generated records.
        """
        self._prepare_flexible_timeseries_artifacts()
        generation_start = time.monotonic()
        num_records = self.config.generation.num_records

        sampling_params = SamplingParams(
            temperature=self.config.generation.temperature,
            repetition_penalty=self.config.generation.repetition_penalty,
            top_p=self.config.generation.top_p,
            top_k=FIXED_RUNTIME_GENERATE_ARGS["top_k"],
            min_p=FIXED_RUNTIME_GENERATE_ARGS["min_p"],
            max_tokens=self.model_metadata.generation_max_tokens_for(
                self._get_prompt_token_count(),
                multiplier=self.config.generation.max_tokens_multiplier,
            ),
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            ignore_eos=False,
        )

        batches = GenerationBatches(
            target_num_records=num_records,
            data_actions_fn=data_actions_fn,
        )

        # Use parallel group generation (single-sequence is just 1 group)
        num_groups = len(self._groups)

        # Compute total expected records across all groups for snapshot thresholds
        total_expected_records = self._compute_total_expected_records()
        progress_snapshots = self._build_progress_snapshots(total_expected_records, is_group_based=False)

        logger.info(
            f"Generating for {num_groups} groups using parallel generation "
            f"(total expected records: {total_expected_records})",
        )
        self._generate_parallel_groups(
            batches=batches,
            sampling_params=sampling_params,
            progress_snapshots=progress_snapshots,
        )

        self._discard_failed_group_records()
        batches.status = self._resolve_timeseries_status()
        batches.job_complete()
        batches.log_status()

        generation_time_sec = time.monotonic() - generation_start
        self.elapsed_time = generation_time_sec
        self.gen_results = GenerateJobResults.from_batches(
            batches=batches,
            columns=self.columns,
            max_num_records=None,  # Time-range based, not count-based
            elapsed_time=self.elapsed_time,
        )

        internal_df = self._write_internal_output(self.gen_results.df)
        # Sort by group and timestamp for consistent output (also removes pseudo-group column)
        self.gen_results.df = self._sort_dataframe(internal_df)
        self._write_flexible_timeseries_metrics()
        self._finalize_flexible_timeseries_artifacts()

        return self.gen_results
