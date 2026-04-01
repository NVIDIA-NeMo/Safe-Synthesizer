# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Time-series generation backend with chronological validation."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path

import pandas as pd
from vllm.sampling_params import SamplingParams

from .. import utils
from ..config import SafeSynthesizerParameters
from ..data_processing.record_utils import _parse_timestamp_to_seconds, extract_records_from_jsonl_string
from ..defaults import FIXED_RUNTIME_GENERATE_ARGS, LOG_DASHES, PSEUDO_GROUP_COLUMN
from ..generation.batch import Batch
from ..generation.processors import ParsedResponse
from ..generation.results import GenerateJobResults, GenerationBatches, GenerationStatus
from ..generation.vllm_backend import VllmBackend
from ..llm.metadata import ModelMetadata
from ..observability import get_logger

logger = get_logger(__name__)


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
class GroupState:
    """Mutable state for tracking a single group during parallel generation.

    Each group maintains its own sliding-window context, timestamp cursor,
    and retry counters so that multiple groups can be generated in parallel
    while tracking progress independently.
    """

    group_id: str
    """Unique identifier for this group (e.g., device ID, customer ID)."""

    initial_prefill: str
    """Original prefill string (first few records) used to seed generation.  Preserved for potential resets."""

    current_prefill: str
    """Current prefill string, updated as generation progresses to include recently generated records."""

    recent_records: list[dict] = field(default_factory=list)
    """Sliding window of recently generated records used to build the next prompt context."""

    expected_records: int = 0
    """Target record count, calculated from ``(stop_timestamp - start_timestamp) / interval_seconds``."""

    last_timestamp_seconds: int | None = None
    """Timestamp (in seconds) of the most recently generated record, used for chronological validation."""

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
    """Group failed (e.g., too many retries); remove from active, no batch added."""


class TimeseriesBackend(VllmBackend):
    """Time-series aware generator that enforces chronological constraints.

    This backend extends VllmBackend to generate synthetic time-series data with
    strict chronological ordering. It uses a sliding window approach where recently
    generated records are used as context (prefill) for subsequent generation,
    ensuring temporal continuity.

    Key Concepts:
        - Time-Range Based Generation: The number of records generated is
          determined by the configured time range and interval, not by a target
          count. Specifically: (stop_timestamp - start_timestamp) / interval_seconds.
          The `config.generation.num_records` parameter is used only for progress
          tracking, not to limit output.
        - Sliding Window: The backend maintains a window of recent records
          (controlled by `_prefill_context_size`) that are included in each prompt
          to provide context for the LLM, ensuring generated records follow the
          established patterns and timestamps.
        - Parallel Group Generation: Multiple time-series groups (e.g., different
          devices, customers) are processed in parallel batches for efficiency.
          Even single-sequence data uses this path (treated as 1 group via a
          pseudo-group column added during preprocessing). Groups are the same as
          those seen during training (from `model_metadata.initial_prefill`).
        - Chronological Validation: Each generated record must continue from the
          previous timestamp at the expected interval. Out-of-order records are
          marked invalid.

    Generation Flow (parallel group mode):
        1. Initialize GroupState for each group with its prefill context
        2. While groups remain pending or active:
           a. Fill active slots with pending groups (up to max_groups_per_batch)
           b. Build prompts for all active groups using their current prefill
           c. Generate completions for all prompts in a single LLM batch call
           d. Process LLM outputs into per-group Batch objects
           e. For each group:
              - Validate chronological order against group's last timestamp
              - Retain the response with the most valid records (discard others)
              - Update group state with new records (prefill, last_timestamp)
              - Check if stop timestamp was reached (marks group complete)
              - Track low valid fraction; fail group after max retries
           f. Remove completed/failed groups from active list
           g. Save progress snapshots if thresholds are met
           h. Log per-group progress summary

    Stopping Conditions:
        Generation stops when all groups finish (either completed or failed). Individual
        groups and the overall generation can stop for different reasons:

        Per-Group Stopping:
            - Completion (success): A group completes when any generated record
              has a timestamp >= `_stop_timestamp_value`. The group is marked as
              completed and removed from active processing.
            - Failure (low valid fraction): A group fails after
              `config.generation.patience` consecutive batches where the invalid
              record fraction >= `config.generation.invalid_fraction_threshold`.
              This prevents infinite loops when the model consistently produces
              bad output for a particular group. Failed groups are not retried
              and produce no synthetic data for that group ID. The failure is
              reflected in `all_groups_succeeded` returning False.

        Global Stopping:
            - Natural completion: Generation ends when both the pending groups
              queue and active groups list are empty (all groups processed).
            - No records: If `GenerationBatches` detects too many consecutive
              batches with no valid records globally, it signals `STOP_NO_RECORDS`.
            - Target reached: If the target number of records is reached,
              `GenerationBatches` signals `STOP_METRIC_REACHED`.

        When global stopping occurs before all groups complete, `all_groups_succeeded`
        returns False, and the final generation status reflects partial completion.

    Attributes:
        _schema_fragment (str): JSON schema template with column placeholders,
            e.g., '"col1":<unk>,"col2":<unk>'. Used in prompt formatting.
        _samples_per_prompt (int): Number of completion samples to generate per
            prompt. Multiple samples increase chances of getting valid records.
            Default: 5.
        _max_prompts_per_batch (int): Maximum number of prompts to include in a
            single LLM generation call. Controls parallelism. Default: 100.
        _prefill_context_size (int): Number of recent records to include in the
            sliding window prefill context. Computed dynamically as
            max(int(prefill_context_ratio * avg_records_per_example), 3),
            capped by model context length. Falls back to 3 when the ratio
            is 0 or training stats are unavailable.
        _time_column (str): Name of the timestamp column in the data.
        _time_format (str): Format string for parsing timestamps (strptime format),
            or "elapsed_seconds" for numeric elapsed time.
        _is_elapsed_time (bool): True if timestamps are numeric elapsed seconds.
        _start_timestamp_value: Starting timestamp for generation range.
        _stop_timestamp_value: Ending timestamp for generation range. Generation
            stops when a record reaches or exceeds this timestamp.
        _timestamp_interval_seconds (int | None): Expected interval between
            consecutive timestamps. Used for chronological validation.
        _group_column (str | None): Column name used to group time-series data.
            If None or PSEUDO_GROUP_COLUMN, treated as single-sequence.
        _group_prefills (dict[str, str]): Mapping of group_id -> initial prefill
            string. Prefills are the first few records from training data used
            to seed generation for each group.
        _groups (list[str]): List of all group IDs to generate.
    """

    def __init__(self, config: SafeSynthesizerParameters, model_metadata: ModelMetadata, **kwargs):
        super().__init__(config, model_metadata, **kwargs)

        self._schema_fragment = ",".join([f'"{c}":<unk>' for c in self.columns])
        self._samples_per_prompt = 3  # num of samples per prompt
        self._max_prompts_per_batch = 120  # max prompts per batch for parallel group generation
        self._prefill_context_size = self._compute_prefill_context_size(config, model_metadata)
        self._time_column = config.time_series.timestamp_column
        self._time_format = config.time_series.timestamp_format
        self._is_elapsed_time = self._time_format == "elapsed_seconds"
        self._start_timestamp_value = config.time_series.start_timestamp
        self._stop_timestamp_value = config.time_series.stop_timestamp
        self._timestamp_interval_seconds = config.time_series.timestamp_interval_seconds

        # Grouped generation support
        # Note: Since time series preprocessing adds a pseudo-group column when no group
        # is specified, we always have grouped mode (even single-sequence is 1 group).
        self._group_column = config.data.group_training_examples_by
        initial_prefill_value = self.model_metadata.initial_prefill

        if not isinstance(initial_prefill_value, dict):
            raise ValueError(
                "TimeseriesBackend requires initial_prefill to be a dict mapping group -> prefill string. "
                "This should be set by SequentialExampleAssembler during training."
            )

        # Prefills is a dict mapping group -> prefill string
        self._group_prefills: dict[str, str] = initial_prefill_value
        self._groups: list[str] = list(self._group_prefills.keys())

        # Checkpoint/resume support
        self._shutdown_requested = threading.Event()
        self._checkpoint_dir: Path = self.model_metadata.adapter_path

    @staticmethod
    def _compute_prefill_context_size(
        config: SafeSynthesizerParameters, model_metadata: ModelMetadata
    ) -> int:
        """Compute the sliding window prefill context size.

        Uses ``config.generation.prefill_context_ratio`` scaled by the average
        records per training example (stored in model metadata during training).
        The result is floored at 3 and capped so the prefill cannot exceed the
        model's context window.

        Args:
            config: Pipeline configuration (provides the ratio).
            model_metadata: Trained model metadata (provides training stats).

        Returns:
            Number of recent records to include in each generation prompt.
        """
        _DEFAULT_PREFILL = 3
        ratio = config.generation.prefill_context_ratio
        avg_records = model_metadata.avg_records_per_example
        avg_tokens = model_metadata.avg_tokens_per_record

        if ratio <= 0 or avg_records is None:
            logger.info(f"Using default prefill context size: {_DEFAULT_PREFILL}")
            return _DEFAULT_PREFILL

        prefill_size = max(int(ratio * avg_records), _DEFAULT_PREFILL)

        if avg_tokens is not None and avg_tokens > 0:
            max_prefill = int(model_metadata.max_seq_length // avg_tokens)
            if prefill_size > max_prefill:
                logger.warning(
                    f"Prefill context size {prefill_size} exceeds context-length limit "
                    f"({max_prefill} records at ~{avg_tokens:.0f} tokens/record). "
                    f"Capping to {max_prefill}.",
                )
                prefill_size = max(max_prefill, _DEFAULT_PREFILL)

        logger.info(
            f"Prefill context size: {prefill_size} "
            f"(ratio={ratio}, avg_records_per_example={avg_records:.1f})",
        )
        return prefill_size

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
        for fraction in (0.25, 0.5, 0.75, 1.0):
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

    # ── Checkpoint save/restore for resumable generation ────────────────────

    _CHECKPOINT_JSON = ".generation_checkpoint.json"
    _CHECKPOINT_CSV = ".generation_partial_records.csv"

    @property
    def _checkpoint_json_path(self) -> Path:
        return self._checkpoint_dir / self._CHECKPOINT_JSON

    @property
    def _checkpoint_csv_path(self) -> Path:
        return self._checkpoint_dir / self._CHECKPOINT_CSV

    def _save_checkpoint(
        self,
        batches: GenerationBatches,
        all_group_states: dict[str, GroupState],
    ) -> None:
        """Atomically save generation checkpoint (group states + partial records).

        Writes to a temp file first, then renames to ensure crash safety.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Serialize group states
        states_data = {}
        for gid, state in all_group_states.items():
            states_data[gid] = asdict(state)

        manifest = {
            "group_states": states_data,
            "num_valid_records": batches.num_valid_records,
            "num_invalid_records": batches.num_invalid_records,
            "timestamp": time.time(),
        }

        # Write JSON atomically
        json_tmp = self._checkpoint_json_path.with_suffix(".tmp")
        try:
            json_tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
            os.rename(json_tmp, self._checkpoint_json_path)
        except Exception:
            logger.exception("Failed to save checkpoint JSON")
            json_tmp.unlink(missing_ok=True)
            return

        # Write partial records CSV atomically
        csv_tmp = self._checkpoint_csv_path.with_suffix(".tmp")
        try:
            df = batches.to_dataframe(self.columns)
            if not df.empty:
                df = self._sort_dataframe(df)
                df.to_csv(csv_tmp, index=False)
                os.rename(csv_tmp, self._checkpoint_csv_path)
            else:
                # Remove stale CSV if no records yet
                self._checkpoint_csv_path.unlink(missing_ok=True)
        except Exception:
            logger.exception("Failed to save checkpoint CSV")
            csv_tmp.unlink(missing_ok=True)
            return

        logger.info(
            f"Saved generation checkpoint: {batches.num_valid_records} records, "
            f"{sum(1 for s in all_group_states.values() if s.completed)} completed groups "
            f"-> {self._checkpoint_json_path}",
        )

    def _load_checkpoint(self) -> dict | None:
        """Load a generation checkpoint if one exists.

        Returns:
            Parsed checkpoint manifest dict, or None if no valid checkpoint found.
        """
        if not self._checkpoint_json_path.exists():
            return None

        try:
            manifest = json.loads(self._checkpoint_json_path.read_text())
            if "group_states" not in manifest:
                raise ValueError("Checkpoint missing 'group_states' key")
            logger.info(
                f"Found generation checkpoint: {manifest.get('num_valid_records', 0)} records "
                f"from {self._checkpoint_json_path}",
            )
            return manifest
        except Exception:
            logger.exception(
                f"Corrupt checkpoint at {self._checkpoint_json_path}, starting fresh"
            )
            self._delete_checkpoint()
            return None

    def _restore_group_states(
        self, manifest: dict
    ) -> tuple[dict[str, GroupState], list[str], list[str]]:
        """Restore GroupState objects from checkpoint manifest.

        Returns:
            Tuple of (all_group_states, pending_group_ids, completed_or_failed_ids).
        """
        saved_states = manifest["group_states"]
        all_group_states: dict[str, GroupState] = {}
        pending_groups: list[str] = []
        finished_groups: list[str] = []

        for group_id in self._groups:
            if group_id in saved_states:
                sd = saved_states[group_id]
                state = GroupState(
                    group_id=sd["group_id"],
                    initial_prefill=sd["initial_prefill"],
                    current_prefill=sd["current_prefill"],
                    recent_records=sd.get("recent_records", []),
                    expected_records=sd.get("expected_records", 0),
                    last_timestamp_seconds=sd.get("last_timestamp_seconds"),
                    low_valid_fraction_count=sd.get("low_valid_fraction_count", 0),
                    completed=sd.get("completed", False),
                    failed=sd.get("failed", False),
                    total_valid_records=sd.get("total_valid_records", 0),
                    total_invalid_records=sd.get("total_invalid_records", 0),
                )
            else:
                # Group not in checkpoint — initialize fresh
                state = self._init_group_state(group_id)

            all_group_states[group_id] = state
            if state.completed or state.failed:
                finished_groups.append(group_id)
            else:
                pending_groups.append(group_id)

        return all_group_states, pending_groups, finished_groups

    def _load_checkpoint_records(self, batches: GenerationBatches) -> None:
        """Load previously generated records from checkpoint CSV into batches.

        Creates a synthetic Batch containing the restored records so that
        GenerationBatches counters and to_dataframe() include them.
        """
        if not self._checkpoint_csv_path.exists():
            return

        try:
            df = pd.read_csv(self._checkpoint_csv_path)
        except Exception:
            logger.exception("Failed to read checkpoint CSV, starting with no prior records")
            return

        if df.empty:
            return

        # Create a synthetic batch with restored records
        restored_batch = Batch(processor=self.processor)
        records = df.to_dict("records")
        # Build a single ParsedResponse containing all restored records
        restored_response = ParsedResponse(
            valid_records=records,
            invalid_records=[],
            errors=[],
        )
        restored_batch._responses.append(restored_response)
        batches.add_batch(restored_batch)

        logger.info(f"Restored {len(records)} records from checkpoint CSV")

    def _delete_checkpoint(self) -> None:
        """Remove checkpoint files after successful completion."""
        self._checkpoint_json_path.unlink(missing_ok=True)
        self._checkpoint_csv_path.unlink(missing_ok=True)
        # Also clean up any stale tmp files
        self._checkpoint_json_path.with_suffix(".tmp").unlink(missing_ok=True)
        self._checkpoint_csv_path.with_suffix(".tmp").unlink(missing_ok=True)

    # ── End checkpoint methods ────────────────────────────────────────────

    def _format_prompt(self, prefill: str) -> str:
        """Format a generation prompt using the model's template and the given prefill."""
        return self.model_metadata.prompt_config.template.format(
            instruction=self.model_metadata.instruction,
            schema=self._schema_fragment,
            prefill=prefill,
        )

    def _parse_timestamp_seconds(self, timestamp_value: object) -> int | None:
        """Parse a timestamp value to seconds, returning None on failure.

        Uses the shared _parse_timestamp_to_seconds from record_utils but wraps
        exceptions to return None instead of raising.
        """
        if timestamp_value is None:
            return None

        try:
            return _parse_timestamp_to_seconds(timestamp_value, self._time_format)
        except (ValueError, TypeError):
            return None

    def _advance_expected_time(self, timestamp_seconds: int) -> int:
        """Return the next expected timestamp by adding the configured interval."""
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

    def _init_group_state(self, group_id: str) -> GroupState:
        """Initialize a GroupState for a given group.

        Args:
            group_id: The group identifier.

        Returns:
            A new GroupState initialized with the group's prefill.
        """
        initial_prefill = self._group_prefills.get(group_id, "")

        # Calculate expected number of records: (stop - start) / interval + 1
        expected_records = self._compute_expected_records_per_group()

        state = GroupState(
            group_id=group_id,
            initial_prefill=initial_prefill,
            current_prefill=initial_prefill,
            expected_records=expected_records,
        )
        # Parse the last timestamp from the prefill
        state.last_timestamp_seconds = self._get_timestamp_from_prefill(initial_prefill)
        return state

    def _compute_expected_records_per_group(self) -> int:
        """Compute expected number of records per group based on time range and interval.

        Returns:
            Expected number of records per group, or 0 if cannot be computed.
        """
        start_ts = self._parse_timestamp_seconds(self._start_timestamp_value)
        stop_ts = self._parse_timestamp_seconds(self._stop_timestamp_value)
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

    def _get_timestamp_from_prefill(self, prefill: str) -> int | None:
        """Parse prefill string to extract the timestamp of the last record.

        Args:
            prefill: The prefill string containing JSONL records.

        Returns:
            The timestamp in seconds of the last record, or None if not found.
        """
        if not prefill:
            return None

        json_strings = extract_records_from_jsonl_string(prefill)
        if not json_strings:
            return None

        try:
            last_record = json.loads(json_strings[-1])
            timestamp_value = last_record.get(self._time_column)
            if timestamp_value is not None:
                return self._parse_timestamp_seconds(timestamp_value)
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse timestamp from prefill: {e}")

        return None

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
            if timestamp_seconds != expected_ts:
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
        for response in batch._responses:
            if not response.valid_records:
                continue
            if self._is_chronological_for_group(response.valid_records, group_state):
                continue
            response.invalid_records.extend([json.dumps(record) for record in response.valid_records])
            response.errors.extend([("Out-of-order time step", "TimeSeries")] * len(response.valid_records))
            response.valid_records = []

    def _update_group_state(self, group_state: GroupState, records: list[dict]) -> None:
        """Update a group's state with new records.

        Args:
            group_state: The group state to update.
            records: The new valid records.
        """
        if not records:
            return

        group_state.recent_records.extend(records)
        if len(group_state.recent_records) > self._prefill_context_size:
            group_state.recent_records = group_state.recent_records[-self._prefill_context_size :]

        # Update prefill
        tail = group_state.recent_records[-self._prefill_context_size :]
        lines = [json.dumps(record, ensure_ascii=False) for record in tail]
        group_state.current_prefill = "\n".join(lines) + "\n"

        # Update last timestamp
        last_record = records[-1]
        timestamp_seconds = self._parse_timestamp_seconds(last_record.get(self._time_column))
        if timestamp_seconds is not None:
            group_state.last_timestamp_seconds = timestamp_seconds

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

        sort_columns = []
        if self._group_column is not None and self._group_column in df.columns:
            sort_columns.append(self._group_column)
        if self._time_column in df.columns:
            sort_columns.append(self._time_column)

        if sort_columns:
            df = df.sort_values(by=sort_columns).reset_index(drop=True)

        # Remove pseudo-group column from output (it's only used internally)
        if PSEUDO_GROUP_COLUMN in df.columns:
            df = df.drop(columns=[PSEUDO_GROUP_COLUMN])

        return df

    def _build_modified_sampling_params(
        self, sampling_params: SamplingParams, num_active: int
    ) -> tuple[SamplingParams, int]:
        """Build modified sampling params with dynamic samples per prompt.

        Args:
            sampling_params: Base sampling parameters.
            num_active: Number of active groups.

        Returns:
            Tuple of (modified SamplingParams, effective_samples_per_prompt).
        """
        effective_samples_per_prompt = min(
            10,
            max(self._samples_per_prompt, self._max_prompts_per_batch // num_active),
        )

        modified_params = SamplingParams(
            n=effective_samples_per_prompt,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
            min_p=sampling_params.min_p,
            max_tokens=sampling_params.max_tokens,
            repetition_penalty=sampling_params.repetition_penalty,
            skip_special_tokens=sampling_params.skip_special_tokens,
            include_stop_str_in_output=sampling_params.include_stop_str_in_output,
            ignore_eos=sampling_params.ignore_eos,
            stop=sampling_params.stop,
            stop_token_ids=sampling_params.stop_token_ids,
        )

        return modified_params, effective_samples_per_prompt

    def _process_group_result(
        self,
        state: GroupState,
        batch: Batch,
        invalid_fraction_threshold: float,
    ) -> GroupProcessingResult:
        """Process generation result for a single group.

        Args:
            state: The group state.
            batch: The batch containing results for this group.
            invalid_fraction_threshold: Threshold for invalid fraction.

        Returns:
            GroupProcessingResult enum indicating the group's status.
        """
        if self.config.time_series.timestamp_interval_seconds is not None:
            self._check_chronological_for_group(batch, state)

        batch_records = self._retain_single_valid_response(batch)
        reached_stop = self._has_reached_stop_time(batch_records)
        self._update_group_state(state, batch_records)

        # Check if batch has high invalid fraction
        invalid_fraction = 1.0 - batch.valid_record_fraction
        if invalid_fraction >= invalid_fraction_threshold:
            state.low_valid_fraction_count += 1

            patience = self.config.generation.patience
            if batch.num_valid_records == 0:
                logger.warning(
                    f"Group '{state.group_id}' batch produced no valid records "
                    f"(attempt {state.low_valid_fraction_count}/{patience})",
                )
            else:
                logger.warning(
                    f"Group '{state.group_id}' batch has high invalid fraction "
                    f"({invalid_fraction:.1%} >= {invalid_fraction_threshold:.1%}), "
                    f"attempt {state.low_valid_fraction_count}/{patience}",
                )

            if state.low_valid_fraction_count >= patience:
                state.failed = True
                logger.warning(
                    f"Group '{state.group_id}' skipped after {patience} "
                    f"consecutive batches with high invalid fraction (>= {invalid_fraction_threshold:.0%})",
                )
                return GroupProcessingResult.FAILED

            return GroupProcessingResult.IN_PROGRESS

        # Reset the counter when we get a good batch
        state.low_valid_fraction_count = 0

        # Update cumulative stats
        state.total_valid_records += batch.num_valid_records
        state.total_invalid_records += batch.num_invalid_records

        if reached_stop:
            state.completed = True
            return GroupProcessingResult.COMPLETED

        return GroupProcessingResult.IN_PROGRESS

    def _log_parallel_batch_summary(
        self,
        active_states: list[GroupState],
        group_batches: dict[str, Batch],
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
                f"  {status} {state.group_id}: +{batch.num_valid_records} ({batch_valid_rate:.0%} valid), "
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

        Supports checkpoint/resume: if a checkpoint exists from a previous run,
        group states and partial records are restored and generation continues
        from where it left off. Checkpoints are saved periodically (controlled by
        ``config.generation.checkpoint_interval_batches``) and on graceful shutdown
        (when ``_shutdown_requested`` event is set).

        Args:
            batches: The GenerationBatches object to accumulate results.
            sampling_params: Sampling parameters for generation.
            progress_snapshots: Progress snapshots for saving intermediate results (record-based).

        Returns:
            True if all groups completed successfully, False otherwise.
        """
        invalid_fraction_threshold = self.config.generation.invalid_fraction_threshold
        max_groups_per_batch = max(1, self._max_prompts_per_batch // self._samples_per_prompt)
        checkpoint_interval = self.config.generation.checkpoint_interval_batches

        # ── Try to resume from checkpoint ──────────────────────────────────
        checkpoint = self._load_checkpoint()
        if checkpoint is not None:
            all_group_states, pending_groups, finished_groups = self._restore_group_states(checkpoint)
            self._load_checkpoint_records(batches)
            groups_completed = len(finished_groups)
            all_groups_succeeded = all(
                all_group_states[gid].completed for gid in finished_groups
            )
            logger.info(
                f"Resumed from checkpoint: {groups_completed}/{len(self._groups)} groups done, "
                f"{len(pending_groups)} pending, {batches.num_valid_records} records restored",
            )
        else:
            # Initialize states for all groups (fresh start)
            all_group_states = {
                group_id: self._init_group_state(group_id) for group_id in self._groups
            }
            pending_groups = list(self._groups)
            groups_completed = 0
            all_groups_succeeded = True

        active_states: list[GroupState] = []
        batch_iteration = 0

        logger.info(
            f"Starting parallel generation for {len(self._groups)} groups "
            f"(max {max_groups_per_batch} groups per batch, {self._samples_per_prompt} samples per prompt)",
        )

        shutdown_triggered = False

        while pending_groups or active_states:
            # Check for graceful shutdown request
            if self._shutdown_requested.is_set():
                logger.info("Shutdown requested — saving checkpoint and exiting")
                self._save_checkpoint(batches, all_group_states)
                shutdown_triggered = True
                break

            # Fill active slots with pending groups
            while len(active_states) < max_groups_per_batch and pending_groups:
                next_group = pending_groups.pop(0)
                state = all_group_states[next_group]
                active_states.append(state)
                logger.debug(f"Activated group '{next_group}' for parallel generation")

            if not active_states:
                break

            start_time = time.perf_counter()

            # Build prompts and batches for all active groups
            prompts = [self._format_prompt(state.current_prefill) for state in active_states]
            group_batches: dict[str, Batch] = {
                state.group_id: Batch(processor=self.processor) for state in active_states
            }

            # Build modified sampling params
            modified_params, effective_samples_per_prompt = self._build_modified_sampling_params(
                sampling_params, len(active_states)
            )

            # Generate for all prompts at once
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
                    batch.process(completion_idx, completion.text)

            duration = time.perf_counter() - start_time

            # Process results for each group
            states_to_remove = []
            for state in active_states:
                batch = group_batches[state.group_id]
                result = self._process_group_result(state, batch, invalid_fraction_threshold)
                if result == GroupProcessingResult.FAILED:
                    # Failed groups are not retried and produce no output for that group ID.
                    states_to_remove.append(state)
                    groups_completed += 1
                    all_groups_succeeded = False
                elif result == GroupProcessingResult.COMPLETED:
                    states_to_remove.append(state)
                    groups_completed += 1
                    batches.add_batch(batch)
                    logger.info(
                        f"Group '{state.group_id}' completed (stop timestamp reached). "
                        f"Progress: {groups_completed}/{len(self._groups)} groups.",
                    )
                elif result == GroupProcessingResult.IN_PROGRESS:
                    batches.add_batch(batch)

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

            # Periodic checkpoint save
            batch_iteration += 1
            if checkpoint_interval > 0 and batch_iteration % checkpoint_interval == 0:
                self._save_checkpoint(batches, all_group_states)

            # Check for global stop conditions
            if batches.status in [
                GenerationStatus.STOP_NO_RECORDS,
                GenerationStatus.STOP_METRIC_REACHED,
            ]:
                all_groups_succeeded = False
                break

        # Clean up checkpoint on successful completion (not on shutdown)
        if not shutdown_triggered and all_groups_succeeded:
            self._delete_checkpoint()

        return all_groups_succeeded

    def _retain_single_valid_response(self, batch: Batch) -> list[dict]:
        """Retain the response with the most valid records, discarding all others.

        For time-series sliding window generation, only one response can be used
        per batch to maintain chronological continuity. This method selects the
        response with the most valid records and discards all other responses
        (both their valid and invalid records are cleared, and an error note is
        added to track that they were trimmed).

        Args:
            batch: The batch to retain the response from.

        Returns:
            List of valid records from the retained response.
        """
        final_records = []

        # Find the index of the response with the most valid records
        max_valid_idx = None
        max_valid_count = -1
        for idx, response in enumerate(batch._responses):
            count = len(response.valid_records)
            if count > max_valid_count:
                max_valid_count = count
                max_valid_idx = idx
        for idx, response in enumerate(batch._responses):
            if idx != max_valid_idx:
                response.invalid_records = []
                response.errors.extend([("Extra response trimmed for sliding window", "TimeSeries")])
                response.valid_records = []
            else:
                final_records.extend(response.valid_records)

        return final_records

    def generate(
        self,
        data_actions_fn: utils.DataActionsFn | None = None,
    ) -> GenerateJobResults:
        """Generate time-series tabular data using Nemo Safe Synthesizer.

        All time series are processed as groups (single-sequence is treated as 1 group
        via pseudo-group column added during preprocessing).

        Note:
            Generation is time-range based, not count-based. The number of records
            generated is determined by (stop_timestamp - start_timestamp) / interval_seconds
            for each group. The config.generation.num_records parameter is used for
            progress tracking but does not limit output. Groups are the same as those
            seen during training (from model_metadata.initial_prefill).

        Args:
            data_actions_fn: Optional function that takes a DataFrame and returns a modified DataFrame.

        Returns:
            Generation results object, which includes a DataFrame of generated records.
        """
        generation_start = time.monotonic()
        num_records = self.config.generation.num_records

        sampling_params = SamplingParams(
            temperature=self.config.generation.temperature,
            repetition_penalty=self.config.generation.repetition_penalty,
            top_p=self.config.generation.top_p,
            top_k=FIXED_RUNTIME_GENERATE_ARGS["top_k"],
            min_p=FIXED_RUNTIME_GENERATE_ARGS["min_p"],
            max_tokens=self.model_metadata.max_seq_length,
            skip_special_tokens=True,
            include_stop_str_in_output=False,
            ignore_eos=False,
        )

        batches = GenerationBatches(
            target_num_records=num_records,
            patience=self.config.generation.patience,
            invalid_fraction_threshold=self.config.generation.invalid_fraction_threshold,
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
        all_groups_completed = self._generate_parallel_groups(
            batches=batches,
            sampling_params=sampling_params,
            progress_snapshots=progress_snapshots,
        )

        # If shutdown was requested, mark as INCOMPLETE rather than letting
        # job_complete/log_status raise errors for partial results.
        if self._shutdown_requested.is_set():
            batches.status = GenerationStatus.INCOMPLETE
            logger.info(
                f"Generation checkpointed after shutdown request "
                f"({batches.num_valid_records} records saved)",
            )
        else:
            if all_groups_completed and batches.status == GenerationStatus.IN_PROGRESS:
                batches.status = GenerationStatus.COMPLETE
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

        # Sort by group and timestamp for consistent output (also removes pseudo-group column)
        self.gen_results.df = self._sort_dataframe(self.gen_results.df)

        return self.gen_results
