# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-batch container for generated records and error statistics."""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from ..data_processing.record_utils import (
    normalize_dataframe,
)
from ..defaults import (
    HUMAN_READABLE_ERR_MSGS,
    LOG_NUM_ERRORS,
)
from ..generation.processors import (
    ParsedResponse,
    Processor,
)
from ..observability import get_logger

logger = get_logger(__name__)


def _group_error_messages(all_error_messages: list[str], common_error_string: str) -> list[str]:
    """Consolidate error messages that share a common suffix.

    For example, ``"'col1' is a required property"`` and
    ``"'col2' is a required property"`` become a single entry:
    ``"Grouped error message: 'col1'/'col2' is a required property"``.

    Args:
        all_error_messages: Raw error messages from record validation.
        common_error_string: The shared substring to group on
            (e.g. ``" is a required property"``).

    Returns:
        Updated error messages with duplicates consolidated.
    """
    # Split the error messages to obtain the invidual parts from the similar error messages
    all_message_parts = set()
    for msg in all_error_messages:
        if common_error_string in msg:
            all_message_parts.add(tuple(msg.split(common_error_string)))

    # Group the invidivual parts
    grouped = defaultdict(list)
    for message_parts in all_message_parts:
        grouped[message_parts[1]].append(message_parts[0])
    grouped = dict(grouped)

    # Truncate the list of grouped error messages and format into a single string
    is_grouped = {k: len(v) > 1 for k, v in grouped.items()}
    grouped = {k: sorted(v) for k, v in grouped.items()}  # Make it deterministic
    grouped = {k: v[:5] + ["..."] if len(v) > 5 else v for k, v in grouped.items()}
    grouped = {k: "/".join(v) for k, v in grouped.items()}

    # Update the error messages with the grouped error messages
    updated_error_messages = []
    for msg in all_error_messages:
        if common_error_string in msg:
            first, second = msg.split(common_error_string)
            # If there's only one unique element after grouping, don't group
            if not is_grouped[second]:
                updated_error_messages.append(msg[:100] + "..." if len(msg) > 100 else msg)
                continue
            updated_error_messages.append(
                "Grouped error message: "
                + grouped[second]
                + common_error_string
                + (second[:50] + "..." if len(second) > 50 else second)
            )
        else:
            updated_error_messages.append(msg)
    return updated_error_messages


class Batch:
    """Container for the results of a single generation batch.

    Collects
    [`ParsedResponse`][nemo_safe_synthesizer.generation.processors.ParsedResponse]
    objects produced by the processor and exposes aggregate counts and error
    statistics.

    Args:
        processor: The processor used to parse LLM outputs into records.
    """

    def __init__(self, processor: Processor):
        self._responses: list[ParsedResponse] = []
        self._processor = processor

    @property
    def num_prompts(self) -> int:
        """Total number of prompts submitted in this batch."""
        return len(self._responses)

    @property
    def num_invalid_records(self) -> int:
        """Number of invalid records generated in this batch."""
        return sum([len(resp.invalid_records) for resp in self._responses])

    @property
    def num_valid_records(self) -> int:
        """Number of valid records generated in this batch."""
        return sum([len(resp.valid_records) for resp in self._responses])

    @property
    def data_config_rejected_records(self) -> list[tuple[str, str]]:
        """Error tuples for records rejected by ``data_config`` validation."""
        return [error for resp in self._responses for error in resp.errors if "data_config" in error[1]]

    @property
    def num_data_config_rejected_records(self) -> int:
        """Count of records rejected by ``data_config`` validation."""
        return len(self.data_config_rejected_records)

    @property
    def valid_record_fraction(self) -> float:
        """Fraction of generated records that passed validation."""
        total_records = self.num_valid_records + self.num_invalid_records
        return 0 if total_records == 0 else self.num_valid_records / total_records

    def error_statistics(self, detailed_errors: bool) -> pd.DataFrame:
        """Return count statistics on errors encountered during generation.

        Args:
            detailed_errors: If ``True``, include expected column names
                and allowed field values.  If ``False``, report only
                high-level error categories.

        Returns:
            DataFrame indexed by error message with a ``Percentage``
            column, sorted by frequency descending.
        """
        idx = 0 if detailed_errors else 1
        err_msgs = [e[idx] for resp in self._responses for e in resp.errors]
        # Map error messages to human-readable categories, as necessary
        err_msgs = [HUMAN_READABLE_ERR_MSGS.get(msg, msg) for msg in err_msgs]
        if detailed_errors:
            # Group similar error messages to consolidate the error counts
            common_error_strings = [
                " is not one of ",
                " is a required property",
                " is greater than the maximum of ",
                " is less than the minimum of ",
            ]
            for common_error_string in common_error_strings:
                err_msgs = _group_error_messages(err_msgs, common_error_string)
        err_stats = pd.DataFrame.from_dict(Counter(err_msgs), orient="index", columns=["cnt"])
        err_stats["Percentage"] = err_stats["cnt"] / err_stats["cnt"].sum()
        err_stats = err_stats.drop("cnt", axis=1)
        # sort `index` (error messages) then `Percentage`, so it's deterministic
        err_stats = err_stats.sort_index()
        err_stats = err_stats.sort_values("Percentage", ascending=False, kind="mergesort")
        if LOG_NUM_ERRORS is not None:
            # separate out data_config errors to ensure they don't get truncated

            data_config_errors = err_stats[err_stats.index.astype("str").str.contains("data_config", na=False)]
            other_errors = err_stats[~err_stats.index.astype("str").str.contains("data_config", na=False)]
            other_errors = other_errors.head(LOG_NUM_ERRORS)

            err_stats = pd.concat([data_config_errors, other_errors])
            err_stats = err_stats.sort_values("Percentage", ascending=False)
        return err_stats

    @property
    def stopping_metric(self) -> float:
        """Invalid record fraction, used by
        [`GenerationStopCondition`][nemo_safe_synthesizer.generation.stopping.GenerationStopCondition].
        """
        return 1.0 - self.valid_record_fraction

    def to_dataframe(self) -> pd.DataFrame | None:
        """Return the valid records as a normalized DataFrame.

        Returns:
            DataFrame of valid records, or ``None`` if no valid records
            were generated.
        """
        valid = [resp.valid_records for resp in self._responses]
        flat_records = [record for records in valid for record in records]
        df = pd.DataFrame.from_records(flat_records)
        return None if df.empty else normalize_dataframe(df)

    def log_summary(self, detailed_errors: bool = False) -> None:
        """Log a summary of the batch generation results.

        Emits structured data via ``logger.user.info`` that is rendered
        as Rich ASCII tables on the console and as key/value pairs in
        JSON logs.

        Args:
            detailed_errors: If ``True``, include per-column error
                statistics in the log output.
        """
        err_stats: pd.DataFrame = self.error_statistics(detailed_errors=detailed_errors)

        # Build structured summary data - processor renders as table for console
        summary_data = {
            "num_prompts": self.num_prompts,
            "num_valid_records": self.num_valid_records,
            "num_invalid_records": self.num_invalid_records,
            "valid_record_fraction": round(self.valid_record_fraction, 2),
        }
        if self.num_data_config_rejected_records:
            summary_data["num_data_config_rejected_records"] = self.num_data_config_rejected_records

        # Pass structured data - processor renders for console, JSON keeps as-is
        logger.user.info(
            "",
            extra={
                "ctx": {
                    "render_table": True,
                    "tabular_data": summary_data,
                    "title": "Batch Generation Summary",
                }
            },
        )

        # Log error statistics if present
        if not err_stats.empty:
            # Build structured error data - processor renders as table for console
            error_data = {str(cat): round(row["Percentage"], 2) for cat, row in err_stats.iterrows()}
            logger.user.info(
                "",
                extra={
                    "ctx": {
                        "render_table": True,
                        "tabular_data": error_data,
                        "title": "Error Statistics",
                    }
                },
            )

    def process(self, prompt_number: int, text: str) -> None:
        """Process text response from a single prompt in the current batch.

        Args:
            prompt_number: The prompt number in the current batch.
            text: Text generated by the fine-tuned model.
        """
        self._responses.append(self._processor(prompt_number, text))
