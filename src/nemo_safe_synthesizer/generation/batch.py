# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
    """
    Take messages like
    'col1' is a required property
    'col2' is a required property

    and group them into a single message like
    "Grouped error message: 'col1'/'col2' is a required property"
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
    """A class to store the results of a batch generation."""

    def __init__(self, processor: Processor):
        self._responses: list[ParsedResponse] = []
        self._processor = processor

    @property
    def num_prompts(self) -> int:
        """The total number of prompts submitted."""
        return len(self._responses)

    @property
    def num_invalid_records(self) -> int:
        """The number of invalid records generated."""
        return sum([len(resp.invalid_records) for resp in self._responses])

    @property
    def num_valid_records(self) -> int:
        """The number of valid records generated."""
        return sum([len(resp.valid_records) for resp in self._responses])

    @property
    def data_config_rejected_records(self) -> list[tuple[str, str]]:
        return [error for resp in self._responses for error in resp.errors if "data_config" in error[1]]

    @property
    def num_data_config_rejected_records(self) -> int:
        """The number of records rejected due to data_config."""
        return len(self.data_config_rejected_records)

    @property
    def valid_record_fraction(self) -> float:
        """The fraction of valid records generated."""
        total_records = self.num_valid_records + self.num_invalid_records
        return 0 if total_records == 0 else self.num_valid_records / total_records

    def error_statistics(self, detailed_errors: bool) -> pd.DataFrame:
        """
        Return count statistics on the errors encountered during generation.

        Args:
            detailed_errors (bool): If True, return detailed error statistics,
                including expected column names or allowed field values.
                If False, only report high-level error categories.
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
        """The metric used to determine if generation should stop."""
        return 1.0 - self.valid_record_fraction

    def to_dataframe(self) -> pd.DataFrame | None:
        """Return the valid records as a DataFrame.

        Returns:
            DataFrame of valid records.
        """
        valid = [resp.valid_records for resp in self._responses]
        flat_records = [record for records in valid for record in records]
        df = pd.DataFrame.from_records(flat_records)
        return None if df.empty else normalize_dataframe(df)

    def log_summary(self, detailed_errors: bool = False) -> None:
        """
        Output a summary of the batch generation results to the log.

        Outputs:
            - Console: Automatically rendered as Rich ASCII tables by structlog processor
            - JSON logs: Structured key/value pairs for machine parsing

        Args:
            detailed_errors (bool): If True, include detailed error statistics in the log.
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
