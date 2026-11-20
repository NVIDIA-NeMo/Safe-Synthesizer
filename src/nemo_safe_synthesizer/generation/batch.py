# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Single-batch container for generated records, plus shared error-statistics and summary-table helpers."""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

import pandas as pd

from ..data_processing.record_utils import (
    normalize_dataframe,
)
from ..defaults import (
    HUMAN_READABLE_ERR_MSGS,
    LOG_NUM_ERRORS,
)
from ..generation.processors import (
    ParsedRecord,
    ParsedResponse,
    Processor,
)
from ..observability import get_logger

if TYPE_CHECKING:
    from ..generation.results import GenerationBatches

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


def summarize_errors(errors: Iterable[tuple[str, str]], detailed_errors: bool) -> pd.DataFrame:
    """Return count statistics on record validation errors.

    Args:
        errors: ``(detailed_msg, validator)`` tuples, one per invalid record.
        detailed_errors: If ``True``, include expected column names
            and allowed field values.  If ``False``, report only
            high-level error categories.

    Returns:
        DataFrame indexed by error message with a ``Percentage``
        column, sorted by frequency descending.
    """
    idx = 0 if detailed_errors else 1
    err_msgs = [e[idx] for e in errors]
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


def log_generation_summary(counts: Batch | GenerationBatches, *, detailed_errors: bool, title: str) -> None:
    """Log a generation summary table and, when there are errors, an "Error Statistics" table.

    Emits structured data via ``logger.user.info`` that is rendered
    as Rich ASCII tables on the console and as key/value pairs in
    JSON logs. Optional metrics are omitted from the summary when empty.

    Args:
        counts: The batch, or the job-wide accumulator, whose counts are summarized.
        detailed_errors: If ``True``, include per-column error
            statistics in the log output.
        title: Title of the summary table.
    """
    summary_data: dict[str, object] = {
        "num_prompts": counts.num_prompts,
        "num_valid_records": counts.num_valid_records,
        "num_invalid_records": counts.num_invalid_records,
        "valid_record_fraction": round(counts.valid_record_fraction, 2),
    }
    if counts.finish_reasons:
        summary_data["finish_reasons"] = dict(counts.finish_reasons)
    if counts.num_data_config_rejected_records:
        summary_data["num_data_config_rejected_records"] = counts.num_data_config_rejected_records
    if counts.total_completion_tokens > 0:
        summary_data["completion_tokens"] = counts.total_completion_tokens
        summary_data["valid_record_tokens"] = counts.total_valid_record_tokens
        summary_data["invalid_record_tokens"] = counts.total_invalid_record_tokens
        summary_data["non_record_tokens"] = counts.total_non_record_tokens
        if counts.total_discarded_sample_tokens > 0:
            summary_data["discarded_sample_tokens"] = counts.total_discarded_sample_tokens

    # Pass structured data - processor renders for console, JSON keeps as-is
    logger.user.info(
        "",
        extra={
            "ctx": {
                "render_table": True,
                "tabular_data": summary_data,
                "title": title,
            }
        },
    )

    # Log error statistics if present
    err_stats = counts.error_statistics(detailed_errors=detailed_errors)
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
        self._total_completion_tokens: int = 0
        self._discarded_response_indices: set[int] = set()
        self._discarded_record_tokens: int = 0
        self.finish_reasons: Counter[str] = Counter()

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
    def errors(self) -> list[tuple[str, str]]:
        """``(detailed_msg, validator)`` tuples for every invalid record in this batch."""
        return list(itertools.chain.from_iterable(resp.errors for resp in self._responses))

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

    @property
    def total_completion_tokens(self) -> int:
        """Total tokens across all completions in this batch (from vLLM output)."""
        return self._total_completion_tokens

    @property
    def num_length_truncated_completions(self) -> int:
        """Number of completions that stopped because they reached ``max_tokens``."""
        return self.finish_reasons["length"]

    def _record_token_totals(self) -> tuple[int, int]:
        """Return ``(valid_tokens, invalid_tokens)`` from a single scan of ``_responses``."""
        valid = invalid = 0
        for resp in self._responses:
            for record in resp.records:
                if record.is_valid:
                    valid += record.token_count
                else:
                    invalid += record.token_count
        return valid, invalid

    @property
    def total_valid_record_tokens(self) -> int:
        """Sum of token counts for all valid records in this batch."""
        valid, _ = self._record_token_totals()
        return valid

    @property
    def total_invalid_record_tokens(self) -> int:
        """Sum of token counts for all invalid records in this batch."""
        _, invalid = self._record_token_totals()
        return invalid

    @property
    def total_discarded_sample_tokens(self) -> int:
        """Tokens of the responses and records removed by ``discard_response`` and ``discard_records``."""
        discarded_responses = sum(
            self._responses[index].completion_tokens for index in self._discarded_response_indices
        )
        return discarded_responses + self._discarded_record_tokens

    @property
    def unclamped_non_record_tokens(self) -> int:
        """Completion tokens minus valid, invalid, and discarded tokens, without clamping or warnings.

        Can be slightly negative, because records are tokenized one at a time
        and may include prompt-prefix bytes the model did not generate.
        """
        valid, invalid = self._record_token_totals()
        return self._total_completion_tokens - valid - invalid - self.total_discarded_sample_tokens

    @property
    def total_non_record_tokens(self) -> int:
        """Tokens not part of any recognized record (completion - valid - invalid - discarded).

        Clamped to zero if negative (possible due to tokenizer boundary effects).
        """
        value = self.unclamped_non_record_tokens
        if value < 0:
            valid, invalid = self._record_token_totals()
            logger.warning(
                "Non-record token count is negative; clamping to 0. "
                "This can happen due to tokenizer boundary effects between individual records and full completions.",
                extra={
                    "ctx": {
                        "delta": value,
                        "completion_tokens": self._total_completion_tokens,
                        "valid_record_tokens": valid,
                        "invalid_record_tokens": invalid,
                    }
                },
            )
            return 0
        return value

    @property
    def total_tokenization_time_sec(self) -> float:
        """Wall-clock seconds spent tokenizing records in this batch."""
        return sum(r.tokenization_time_sec for r in self._responses)

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
        return summarize_errors(self.errors, detailed_errors)

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
        log_generation_summary(self, detailed_errors=detailed_errors, title="Batch Generation Summary")

    def process(self, prompt_number: int, text: str, completion_tokens: int = 0) -> None:
        """Process text response from a single prompt in the current batch.

        Args:
            prompt_number: The prompt number in the current batch.
            text: Text generated by the fine-tuned model.
            completion_tokens: Number of tokens in this completion
                (from vLLM ``output.token_ids``).  Defaults to 0 when
                the backend does not provide token counts.
        """
        response = self._processor(prompt_number, text)
        response.completion_tokens = completion_tokens
        self._responses.append(response)
        self._total_completion_tokens += completion_tokens

    def discard_response(self, index: int, error: tuple[str, str]) -> None:
        """Empty one response so its records leave the batch statistics.

        The response stays in the batch, so ``num_prompts`` still counts it,
        and its completion tokens are reported as
        ``total_discarded_sample_tokens`` rather than non-record tokens.
        Still-valid records are invalidated with ``error`` before removal, so
        post-processing bookkeeping that still references them treats them as
        rejected. Discarding the same response again has no further effect.

        Args:
            index: Position of the response in the batch.
            error: ``(detailed_msg, validator)`` tuple for the discarded records.
        """
        response = self._responses[index]
        for record in response.records:
            if record.is_valid:
                record.invalidate(error)
        response.records = []
        self._discarded_response_indices.add(index)

    def discard_records(self, index: int, records: Collection[ParsedRecord], error: tuple[str, str]) -> None:
        """Remove some records of one response so they leave the batch statistics.

        Like ``discard_response`` for part of a response: the records are
        invalidated with ``error`` before removal, and their tokens are reported
        as ``total_discarded_sample_tokens``.

        Args:
            index: Position of the response in the batch.
            records: Records of that response to discard.
            error: ``(detailed_msg, validator)`` tuple for the discarded records.
        """
        response = self._responses[index]
        discarded_ids = {id(record) for record in records}
        for record in response.records:
            if id(record) in discarded_ids:
                if record.is_valid:
                    record.invalidate(error)
                self._discarded_record_tokens += record.token_count
        response.records = [record for record in response.records if id(record) not in discarded_ids]
