# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ``"auto"`` sentinel values in config parameters to concrete values.

Inspects dataset characteristics (token counts, record counts) to replace
``"auto"`` placeholders in ``SafeSynthesizerParameters`` with computed values
for rope scaling factor, number of input records to sample, delta, and other
training/privacy parameters.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from ..defaults import DEFAULT_MAX_SEQ_LENGTH, MAX_ROPE_SCALING_FACTOR
from ..observability import get_logger
from ..utils import merge_dicts
from .parameters import SafeSynthesizerParameters
from .types import AUTO_STR

if TYPE_CHECKING:
    pass

POW = 1.2

logger = get_logger(__name__)


def choose_num_input_records_to_sample(rope_scaling_factor: int) -> int:
    """Scale training records linearly with the rope scaling factor.

     ``num_records = rope_scaling_factor * 25000``

    Args:
        rope_scaling_factor: The RoPE scaling multiplier (1 means no scaling).

    Returns:
        Number of records to sample for training.
    """
    return rope_scaling_factor * 25_000


def get_max_token_count(data: pd.DataFrame, group_by: list[str] | str | None) -> int:
    """Estimate the maximum tokens per training example.

    Accounts for prompt overhead (~40 tokens), column names (repeated in JSON
    formatting), and content character counts. Digits are counted as one token
    each; other characters use a 4-chars-per-token heuristic (Llama-2 tokenizer).
    Samples up to 5,000 records from ``data`` for analysis.

    Args:
        data: Training dataframe to analyze.
        group_by: Column(s) used to group records into single training examples.
            When set, grouped records are concatenated before token estimation.

    Returns:
        Estimated maximum token count across all sampled training examples,
        or 1 if the dataframe is empty.
    """
    if data.size == 0:
        return 1

    # Limit to 5k records to keep run time under 5 seconds
    if data.shape[0] > 5000:
        if group_by:
            # Sort by group_by so that we don't break up groups
            data = data.sort_values(group_by)
        data = data.head(5000)

    counts = pd.DataFrame()
    # Estimate the character count introduced by the column names,
    # counting the characters separately for digits and other
    title = " ".join(data.columns)
    title_text = re.sub(r"\d", "", title)
    title_text_char_count = len(title_text)
    title_num_char_count = len(title) - len(title_text)

    # Estimate the character count introduced by each record in the dataset
    counts["content"] = data.apply(lambda x: " ".join([str(x[col]) for col in data.columns]), axis=1)
    if group_by:
        # Concatenate the content of all records with the same group_by value,
        # and count the number of records in each group
        counts[group_by] = data[group_by]
        grouped_counts = counts.groupby(group_by)["content"].apply(lambda x: "\n".join(x)).to_frame()
        grouped_counts.reset_index(inplace=True)
        grouped_counts["num_rows"] = counts.groupby(group_by).size().values
        counts = grouped_counts
    else:
        counts["num_rows"] = 1

    counts["content_text"] = counts["content"].apply(lambda x: re.sub(r"\d.", "", x))
    counts["content_text_char_count"] = counts["content_text"].apply(lambda x: len(x))
    counts["content_num_char_count"] = counts.apply(lambda x: len(x["content"]) - len(x["content_text"]), axis=1)

    # Estimate the token count from the character count
    # For numbers, every digit is one token; for the rest, we estimate 4 characters per token
    # This is assuming we use TinyLlama, which uses the Llama-2 tokenizer
    counts["estimated_content_token_count"] = counts["content_text_char_count"] / 4 + counts["content_num_char_count"]
    estimated_title_token_count = title_text_char_count / 4 + title_num_char_count

    # Get the token count of the assembled example
    num_columns = data.shape[1]
    # These coefficients are estimated using a linear mixed effects model
    # based on a small number of real or simulated datasets
    counts["num_tokens"] = (
        40  # Roughly accounts for the prompt
        + counts["estimated_content_token_count"]
        # Column names are used twice in the json, plus some json formatting
        + (2 + 0.5 * counts["num_rows"]) * estimated_title_token_count
        # Roughly accounts for the json formatting
        + 4 * num_columns * counts["num_rows"]
    )

    max_token_count = counts.num_tokens.max()
    logger.info(
        f"Estimated max token count for examples in dataset - this is used to determine the rope scaling factor: {max_token_count}"
    )
    return max_token_count


def choose_rope_scaling_factor(max_token_count: int, context_length: int = DEFAULT_MAX_SEQ_LENGTH) -> int:
    """Compute the RoPE scaling factor from the estimated max token count.

    Divides ``max_token_count`` by ``context_length``, rounds up, and
    caps the result at ``MAX_ROPE_SCALING_FACTOR``.

    Args:
        max_token_count: Estimated maximum tokens per training example.
        context_length: Base context window size (default ``DEFAULT_MAX_SEQ_LENGTH``).

    Returns:
        Integer scaling factor in the range [1, ``MAX_ROPE_SCALING_FACTOR``].
    """
    rope_scaling_factor = math.ceil(max_token_count / context_length)
    rope_scaling_factor = min(rope_scaling_factor, MAX_ROPE_SCALING_FACTOR)

    return rope_scaling_factor


class AutoConfigResolver:
    """Resolve all ``"auto"`` sentinel values in ``SafeSynthesizerParameters``.

    Inspects the training dataset to compute concrete values for parameters
    left as ``"auto"`` (rope scaling, number of input records, unsloth,
    delta, max sequences per example). Resolution order matters:
    ``rope_scaling_factor`` is resolved first because
    ``num_input_records_to_sample`` depends on it.

    Args:
        data: Training dataframe used to derive auto parameters.
        config: Configuration containing ``"auto"`` sentinel values to resolve.
    """

    def __init__(self, data: pd.DataFrame, config: SafeSynthesizerParameters):
        self._data = data
        self._config = config
        self._record_count = data.shape[0]
        self._delta: float | str | None = config.get("delta")
        self._dp_enabled: bool | None = config.get("dp_enabled")
        self._rope_scaling_factor: int | None = None

    def __call__(self) -> SafeSynthesizerParameters:
        """Delegate to [`resolve`][nemo_safe_synthesizer.config.autoconfig.AutoConfigResolver.resolve]."""
        return self.resolve()

    def _determine_rope_scaling_factor(self) -> dict[str, int]:
        """Determine the rope scaling factor if set to auto.

        Returns:
            Dict with rope_scaling_factor if auto-determined, empty dict otherwise.
        """
        if self._config.get("rope_scaling_factor") != AUTO_STR:
            return {}

        # this is separated into two functions
        # to enable carrying the max_token_count forward to the ModelMetadata class
        # in the future. we don't need to save it at the moment.
        max_token_count = get_max_token_count(data=self._data, group_by=self._config.data.group_training_examples_by)
        self._rope_scaling_factor = choose_rope_scaling_factor(max_token_count=max_token_count)
        logger.info(
            f"Parameter `rope_scaling_factor` was automatically set to "
            f"{self._rope_scaling_factor} based on an estimated token count given "
            f"the lengths of each training record and the column names."
        )
        return {"rope_scaling_factor": self._rope_scaling_factor}

    def _determine_num_input_records_to_sample(self) -> dict[str, int]:
        """Determine the number of input records to sample if set to auto.

        Returns:
            Dict with num_input_records_to_sample if auto-determined, empty dict otherwise.
        """
        if self._config.training.num_input_records_to_sample != AUTO_STR:
            return {}

        num_records = choose_num_input_records_to_sample(rope_scaling_factor=self._rope_scaling_factor or 1)
        return {"num_input_records_to_sample": num_records}

    def _determine_use_unsloth(self) -> dict[str, bool]:
        """Determine whether to use unsloth if set to auto.

        Returns:
            Dict with use_unsloth if auto-determined, empty dict otherwise.
        """
        if self._config.training.use_unsloth != AUTO_STR:
            logger.info(f"unsloth was set to {self._config.training.use_unsloth}, using that value")
            return {}

        if self._dp_enabled:
            logger.info("unsloth was set to 'auto', disabling because DP is enabled")
            return {"use_unsloth": False}
        else:
            logger.info("unsloth was set to 'auto', enabling")
            return {"use_unsloth": True}

    def _determine_delta(self) -> dict[str, float]:
        r"""Determine the delta parameter for differential privacy if set to auto.

        We must set $\delta \ll 1/n$, where $n$ is the training record count.
        With approximate DP, the probability that at least one person has
        their data exposed is $1 - (1 - \delta)^n$. For small $\delta$, the
        Taylor expansion is roughly $\delta \cdot n$, which we want to bound
        by e.g. 10%. To achieve this, we set $\delta = 1 / n^{1.2}$ when
        $n \ge 100$, and $0.1 / n$ otherwise.

        Returns:
            Dict with delta if auto-determined, empty dict otherwise.
        """
        if not (self._dp_enabled and self._delta == AUTO_STR):
            return {}

        if self._record_count < 100:
            d = 0.1 / self._record_count
        else:
            d = 1 / self._record_count**POW

        logger.info(
            f"Parameter `delta` was automatically set to {d:.2g} based "
            "on the number of records, n. Note that n was not determined "
            "with differential privacy."
        )
        return {"delta": d}

    def _determine_max_sequences_per_example(self) -> dict[str, int | None]:
        """Determine max_sequences_per_example if set to auto.

        Returns:
            Dict with max_sequences_per_example resolved to a concrete value:
            1 if DP is enabled, 10 if auto with DP disabled, or the
            explicit value (int) if manually specified, or None if not specified.
        """
        if self._dp_enabled is True:
            if self._config.data.max_sequences_per_example in [None, AUTO_STR, 1]:
                logger.info(
                    "Parameter `max_sequences_per_example` was automatically set "
                    "to 1 based on the use of differential privacy."
                )
            else:
                logger.info(
                    "Parameter `max_sequences_per_example` does not allow the value of "
                    "{self._config.data.max_sequences_per_example} when DP is enabled. Setting to 1 instead."
                )
            return {"max_sequences_per_example": 1}
        elif self._config.data.max_sequences_per_example != AUTO_STR:
            if self._config.data.max_sequences_per_example is None:
                logger.info(
                    "Parameter `max_sequences_per_example` is not specified, so each example will fill up the context window."
                )
            return {"max_sequences_per_example": self._config.data.max_sequences_per_example}

        else:
            logger.info(
                "Parameter `max_sequences_per_example` was automatically set to 10 for best performance/efficiency."
            )
            return {"max_sequences_per_example": 10}

    def _build_updated_params(
        self,
        training_params: dict[str, Any],
        data_params: dict[str, Any],
        privacy_params: dict[str, Any],
    ) -> SafeSynthesizerParameters:
        """Build and validate the updated configuration parameters.

        Args:
            training_params: Auto-determined training parameters.
            data_params: Auto-determined data parameters.
            privacy_params: Auto-determined privacy parameters.

        Returns:
            The validated SafeSynthesizerParameters.
        """
        new_params = {
            "training": training_params,
            "data": data_params,
            "privacy": privacy_params,
        }
        updated_params = merge_dicts(self._config.model_dump(exclude_unset=True), new_params)
        logger.debug(f"params to update: {updated_params}")
        my_config = SafeSynthesizerParameters.model_validate(updated_params)
        logger.debug(f"auto-updated config: {my_config.model_dump(exclude_unset=True)}")
        return my_config

    def resolve(self) -> SafeSynthesizerParameters:
        """Replace all ``"auto"`` parameters with concrete values.

        Resolution order matters: ``rope_scaling_factor`` is resolved before
        ``num_input_records_to_sample`` because the latter depends on it.

        Returns:
            A new ``SafeSynthesizerParameters`` with all ``"auto"`` values resolved.
        """
        # Determine training params (order matters: rope_scaling_factor first)
        training_params: dict[str, Any] = {}
        training_params.update(self._determine_rope_scaling_factor())
        training_params.update(self._determine_num_input_records_to_sample())
        training_params.update(self._determine_use_unsloth())

        # Determine data params
        data_params: dict[str, Any] = {}
        data_params.update(self._determine_max_sequences_per_example())

        # Determine privacy params
        privacy_params: dict[str, Any] = {}
        privacy_params.update(self._determine_delta())

        return self._build_updated_params(training_params, data_params, privacy_params)
