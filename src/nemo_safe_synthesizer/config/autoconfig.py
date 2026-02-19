# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration update logic for auto-params for NSS models.
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal

import pandas as pd
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

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
    # With no rope scaling (rope=1), the default is 25_000. For now our best guess
    # is to increase training steps linearly as rope scaling increases. So rope=2 uses a
    # sample of 50_000, etc.
    return rope_scaling_factor * 25_000


def get_max_token_count(data: pd.DataFrame, group_by: list[str] | str | None) -> int:
    """
    Estimate the maximum token count needed for the data.
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
    """
    Estimate the rope scaling factor based on the max token count and context length.
    We assume a context length of 2048 tokens but this can be overridden.
    """
    rope_scaling_factor = math.ceil(max_token_count / context_length)
    rope_scaling_factor = min(rope_scaling_factor, MAX_ROPE_SCALING_FACTOR)

    return rope_scaling_factor


class AutoConfigResolver:
    """
    Handles auto-determination of config parameters based on the provided dataset.

    This class decomposes the config update logic into testable private methods.
    """

    def __init__(self, data: pd.DataFrame, config: SafeSynthesizerParameters):
        """
        Initialize the ConfigUpdater.

        Args:
            data: The data to use for auto-determination.
            config: The config to update.
        """
        self._data = data
        self._config = config
        self._record_count = data.shape[0]
        self._delta: float | str | None = config.get("delta")
        self._dp_enabled: bool | None = config.get("dp_enabled")
        self._rope_scaling_factor: int | None = None

    def __call__(self) -> SafeSynthesizerParameters:
        """
        Resolve the auto-determined config parameters.

        Returns:
            The updated config with auto parameters resolved.
        """
        return self.resolve()

    def _determine_rope_scaling_factor(self) -> dict[str, int]:
        """
        Determine the rope scaling factor if set to auto.

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
        """
        Determine the number of input records to sample if set to auto.

        Returns:
            Dict with num_input_records_to_sample if auto-determined, empty dict otherwise.
        """
        if self._config.training.num_input_records_to_sample != AUTO_STR:
            return {}

        num_records = choose_num_input_records_to_sample(rope_scaling_factor=self._rope_scaling_factor or 1)
        return {"num_input_records_to_sample": num_records}

    def _determine_use_unsloth(self) -> dict[str, bool]:
        """
        Determine whether to use unsloth if set to auto.

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
        """
        Determine the delta parameter for differential privacy if set to auto.

        We must set delta <<1/n, where n is the training record count.
        With approximate DP, the probability that at least one person has
        their data exposed is `1-(1-delta)^n`. For small delta, the Taylor
        expansion is roughly `delta * n`, which we want to bound by e.g.
        10%. To achieve this, we set `delta = 1/n^POW` when the record
        count is >= 100, and 0.1/n otherwise.

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

    def _determine_max_sequences_per_example(self) -> dict[str, Literal["auto"] | int | None]:
        """
        Determine max_sequences_per_example if set to auto.

        Returns:
            Dict with max_sequences_per_example if auto-determined, empty dict otherwise.
        """
        if self._dp_enabled is True:
            logger.info(
                "Parameter `max_sequences_per_example` was automatically set "
                "to 1 based on the use of differential privacy."
            )
            return {"max_sequences_per_example": 1}
        elif self._config.data.max_sequences_per_example != AUTO_STR:
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
        """
        Build and validate the updated configuration parameters.

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
        """
        Update the config's `auto` parameters with concrete values.

        Returns:
            The updated config with auto parameters resolved.
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


@dataclass(frozen=True)
class AutoParamsValidator:
    value_func: Callable[[Any], bool]

    def validate(self, value, _info):
        if isinstance(value, str) and value == "auto":
            return value
        elif self.value_func(value):
            return value
        else:
            raise ValueError(f"AutoParam validation failed: {inspect.getsource(self.value_func)}, got {value}")

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.with_info_after_validator_function(
            self.validate, handler(source_type), field_name=handler.field_name
        )
