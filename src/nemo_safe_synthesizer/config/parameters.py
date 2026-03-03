# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import (
    Field,
    field_validator,
)
from pydantic_core.core_schema import ValidationInfo

from ..configurator.parameters import (
    Parameters,
)
from ..errors import ParameterError
from ..observability import get_logger
from .data import DataParameters
from .differential_privacy import DifferentialPrivacyHyperparams
from .evaluate import EvaluationParameters
from .generate import GenerateParameters
from .replace_pii import PiiReplacerConfig
from .time_series import TimeSeriesParameters
from .training import TrainingHyperparams
from .types import (
    AUTO_STR,
)

__all__ = ["SafeSynthesizerParameters"]


logger = get_logger(__name__)


class SafeSynthesizerParameters(Parameters):
    """Main configuration class for the Safe Synthesizer pipeline.

    This is the top-level configuration class that orchestrates all aspects of
    synthetic data generation including training, generation, privacy, evaluation,
    and data handling. It provides validation to ensure parameter compatibility.
    """

    data: DataParameters = Field(description="Data parameters.", default_factory=DataParameters)

    evaluation: EvaluationParameters = Field(default_factory=EvaluationParameters, description="Evaluation parameters.")

    enable_synthesis: bool = Field(description="Enable synthesizing new data by training a model.", default=True)

    enable_replace_pii: bool = Field(description="Enable replacing PII in the data.", default=True)

    training: TrainingHyperparams = Field(description="Training parameters.", default_factory=TrainingHyperparams)

    generation: GenerateParameters = Field(description="Generation parameters.", default_factory=GenerateParameters)

    privacy: DifferentialPrivacyHyperparams | None = Field(
        description="Privacy parameters. Optional.", default_factory=DifferentialPrivacyHyperparams
    )

    time_series: TimeSeriesParameters = Field(
        description="Time series parameters.", default_factory=TimeSeriesParameters
    )

    replace_pii: PiiReplacerConfig | None = Field(description="PII replacement parameters. Optional.", default=None)

    @field_validator("privacy", mode="after", check_fields=False)
    def check_dp_compatibility(
        cls, dp_params: DifferentialPrivacyHyperparams | None, info: ValidationInfo
    ) -> DifferentialPrivacyHyperparams | None:
        """Ensure that if DP is enabled, max_sequences_per_example is 1 or auto, as well as that use_unsloth is False."""
        if dp_params is None:
            return dp_params
        logger.debug("Checking DP compatibility for privacy parameters. ")
        # logger.debug(f"Privacy parameters: {dp_params}")
        data: DataParameters | None = info.data.get("data")
        if not data:
            raise ParameterError("Data parameters must be provided when DP is enabled.")

        if not dp_params.dp_enabled:
            if data.max_sequences_per_example is not None and data.max_sequences_per_example == AUTO_STR:
                logger.debug("setting max_sequences_per_example to the default of 10 because DP is disabled")
                data.max_sequences_per_example = 10
            return dp_params

        match data.max_sequences_per_example:
            # this should be a valid none or parameter[int|str|none]
            case "auto" | None:
                logger.info("Setting max_sequences_per_example to 1 because DP is enabled.")
                data.max_sequences_per_example = 1
            case None:
                data.max_sequences_per_example = 1
            case v if v not in [AUTO_STR, 1]:
                raise ParameterError(
                    f"When enabling DP, max_sequences_per_example must be set to 1 or 'auto'. Received: {v}"
                )

        logger.debug("Checking Training compatibility for training parameters.")

        training: TrainingHyperparams | None = info.data.get("training")
        logger.debug(f"Training parameters: {training}")

        if not training:
            raise ParameterError("Training parameters must be provided when DP is enabled.")

        if training.use_unsloth not in [False, AUTO_STR]:
            raise ParameterError("Unsloth is currently not compatible with DP.")

        return dp_params

    @classmethod
    def from_params(cls, **kwargs) -> "SafeSynthesizerParameters":
        """Convert singular, flat parameters to nested structure.
        This method takes a flat dictionary of parameters, where keys correspond to
        attributes of the nested parameter classes, and constructs a SafeSynthesizerParameters
        instance with the appropriate nested structure, using default values for each subgroup that
        are not explicitly provided.
        """
        thp = TrainingHyperparams().model_copy(update=kwargs)
        gp = GenerateParameters().model_copy(update=kwargs)
        ep = EvaluationParameters().model_copy(update=kwargs)
        pp = DifferentialPrivacyHyperparams().model_copy(update=kwargs)
        dp = DataParameters().model_copy(update=kwargs)
        tsp = TimeSeriesParameters().model_copy(update=kwargs)

        enable_replace_pii = kwargs.pop("enable_replace_pii", False)
        replace_pii_config = kwargs.get("replace_pii", None)

        match enable_replace_pii, replace_pii_config:
            case True, None:
                logger.debug("enable_replace_pii is True but no config provided - using defaults")
                replace_pii_config = PiiReplacerConfig.get_default_config()
            case True, dict():
                logger.debug("enable_replace_pii is True and config provided - using provided config")
                replace_pii_config = PiiReplacerConfig.get_default_config().model_copy(update=replace_pii_config)
            case True, PiiReplacerConfig():
                logger.debug("enable_replace_pii is True and config provided - using provided config")
            case False, dict() | False, None:
                replace_pii_config = None
                logger.debug("enable_replace_pii is False but config provided - ignoring provided config")

        enable_synthesis = kwargs.get("enable_synthesis", True)
        return cls(
            training=thp,
            generation=gp,
            evaluation=ep,
            privacy=pp,
            data=dp,
            time_series=tsp,
            replace_pii=replace_pii_config,
            enable_synthesis=enable_synthesis,
            enable_replace_pii=enable_replace_pii,
        )
