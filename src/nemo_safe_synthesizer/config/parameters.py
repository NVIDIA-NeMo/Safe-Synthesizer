# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from ..configurator.parameters import Parameters
from ..errors import ParameterError
from ..observability import get_logger
from .data import DataParameters
from .differential_privacy import DifferentialPrivacyHyperparams
from .evaluate import EvaluationParameters
from .generate import GenerateParameters
from .replace_pii import PiiReplacerConfig
from .time_series import TimeSeriesParameters
from .training import TrainingHyperparams
from .types import AUTO_STR

__all__ = ["SafeSynthesizerParameters"]


logger = get_logger(__name__)


class SafeSynthesizerParameters(Parameters):
    """Main configuration class for the Safe Synthesizer pipeline.

    This is the top-level configuration class that orchestrates all aspects of
    synthetic data generation including training, generation, privacy, evaluation,
    and data handling. It provides validation to ensure parameter compatibility.
    """

    data: DataParameters = Field(
        description="Configuration controlling how input data is grouped and split for training and evaluation.",
        default_factory=DataParameters,
    )

    evaluation: EvaluationParameters = Field(
        description="Parameters for evaluating the quality of generated synthetic data.",
        default_factory=EvaluationParameters,
    )

    training: TrainingHyperparams = Field(
        description="Hyperparameters for model training such as learning rate, batch size, and LoRA adapter settings.",
        default_factory=TrainingHyperparams,
    )

    generation: GenerateParameters = Field(
        description="Parameters governing synthetic data generation including temperature, top-p, and number of records to produce.",
        default_factory=GenerateParameters,
    )

    privacy: DifferentialPrivacyHyperparams | None = Field(
        description="Differential-privacy hyperparameters. When ``None``, differential privacy is disabled entirely.",
        default_factory=DifferentialPrivacyHyperparams,
    )

    time_series: TimeSeriesParameters = Field(
        description="Configuration for time-series mode. Time-series pipeline is currently experimental.",
        default_factory=TimeSeriesParameters,
    )

    replace_pii: PiiReplacerConfig | None = Field(
        description="PII replacement configuration. When ``None``, PII replacement is skipped.",
        default_factory=PiiReplacerConfig.get_default_config,
    )

    @field_validator("privacy", mode="after", check_fields=False)
    def check_dp_compatibility(
        cls, dp_params: DifferentialPrivacyHyperparams | None, info: ValidationInfo
    ) -> DifferentialPrivacyHyperparams | None:
        """Validate that DP-enabled configs have compatible data and training settings.

        When DP is enabled, enforces that ``max_sequences_per_example``
        is ``1`` (or ``"auto"``, which is resolved to ``1``) to bound
        per-example contribution, and that Unsloth is disabled since it
        is not yet compatible with DP-SGD. When DP is disabled but
        ``max_sequences_per_example`` is ``"auto"``, defaults it to
        ``10``.

        Raises:
            ParameterError: If ``data`` or ``training`` parameters are
                missing, ``max_sequences_per_example`` is not ``1``, or
                Unsloth is enabled alongside DP.
        """
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

          Takes a flat dictionary of parameters, where keys correspond to
          attributes of the nested parameter classes, and constructs a
          ``SafeSynthesizerParameters`` instance with the appropriate nested
          structure, using default values for each subgroup that are not
          explicitly provided.

          Args:
              **kwargs: Flat key-value pairs that map to attributes of the
                  nested parameter classes (e.g., ``TrainingHyperparams``,
                  ``GenerateParameters``).

          Returns:
              A fully initialized ``SafeSynthesizerParameters`` instance with
              nested sub-configurations populated from the provided values.

        Example:
            >>> from nemo_safe_synthesizer.config import SafeSynthesizerParameters
            >>> vals = {"use_structured_generation: True, "pii_replay_enabled": False}}
            >>> SafeSynthesizerParams.from_params(vals)
        """
        thp = TrainingHyperparams().model_copy(update=kwargs)
        gp = GenerateParameters().model_copy(update=kwargs)
        ep = EvaluationParameters().model_copy(update=kwargs)
        pp = DifferentialPrivacyHyperparams().model_copy(update=kwargs)
        dp = DataParameters().model_copy(update=kwargs)
        tsp = TimeSeriesParameters().model_copy(update=kwargs)

        extra: dict = {
            "training": thp,
            "generation": gp,
            "evaluation": ep,
            "privacy": pp,
            "data": dp,
            "time_series": tsp,
        }
        if "replace_pii" in kwargs:
            extra["replace_pii"] = kwargs["replace_pii"]
        return cls(**extra)
