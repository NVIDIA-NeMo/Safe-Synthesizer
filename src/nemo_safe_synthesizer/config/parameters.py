# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, Self, TypeAlias

from pydantic import BaseModel, Field, model_validator
from typing_extensions import override

from ..configurator.parameters import Parameters
from ..data_processing.records.json_types import JsonValue
from ..errors import ParameterError
from ..observability import get_logger
from ..telemetry import _telemetry_enabled
from ..utils import merge_dicts
from .data import DataParameters
from .differential_privacy import DifferentialPrivacyHyperparams
from .evaluate import EvaluationParameters
from .generate import GenerateParameters
from .preflight import PreflightParameters
from .replace_pii import PiiReplacerConfig
from .time_series import TimeSeriesParameters
from .training import TrainingHyperparams
from .types import AUTO_STR

ConfigPatch: TypeAlias = Mapping[str, JsonValue]
_SectionPatch: TypeAlias = dict[str, object]

__all__ = ["ConfigPatch", "SafeSynthesizerParameters"]


logger = get_logger(__name__)


def _collect_set_fields(model: BaseModel) -> _SectionPatch:
    """Recursively collect a model's explicitly-set fields as a nested dict.

    Unlike ``model_dump(exclude_unset=True)``, nested models are always
    traversed -- even when the parent did not mark the nested field as set --
    so in-place mutations of nested fields (e.g.
    ``cfg.generation.validation.foo = True``) are captured. A nested model is
    included only when it has at least one set field of its own.
    """
    overrides: _SectionPatch = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            nested = _collect_set_fields(value)
            if nested:
                overrides[name] = nested
        elif name in model.__pydantic_fields_set__:
            overrides[name] = value
    return overrides


def _overlay_set_fields(saved: Parameters, runtime: Parameters) -> Parameters:
    """Deep-merge ``runtime``'s explicitly-set fields onto ``saved``.

    Only fields ``runtime`` marks as set (recursively, at every nesting level)
    override ``saved``; unset fields keep their saved values. The merged mapping
    is revalidated through the model so nested groups and type coercion are
    handled correctly. Returns ``saved`` unchanged when ``runtime`` sets no
    fields.
    """
    overrides = _collect_set_fields(runtime)
    if not overrides:
        return saved
    return saved.model_validate(merge_dicts(saved.model_dump(), overrides))


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

    preflight: PreflightParameters = Field(
        description="Preflight validation overrides, including checks to skip via ``disabled_checks``.",
        default_factory=PreflightParameters,
    )

    emit_telemetry: bool = Field(
        default_factory=_telemetry_enabled,
        description=(
            "Whether to emit anonymous Safe Synthesizer telemetry events. "
            "Defaults from NEMO_TELEMETRY_ENABLED when unset."
        ),
    )

    @model_validator(mode="after")
    def _validate_and_resolve_data_params(self) -> Self:
        """Validate that DP-enabled configs have compatible data settings.

        When DP is enabled, enforces that ``max_sequences_per_example``
        is ``1`` (or ``"auto"``, which is resolved to ``1``) to bound
        per-example contribution. When DP is disabled but
        ``max_sequences_per_example`` is ``"auto"``, defaults it to
        ``10`` -- or to ``None`` in time-series mode, so each example
        fills the context window.

        DP and time-series mode are mutually exclusive: combining them
        would force ``max_sequences_per_example=1``, which collapses
        the temporal structure time-series mode is designed to
        preserve.

        Raises:
            ParameterError: If DP and time-series are both enabled, or
                if DP is enabled and ``max_sequences_per_example`` is
                not ``1``.
        """
        dp_enabled = self.privacy is not None and self.privacy.dp_enabled
        is_timeseries = self.time_series.is_timeseries

        if dp_enabled and is_timeseries:
            raise ParameterError(
                "Differential privacy is not supported in time-series mode. "
                "Set time_series.is_timeseries=False or privacy.dp_enabled=False."
            )

        max_seq = self.data.max_sequences_per_example

        if dp_enabled:
            match max_seq:
                case "auto" | None:
                    logger.info("Setting max_sequences_per_example to 1 because DP is enabled.")
                    self.data.max_sequences_per_example = 1
                case 1:
                    pass
                case invalid:
                    raise ParameterError(
                        f"When enabling DP, max_sequences_per_example must be 1, 'auto', or None. Received: {invalid!r}"
                    )
            return self

        if max_seq != AUTO_STR:
            return self

        if is_timeseries:
            logger.info(
                "Setting max_sequences_per_example to None for time-series mode "
                "so each example fills the context window."
            )
            self.data.max_sequences_per_example = None
        else:
            logger.debug("Setting max_sequences_per_example to the default of 10.")
            self.data.max_sequences_per_example = 10
        return self

    @model_validator(mode="after")
    def check_timeseries_group_column(self) -> Self:
        if self.time_series is not None and self.time_series.is_timeseries:
            if self.data.group_training_examples_by is None:
                warnings.warn(
                    "is_timeseries=True without group_training_examples_by: "
                    "an internal __nss_sequence_id column will be added automatically.",
                    stacklevel=2,
                )
        return self

    @classmethod
    @override
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
            >>> SafeSynthesizerParameters.from_params(use_structured_generation=True)
        """
        thp = TrainingHyperparams().model_copy(update=kwargs)
        gp = GenerateParameters().model_copy(update=kwargs)
        ep = EvaluationParameters().model_copy(update=kwargs)
        pp = DifferentialPrivacyHyperparams().model_copy(update=kwargs)
        dp = DataParameters().model_copy(update=kwargs)
        tsp = TimeSeriesParameters().model_copy(update=kwargs)

        extra: dict[str, Any] = {
            "training": thp,
            "generation": gp,
            "evaluation": ep,
            "privacy": pp,
            "data": dp,
            "time_series": tsp,
        }
        if "replace_pii" in kwargs:
            extra["replace_pii"] = kwargs["replace_pii"]
        if "preflight" in kwargs:
            extra["preflight"] = kwargs["preflight"]
        if "emit_telemetry" in kwargs:
            extra["emit_telemetry"] = kwargs["emit_telemetry"]
        return cls(**extra)

    @classmethod
    def from_config_patch(cls, patch: ConfigPatch) -> Self:
        """Validate a sparse top-level config patch as a full configuration."""
        return cls.model_validate(patch)

    def with_config_patch(self, patch: ConfigPatch) -> Self:
        """Apply a sparse top-level config patch and revalidate the result.

        Only fields explicitly set on ``self`` are carried into the merge before
        applying ``patch``. This preserves file/CLI precedence while keeping
        default values implicit for future ``exclude_unset`` dumps.
        """
        params = merge_dicts(self.model_dump(exclude_unset=True), patch)
        return type(self).model_validate(params)

    def with_runtime_overrides(self, runtime: SafeSynthesizerParameters) -> "SafeSynthesizerParameters":
        """Apply resume-time generation/evaluation/telemetry overrides onto a copy of self.

        ``self`` is the saved training-run config. Only explicitly-set
        ``generation`` and ``evaluation`` fields from ``runtime`` are merged in,
        plus ``emit_telemetry`` when the caller set it. Training, data, privacy,
        and other sections are preserved so training provenance survives a
        generate-only resume.

        Args:
            runtime: Config carrying resume-time CLI/SDK overrides. Typically
                sparse -- only the fields the caller set are applied.

        Returns:
            A new ``SafeSynthesizerParameters`` with overrides applied. The
            result is fully independent of ``self``: sections that are not
            overridden are deep-copied, so later mutation of either object does
            not affect the other.
        """
        updates: dict[str, object] = {}
        # Only record sections that actually changed; unchanged sections are
        # deep-copied by ``model_copy(deep=True)`` below so the returned config
        # never shares mutable sub-objects with ``self``.
        generation = _overlay_set_fields(self.generation, runtime.generation)
        if generation is not self.generation:
            updates["generation"] = generation
        evaluation = _overlay_set_fields(self.evaluation, runtime.evaluation)
        if evaluation is not self.evaluation:
            updates["evaluation"] = evaluation
        # emit_telemetry is a top-level scalar: detect explicit assignment,
        # since there is no sub-model to inspect for set fields.
        if "emit_telemetry" in runtime.__pydantic_fields_set__:
            updates["emit_telemetry"] = runtime.emit_telemetry
        return self.model_copy(update=updates, deep=True)
