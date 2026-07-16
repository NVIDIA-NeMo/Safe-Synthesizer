# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Self, TypeAlias, cast

from pydantic import Field, TypeAdapter, model_validator
from typing_extensions import override

from ..configurator.parameter_paths import (
    ParameterPath,
    ParameterSchema,
)
from ..configurator.parameters import Parameters
from ..errors import ParameterError
from ..observability import get_logger
from ..telemetry import _telemetry_enabled
from .data import DataParameters
from .differential_privacy import DifferentialPrivacyHyperparams
from .evaluate import EvaluationParameters
from .generate import GenerateParameters
from .patch import CompiledConfigPatch, PatchAssignment
from .preflight import PreflightParameters
from .replace_pii import PiiReplacerConfig
from .time_series import TimeSeriesParameters
from .training import TrainingHyperparams
from .types import AUTO_STR

ConfigPatch: TypeAlias = Mapping[str, object]

__all__ = ["ConfigPatch", "SafeSynthesizerParameters"]


logger = get_logger(__name__)


_ExtraBehavior = Literal["ignore", "forbid"]


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

    strict_config: bool = Field(
        default=True,
        description=(
            "Whether unknown configuration keys are rejected recursively. "
            "Disable only when compatibility across mismatched client and service versions is required."
        ),
    )

    _strict_config_adapter: ClassVar[TypeAdapter[bool]] = TypeAdapter(bool)

    @classmethod
    def _strict_config_enabled(cls, value: object = True) -> bool:
        """Validate and resolve the user-facing unknown-field policy flag."""
        return cls._strict_config_adapter.validate_python(value)

    @classmethod
    def _strict_config_from_mapping(cls, source: Mapping[str, object], *, default: bool = True) -> bool:
        return cls._strict_config_enabled(source.get("strict_config", default))

    @classmethod
    def _extra_behavior(cls, source: object, *, default: bool = True) -> _ExtraBehavior:
        if isinstance(source, SafeSynthesizerParameters):
            enabled = source.strict_config
        elif isinstance(source, Mapping):
            enabled = cls._strict_config_from_mapping(cast(Mapping[str, object], source), default=default)
        else:
            enabled = default
        return "forbid" if enabled else "ignore"

    def __init__(self, /, **data: Any) -> None:
        """Validate constructor input with the input's recursive unknown-field policy."""
        self.__pydantic_validator__.validate_python(
            data,
            self_instance=self,
            extra=type(self)._extra_behavior(data),
        )

    # Pydantic marks its own BaseModel initializer this way so the model
    # metaclass does not treat this equivalent initializer as a custom one.
    cast(Any, __init__).__pydantic_base_init__ = True

    @classmethod
    @override
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        extra: Literal["allow", "ignore", "forbid"] | None = None,
        from_attributes: bool | None = None,
        context: Any | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Self:
        """Validate input using ``strict_config`` as the recursive extras policy."""
        return super().model_validate(
            obj,
            strict=strict,
            extra=extra if extra is not None else cls._extra_behavior(obj),
            from_attributes=from_attributes,
            context=context,
            by_alias=by_alias,
            by_name=by_name,
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
    def from_params(cls, **kwargs: object) -> "SafeSynthesizerParameters":
        """Construct parameters from resolved keyword names.

        Names may be top-level fields, canonical dotted paths, unique bare
        names, or supported legacy aliases. Ambiguous bare names raise an error
        that lists the canonical dotted alternatives.

        Args:
            **kwargs: Values keyed by a supported parameter name.

        Returns:
            A validated configuration with unspecified fields defaulted.

        Example:
            >>> from nemo_safe_synthesizer.config import SafeSynthesizerParameters
            >>> SafeSynthesizerParameters.from_params(num_records=2000)
        """
        schema = ParameterSchema.from_model(cls)
        assignments: list[PatchAssignment] = []
        resolved_paths: set[ParameterPath] = set()
        for name, value in kwargs.items():
            if (path := schema.require(name)) in resolved_paths:
                raise ParameterError(f"Duplicate parameter path {str(path)!r}.")
            resolved_paths.add(path)
            assignments.append(PatchAssignment(path, value, f"parameter {name!r}", 0))

        return CompiledConfigPatch.from_paths(cls, assignments).apply()

    @classmethod
    @override
    def from_config_source(
        cls,
        source: Parameters | Mapping[str, object] | None = None,
        *,
        unknown_fields: Literal["ignore", "reject"] | None = None,
        **kwargs: object,
    ) -> Self:
        """Normalize a source using its effective ``strict_config`` policy."""
        if unknown_fields is None:
            if isinstance(source, Mapping):
                enabled = (
                    cls._strict_config_enabled(kwargs["strict_config"])
                    if "strict_config" in kwargs
                    else cls._strict_config_from_mapping(cast(Mapping[str, object], source))
                )
            elif isinstance(source, SafeSynthesizerParameters):
                enabled = source.strict_config
            else:
                enabled = cls._strict_config_enabled(kwargs.get("strict_config", True))
            unknown_fields = "reject" if enabled else "ignore"
        return cast(Self, super().from_config_source(source, unknown_fields=unknown_fields, **kwargs))

    @classmethod
    def from_config_patch(cls, patch: ConfigPatch) -> Self:
        """Validate a sparse top-level config patch as a full configuration."""
        normalized = ParameterSchema.from_model(cls).normalize_aliases(patch)
        unknown_fields = "reject" if cls._strict_config_from_mapping(normalized) else "ignore"
        return CompiledConfigPatch.from_mapping(
            cls, normalized, origin="config patch", precedence=0, unknown_fields=unknown_fields
        ).apply()

    def with_config_patch(self, patch: ConfigPatch) -> Self:
        """Apply a sparse top-level config patch and revalidate the result.

        Only fields explicitly set on ``self`` are carried into the merge before
        applying ``patch``. This preserves file/CLI precedence while keeping
        default values implicit for future ``exclude_unset`` dumps.
        """
        model_type = type(self)
        base = CompiledConfigPatch.from_mapping(
            model_type,
            self.model_dump(exclude_unset=True),
            origin="base config",
            precedence=0,
            unknown_fields="reject",
        )
        normalized = ParameterSchema.from_model(model_type).normalize_aliases(patch)
        strict_config = model_type._strict_config_from_mapping(normalized, default=self.strict_config)
        override = CompiledConfigPatch.from_mapping(
            model_type,
            normalized,
            origin="config patch",
            precedence=1,
            unknown_fields="reject" if strict_config else "ignore",
        )
        return base.combine(override).apply()

    def with_runtime_overrides(self, runtime: SafeSynthesizerParameters) -> "SafeSynthesizerParameters":
        """Apply supported resume-time overrides onto a copy of self.

        ``self`` is the saved training-run config. Only explicitly-set
        ``generation`` and ``evaluation`` fields from ``runtime`` are merged in,
        plus ``emit_telemetry`` and ``strict_config`` when the caller set them.
        Training, data, privacy, and other sections are preserved so training
        provenance survives a generate-only resume.

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

        def _add_section(name: str, section: Parameters) -> None:
            if (materialized := section.explicit_patch().materialize()) or name in runtime.model_fields_set:
                updates[name] = materialized

        _add_section("generation", runtime.generation)
        _add_section("evaluation", runtime.evaluation)
        if "emit_telemetry" in runtime.model_fields_set:
            updates["emit_telemetry"] = runtime.emit_telemetry
        if "strict_config" in runtime.model_fields_set:
            updates["strict_config"] = runtime.strict_config
        patch = CompiledConfigPatch.from_mapping(
            type(self),
            updates,
            origin="runtime override",
            precedence=1,
            unknown_fields="reject",
        )
        return self.apply_patch(patch)
