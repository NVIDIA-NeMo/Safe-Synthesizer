# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder-pattern configuration layer for Safe Synthesizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, TypeAlias

import pandas as pd

from ..config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    EvaluationParameters,
    GenerateParameters,
    PreflightParameters,
    ReplacePiiConfig,
    SafeSynthesizerParameters,
    TimeSeriesParameters,
    TrainingHyperparams,
)
from ..config.unknown_fields import (
    DEFAULT_UNKNOWN_FIELDS,
    UnknownFieldBehavior,
    validate_unknown_fields,
)
from ..observability import get_logger
from ..telemetry import _telemetry_enabled

logger = get_logger(__name__)


DataSource = pd.DataFrame | str
RawConfig: TypeAlias = Mapping[str, object]


class ConfigBuilder:
    """Fluent builder for assembling Safe Synthesizer configuration.

    Accumulates per-section configuration objects (data, training,
    generation, evaluation, privacy, PII replacement, and time-series)
    via ``with_*`` methods.  Call ``resolve()`` (or let
    ``SafeSynthesizer`` do it) to collapse them into a single
    ``SafeSynthesizerParameters``.

    Each ``with_*`` method accepts an optional sparse typed config or raw
    mapping, plus ``**kwargs`` overrides. Keyword arguments take precedence
    over fields in the source. All ``with_*`` methods return ``Self`` so
    subclasses preserve their concrete type through fluent chains.

    Args:
        config: Optional pre-built parameters.  When supplied, the
            individual ``_*_config`` attributes are seeded from its
            sections.
        unknown_fields: Optional SDK-wide override for raw mapping validation.
            Set it at construction time so every section uses the same policy.
    """

    def __init__(
        self,
        config: SafeSynthesizerParameters | None = None,
        *,
        unknown_fields: UnknownFieldBehavior | None = None,
    ) -> None:
        self._nss_config = config.model_copy(deep=True) if config is not None else None
        self._unknown_fields_override = validate_unknown_fields(unknown_fields) if unknown_fields is not None else None
        if self._nss_config is not None:
            self._emit_telemetry_config = self._nss_config.emit_telemetry
            self._evaluation_config = self._nss_config.evaluation
            self._replace_pii_config = self._nss_config.replace_pii
            self._preflight_config = self._nss_config.preflight
            self._privacy_config: DifferentialPrivacyHyperparams | None = self._nss_config.privacy
            self._training_config = self._nss_config.training
            self._generation_config = self._nss_config.generation
            self._data_config = self._nss_config.data
            self._time_series_config = self._nss_config.time_series
        else:
            self._data_config: DataParameters = DataParameters()
            self._evaluation_config: EvaluationParameters = EvaluationParameters()
            self._generation_config: GenerateParameters = GenerateParameters()
            self._replace_pii_config: ReplacePiiConfig | None = ReplacePiiConfig()
            self._preflight_config = PreflightParameters()
            self._privacy_config: DifferentialPrivacyHyperparams = DifferentialPrivacyHyperparams()
            self._training_config: TrainingHyperparams = TrainingHyperparams()
            self._time_series_config: TimeSeriesParameters = TimeSeriesParameters()
            self._emit_telemetry_config = _telemetry_enabled()

        self._data_source: DataSource | None = None
        self._classify_model_provider: str | None = None
        self._hf_token_secret: str | None = None

    @property
    def _effective_unknown_fields(self) -> UnknownFieldBehavior:
        if self._unknown_fields_override is not None:
            return self._unknown_fields_override
        if self._nss_config is not None:
            return self._nss_config.unknown_fields
        return DEFAULT_UNKNOWN_FIELDS

    def with_data_source(self, df_source: DataSource) -> Self:
        """Set the data source for synthetic data generation.

        Args:
            df_source: Training dataset as a pandas DataFrame or a fetchable URL.

        Returns:
            This builder instance with the data source configured.
        """
        self._data_source = df_source
        return self

    def with_data(self, config: DataParameters | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure data processing settings.

        Args:
            config: Data configuration object or raw mapping.
            **kwargs: Field-level overrides (e.g. ``holdout_size``).

        Returns:
            This builder instance with data processing settings applied.
        """
        self._data_config = DataParameters.from_config_source(
            config,
            unknown_field_behavior=self._effective_unknown_fields,
            **kwargs,
        )
        return self

    def with_train(self, config: TrainingHyperparams | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure training hyperparameters.

        Args:
            config: Training configuration object or raw mapping.
            **kwargs: Field-level overrides (e.g. ``learning_rate``).

        Returns:
            This builder instance with training hyperparameters applied.
        """
        self._training_config = TrainingHyperparams.from_config_source(
            config,
            unknown_field_behavior=self._effective_unknown_fields,
            **kwargs,
        )
        return self

    def with_generate(self, config: GenerateParameters | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure generation settings.

        Args:
            config: Generation configuration object or raw mapping.
            **kwargs: Field-level overrides (e.g. ``num_records``).

        Returns:
            This builder instance with generation settings applied.
        """
        self._generation_config = GenerateParameters.from_config_source(
            config,
            unknown_field_behavior=self._effective_unknown_fields,
            **kwargs,
        )
        return self

    def with_time_series(self, config: TimeSeriesParameters | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure time-series synthesis settings.

        Args:
            config: Time-series configuration object or raw mapping.
            **kwargs: Field-level overrides (e.g. ``time_column``).

        Returns:
            This builder instance with time-series synthesis settings applied.
        """
        self._time_series_config = TimeSeriesParameters.from_config_source(
            config,
            unknown_field_behavior=self._effective_unknown_fields,
            **kwargs,
        )
        return self

    def with_differential_privacy(
        self, config: DifferentialPrivacyHyperparams | RawConfig | None = None, **kwargs: object
    ) -> Self:
        """Configure differential privacy settings.

        Args:
            config: DP configuration object or raw mapping.
            **kwargs: Field-level overrides (e.g. ``epsilon``).

        Returns:
            This builder instance with differential privacy settings applied.
        """
        self._privacy_config = DifferentialPrivacyHyperparams.from_config_source(
            config,
            unknown_field_behavior=self._effective_unknown_fields,
            **kwargs,
        )
        return self

    def with_replace_pii(
        self, config: ReplacePiiConfig | RawConfig | None = None, *, enable: bool = True, **kwargs: object
    ) -> Self:
        """Configure PII replacement settings.

        Falls back to the default ``ReplacePiiConfig()`` (auto-discovery)
        when ``config`` is ``None``.  Pass ``enable=False`` to explicitly
        disable PII replacement for this run -- this sets
        ``replace_pii=None``, which is the sole disabled signal.

        Note: PII replacement uses ``replace_pii=None`` as the disabled
        signal rather than a ``ReplacePiiConfig.enabled`` boolean field.
        This differs from ``EvaluationConfig.enabled`` but is intentional:
        ``replace_pii`` has a non-trivial ``default_factory`` that must
        fire when the field is absent from a YAML config.  Adding an
        ``enabled`` boolean inside the sub-config would require a
        ``model_validator`` to reconcile the two signals and would not
        interact cleanly with Pydantic's ``exclude_unset`` semantics used
        in ``from_params``.

        Args:
            config: PII replacement configuration object or raw mapping.
            enable: When ``False``, disables PII replacement entirely
                and clears any previously set config.
            **kwargs: Field-level overrides (e.g. ``replacement``, ``person``).

        Returns:
            This builder instance with PII replacement configured.

        Raises:
            ValueError: If ``config`` is not a ``ReplacePiiConfig``,
                raw mapping, or ``None``.

        Example::

            builder = SafeSynthesizer().with_data_source(your_dataframe).with_replace_pii(config=custom_pii_config)
        """
        if not enable:
            self._replace_pii_config = None
            return self

        match config:
            case ReplacePiiConfig() | Mapping() as values:
                cfg = ReplacePiiConfig.from_config_source(
                    values,
                    unknown_field_behavior=self._effective_unknown_fields,
                    **kwargs,
                )
            case None:
                cfg = ReplacePiiConfig.from_config_source(
                    ReplacePiiConfig(),
                    unknown_field_behavior=self._effective_unknown_fields,
                    **kwargs,
                )
            case _:
                raise ValueError(f"Config must be a ReplacePiiConfig, raw mapping, or None, got {config!r}")

        self._replace_pii_config = cfg
        return self

    def with_evaluate(self, config: EvaluationParameters | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure evaluation settings.

        Args:
            config: Evaluation configuration object or raw mapping.
            **kwargs: Field-level overrides (e.g. ``enabled``).

        Returns:
            This builder instance with evaluation settings applied.
        """
        self._evaluation_config = EvaluationParameters.from_config_source(
            config,
            unknown_field_behavior=self._effective_unknown_fields,
            **kwargs,
        )
        return self

    def resolve(self) -> Self:
        """Finalize configuration and data source.

        Assembles the individual ``_*_config`` sections into a single
        ``SafeSynthesizerParameters`` and converts the data source
        (URL string or DataFrame) into a ``DataFrame``.

        Returns:
            This builder instance with all configuration sections finalized.
        """
        self._resolve_nss_config()
        self._resolve_datasource()
        return self

    def _resolve_nss_config(self) -> None:
        """Assemble per-section configs into a ``SafeSynthesizerParameters``.

        Constructs the unified config from already-normalized typed sections,
        then injects ``_classify_model_provider`` into PII configuration when
        requested.
        """
        config_values: dict[str, object] = {
            "data": self._data_config,
            "evaluation": self._evaluation_config,
            "training": self._training_config,
            "generation": self._generation_config,
            "privacy": self._privacy_config,
            "time_series": self._time_series_config,
            "replace_pii": self._replace_pii_config,
            "preflight": self._preflight_config,
            "emit_telemetry": self._emit_telemetry_config,
            "unknown_fields": self._effective_unknown_fields,
        }
        self._nss_config = SafeSynthesizerParameters.model_validate(config_values)

        # Inject classify_model_provider into PII replacer config if set
        if self._classify_model_provider and self._nss_config.replace_pii:
            self._nss_config.replace_pii.llm.model_provider = self._classify_model_provider
            logger.debug(f"Injected classify model provider into PII config: {self._classify_model_provider}")

    def _resolve_datasource(self, **kwargs) -> None:
        """Convert the data source into a ``pandas.DataFrame``.

        If ``_data_source`` is already a DataFrame it is kept as-is.
        A string is treated as a CSV URL and fetched via
        ``pd.read_csv``.

        Args:
            **kwargs: Forwarded to ``pd.read_csv`` when loading from URL.

        Raises:
            ValueError: If ``_data_source`` is not a DataFrame or string.
        """
        match self._data_source:
            case pd.DataFrame():
                pass
            case str(url):
                self._data_source: pd.DataFrame = pd.read_csv(url, **kwargs)
            case _:
                raise ValueError("Data source must be a pandas DataFrame or a URL")
