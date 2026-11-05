# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder-pattern configuration layer for Safe Synthesizer."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, TypeAlias, TypeVar

import pandas as pd

from ..cli.datasets import DatasetInfo, file_extension
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
from ..configurator.parameters import Parameters
from ..data_processing.csv_source import CsvSource
from ..data_processing.validation import strip_column_names
from ..observability import get_logger
from ..telemetry import _telemetry_enabled

logger = get_logger(__name__)

DataSource = pd.DataFrame | str | CsvSource
# File formats read with their own pandas reader instead of as CSV.
_TYPED_FILE_EXTENSIONS = frozenset({"parquet", "json", "jsonl"})
_SectionT = TypeVar("_SectionT", bound=Parameters)
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
        self._data_source_csv: CsvSource | None = None
        # Column names trimmed by ``_resolve_datasource``, from original to trimmed name.
        self._column_renames: dict[str, str] = {}
        self._hf_token_secret: str | None = None

    @property
    def _effective_unknown_fields(self) -> UnknownFieldBehavior:
        if self._unknown_fields_override is not None:
            return self._unknown_fields_override
        if self._nss_config is not None:
            return self._nss_config.unknown_fields
        return DEFAULT_UNKNOWN_FIELDS

    def _overlay_section(
        self,
        model_type: type[_SectionT],
        current: _SectionT | None,
        config: _SectionT | RawConfig | None,
        kwargs: Mapping[str, object],
    ) -> _SectionT:
        """Overlay a section source and keyword overrides on the section's current values.

        Only fields that ``config`` or ``kwargs`` set change, so a base configuration passed to the builder keeps
        its other values, and the result stays sparse. ``current`` is ``None`` for a disabled optional section,
        which then starts from the defaults.
        """
        update = model_type.from_config_source(
            config,
            unknown_field_behavior=self._effective_unknown_fields,
            **kwargs,
        )
        if current is None:
            return update
        return update.explicit_patch().apply(current)

    def with_data_source(self, df_source: DataSource) -> Self:
        """Set the data source for synthetic data generation.

        Args:
            df_source: Training dataset as a pandas DataFrame, a path or URL
                to a CSV, Parquet, JSON, or JSONL file, or a ``CsvSource``
                already read from a CSV file.

        Returns:
            This builder instance with the data source configured.
        """
        self._data_source = df_source
        self._data_source_csv = None
        self._column_renames = {}
        return self

    def with_data(self, config: DataParameters | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure data processing settings.

        Args:
            config: Data configuration object or raw mapping. Fields that neither it
                nor ``kwargs`` set keep their current values.
            **kwargs: Field-level overrides (e.g. ``holdout_size``).

        Returns:
            This builder instance with data processing settings applied.
        """
        self._data_config = self._overlay_section(DataParameters, self._data_config, config, kwargs)
        return self

    def with_train(self, config: TrainingHyperparams | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure training hyperparameters.

        Args:
            config: Training configuration object or raw mapping. Fields that neither it
                nor ``kwargs`` set keep their current values.
            **kwargs: Field-level overrides (e.g. ``learning_rate``).

        Returns:
            This builder instance with training hyperparameters applied.
        """
        self._training_config = self._overlay_section(TrainingHyperparams, self._training_config, config, kwargs)
        return self

    def with_generate(self, config: GenerateParameters | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure generation settings.

        Args:
            config: Generation configuration object or raw mapping. Fields that neither it
                nor ``kwargs`` set keep their current values.
            **kwargs: Field-level overrides (e.g. ``num_records``).

        Returns:
            This builder instance with generation settings applied.
        """
        self._generation_config = self._overlay_section(GenerateParameters, self._generation_config, config, kwargs)
        return self

    def with_time_series(self, config: TimeSeriesParameters | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure time-series synthesis settings.

        Args:
            config: Time-series configuration object or raw mapping. Fields that neither it
                nor ``kwargs`` set keep their current values.
            **kwargs: Field-level overrides (e.g. ``time_column``).

        Returns:
            This builder instance with time-series synthesis settings applied.
        """
        self._time_series_config = self._overlay_section(TimeSeriesParameters, self._time_series_config, config, kwargs)
        return self

    def with_differential_privacy(
        self, config: DifferentialPrivacyHyperparams | RawConfig | None = None, **kwargs: object
    ) -> Self:
        """Configure differential privacy settings.

        Args:
            config: DP configuration object or raw mapping. Fields that neither it
                nor ``kwargs`` set keep their current values.
            **kwargs: Field-level overrides (e.g. ``epsilon``).

        Returns:
            This builder instance with differential privacy settings applied.
        """
        self._privacy_config = self._overlay_section(
            DifferentialPrivacyHyperparams, self._privacy_config, config, kwargs
        )
        return self

    def with_replace_pii(
        self, config: ReplacePiiConfig | RawConfig | None = None, *, enable: bool = True, **kwargs: object
    ) -> Self:
        """Request or disable PII replacement.

        The configuration drives :meth:`SafeSynthesizer.plan_pii_replacement`.
        Pass ``enable=False`` to set ``replace_pii=None`` and run the synthesis
        pipeline without replacement.

        Args:
            config: PII replacement configuration object or raw mapping. Fields that neither it
                nor ``kwargs`` set keep their current values, or their defaults when replacement was
                disabled.
            enable: When ``False``, disables PII replacement entirely
                and clears any previously set config.
            **kwargs: Field-level replacement configuration overrides.

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

        if config is not None and not isinstance(config, ReplacePiiConfig | Mapping):
            raise ValueError(f"Config must be a ReplacePiiConfig, raw mapping, or None, got {config!r}")
        self._replace_pii_config = self._overlay_section(ReplacePiiConfig, self._replace_pii_config, config, kwargs)
        return self

    def with_evaluate(self, config: EvaluationParameters | RawConfig | None = None, **kwargs: object) -> Self:
        """Configure evaluation settings.

        Args:
            config: Evaluation configuration object or raw mapping. Fields that neither it
                nor ``kwargs`` set keep their current values.
            **kwargs: Field-level overrides (e.g. ``enabled``).

        Returns:
            This builder instance with evaluation settings applied.
        """
        self._evaluation_config = self._overlay_section(EvaluationParameters, self._evaluation_config, config, kwargs)
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

        Constructs the unified config from already-normalized typed sections.
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

    def _resolve_datasource(self, **kwargs) -> None:
        """Convert the data source into a ``pandas.DataFrame``.

        If ``_data_source`` is already a DataFrame it is kept, with non-string column names converted to
        strings. Leading and trailing spaces are removed from every source's column names (see
        ``strip_column_names``), so a CSV header written as ``a, b`` gives columns ``a`` and ``b``. A string
        ending in ``.parquet``, ``.json``, or ``.jsonl`` is read with the
        matching pandas reader, as the CLI does. Any other string is treated as
        a CSV path or URL and read once into a ``CsvSource``, which is kept so
        the standalone PII replacement output can write unchanged cells exactly
        as they appear in the same bytes.

        Args:
            **kwargs: Forwarded to the pandas reader when loading from a path or URL.

        Raises:
            ValueError: If ``_data_source`` is not a DataFrame, string, or ``CsvSource``.
            DataError: If two column names are the same after removing their spaces.
        """
        match self._data_source:
            case pd.DataFrame():
                pass
            case str(url) if file_extension(url) in _TYPED_FILE_EXTENSIONS:
                self._data_source: pd.DataFrame = DatasetInfo(name=url, url=url, load_args=kwargs or None).fetch()
            case str(url):
                self._data_source_csv = CsvSource.load(url, kwargs)
                self._data_source = self._data_source_csv.read()
            case CsvSource() as source:
                self._data_source_csv = source
                self._data_source = source.read()
            case _:
                raise ValueError("Data source must be a pandas DataFrame or a URL")
        if not all(isinstance(column, str) for column in self._data_source.columns):
            # Configurations and PII plans name columns as text, so a column named 0 would never match "0".
            self._data_source = self._data_source.rename(columns=str)
        trimmed = strip_column_names(self._data_source)
        renames = {
            before: after
            for before, after in zip(self._data_source.columns, trimmed.columns, strict=True)
            if before != after
        }
        if renames:
            examples = ", ".join(f"{before!r} -> {after!r}" for before, after in list(renames.items())[:3])
            logger.user.warning(
                f"Removed leading and trailing spaces from {len(renames)} column names, such as "
                f"{examples}; outputs use the trimmed names."
            )
            self._data_source = trimmed
            # Kept until the next with_data_source: resolving again finds the names already trimmed.
            self._column_renames = renames
