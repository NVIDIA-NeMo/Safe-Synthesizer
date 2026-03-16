# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builder-pattern configuration layer for Safe Synthesizer.

Provides ``ConfigBuilder``, the base builder that accumulates
per-section configuration objects (training, generation, data, etc.)
via fluent ``with_*`` methods before resolving them into a single
``SafeSynthesizerParameters``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Self, TypeAlias, TypeVar

import pandas as pd
from pydantic import BaseModel

from ..config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    EvaluationParameters,
    GenerateParameters,
    PiiReplacerConfig,
    SafeSynthesizerParameters,
    TimeSeriesParameters,
    TrainingHyperparams,
)
from ..observability import get_logger

logger = get_logger(__name__)


KT = TypeVar("KT")
VT = TypeVar("VT")

NSSParameters = (
    DataParameters
    | EvaluationParameters
    | GenerateParameters
    | DifferentialPrivacyHyperparams
    | TimeSeriesParameters
    | TrainingHyperparams
    | SafeSynthesizerParameters
    | PiiReplacerConfig
)

NSSParametersT = (
    type[DataParameters]
    | type[EvaluationParameters]
    | type[GenerateParameters]
    | type[DifferentialPrivacyHyperparams]
    | type[TimeSeriesParameters]
    | type[TrainingHyperparams]
    | type[SafeSynthesizerParameters]
    | type[PiiReplacerConfig]
)


ParamT = TypeVar("ParamT", bound=NSSParameters)
DataSource = pd.DataFrame | str
ParamDict: TypeAlias = dict[str, str | int | float | bool | None | Mapping[KT, VT]]


class ConfigBuilder(object):
    """Fluent builder for assembling Safe Synthesizer configuration.

    Accumulates per-section configuration objects (data, training,
    generation, evaluation, privacy, PII replacement, and time-series)
    via ``with_*`` methods.  Call ``resolve()`` (or let
    ``SafeSynthesizer`` do it) to collapse them into a single
    ``SafeSynthesizerParameters``.

    Each ``with_*`` method accepts an optional typed config object or
    a plain dict, plus ``**kwargs`` overrides.  ``kwargs`` always take
    precedence over fields in the config/dict.  All ``with_*`` methods
    return ``self`` for chaining.

    Args:
        config: Optional pre-built parameters.  When supplied, the
            individual ``_*_config`` attributes are seeded from its
            sections.
    """

    def __init__(self, config: SafeSynthesizerParameters | None = None) -> None:
        self._nss_config: SafeSynthesizerParameters | None = config
        if self._nss_config is not None:
            self._evaluation_config = self._nss_config.evaluation
            self._replace_pii_config = self._nss_config.replace_pii
            self._privacy_config: DifferentialPrivacyHyperparams | None = self._nss_config.privacy
            self._training_config = self._nss_config.training
            self._generation_config = self._nss_config.generation
            self._data_config = self._nss_config.data
            self._time_series_config = self._nss_config.time_series
        else:
            self._data_config: DataParameters = DataParameters()
            self._evaluation_config: EvaluationParameters = EvaluationParameters()
            self._generation_config: GenerateParameters = GenerateParameters()
            self._replace_pii_config: PiiReplacerConfig | None = PiiReplacerConfig.get_default_config()
            self._privacy_config: DifferentialPrivacyHyperparams = DifferentialPrivacyHyperparams()
            self._training_config: TrainingHyperparams = TrainingHyperparams()
            self._time_series_config: TimeSeriesParameters = TimeSeriesParameters()

        self._data_source: DataSource | None = None
        self._classify_model_provider: str | None = None
        self._hf_token_secret: str | None = None
        self._nss_inputs: list[str] = [
            "_data_config",
            "_evaluation_config",
            "_generation_config",
            "_replace_pii_config",
            "_privacy_config",
            "_training_config",
            "_time_series_config",
        ]

    def _resolve_config(self, values: ParamDict | NSSParameters | None, cls: NSSParametersT, **kwargs) -> NSSParameters:
        """Resolve configuration from various input types.

        Precedence: ``kwargs`` override ``values``; ``values`` override
        model defaults.

        Args:
            values: Existing config, a raw dict, or ``None`` for
                defaults-only.
            cls: The Pydantic model class to validate against.
            **kwargs: Field-level overrides applied on top.

        Returns:
            A validated config instance of type ``cls``.
        """
        overrides = kwargs
        match values:
            case BaseModel() as model:
                return model.model_copy(update=overrides)
            case dict() as d:
                return cls.model_validate(d).model_copy(update=overrides)
            case None:
                return cls(**overrides)

    def with_data_source(self, df_source: DataSource) -> Self:
        """Set the data source for synthetic data generation.

        Args:
            df_source: Training dataset as a pandas DataFrame or a fetchable URL.

        Returns:
            This builder instance with the data source configured.
        """
        self._data_source = df_source
        return self

    def with_data(self, config: DataParameters | ParamDict | None = None, **kwargs) -> Self:
        """Configure data processing settings.

        Args:
            config: Data configuration object or dict.
            **kwargs: Field-level overrides (e.g. ``holdout_size``).

        Returns:
            This builder instance with data processing settings applied.
        """
        self._data_config: DataParameters | None = self._resolve_config(values=config, cls=DataParameters, **kwargs)
        return self

    def with_train(self, config: TrainingHyperparams | ParamDict | None = None, **kwargs) -> Self:
        """Configure training hyperparameters.

        Args:
            config: Training configuration object or dict.
            **kwargs: Field-level overrides (e.g. ``learning_rate``).

        Returns:
            This builder instance with training hyperparameters applied.
        """
        self._training_config: TrainingHyperparams | None = self._resolve_config(
            values=config, cls=TrainingHyperparams, **kwargs
        )
        return self

    def with_generate(self, config: GenerateParameters | ParamDict | None = None, **kwargs) -> Self:
        """Configure generation settings.

        Args:
            config: Generation configuration object or dict.
            **kwargs: Field-level overrides (e.g. ``num_records``).

        Returns:
            This builder instance with generation settings applied.
        """
        self._generation_config: GenerateParameters | None = self._resolve_config(
            values=config, cls=GenerateParameters, **kwargs
        )
        return self

    def with_time_series(self, config: TimeSeriesParameters | ParamDict | None = None, **kwargs) -> Self:
        """Configure time-series synthesis settings.

        Args:
            config: Time-series configuration object or dict.
            **kwargs: Field-level overrides (e.g. ``time_column``).

        Returns:
            This builder instance with time-series synthesis settings applied.
        """
        self._time_series_config: TimeSeriesParameters | None = self._resolve_config(
            values=config, cls=TimeSeriesParameters, **kwargs
        )
        return self

    def with_differential_privacy(
        self, config: DifferentialPrivacyHyperparams | ParamDict | None = None, **kwargs
    ) -> Self:
        """Configure differential privacy settings.

        Args:
            config: DP configuration object or dict.
            **kwargs: Field-level overrides (e.g. ``epsilon``).

        Returns:
            This builder instance with differential privacy settings applied.
        """
        self._privacy_config: DifferentialPrivacyHyperparams | None = self._resolve_config(
            values=config, cls=DifferentialPrivacyHyperparams, **kwargs
        )
        return self

    def with_replace_pii(
        self, config: PiiReplacerConfig | ParamDict | None = None, *, enable: bool = True, **kwargs
    ) -> Self:
        """Configure PII replacement settings.

        Falls back to ``PiiReplacerConfig.get_default_config()`` when
        ``config`` is ``None``.  Pass ``enable=False`` to explicitly
        disable PII replacement for this run -- this sets
        ``replace_pii=None``, which is the sole disabled signal.

        Note: PII replacement uses ``replace_pii=None`` as the disabled
        signal rather than a ``PiiReplacerConfig.enabled`` boolean field.
        This differs from ``EvaluationConfig.enabled`` but is intentional:
        ``PiiReplacerConfig`` has a non-trivial ``default_factory`` that
        must fire when the field is absent from a YAML config.  Adding an
        ``enabled`` boolean inside the sub-config would require a
        ``model_validator`` to reconcile the two signals and would not
        interact cleanly with Pydantic's ``exclude_unset`` semantics used
        in ``from_params``.

        Args:
            config: PII replacement configuration object or dict.
            enable: When ``False``, disables PII replacement entirely
                and clears any previously set config.
            **kwargs: Field-level overrides (e.g. ``classify``).

        Returns:
            This builder instance with PII replacement configured.

        Raises:
            ValueError: If ``config`` is not a ``PiiReplacerConfig``,
                dict, or ``None``.

        Example::

            builder = SafeSynthesizer().with_data_source(your_dataframe).with_replace_pii(config=custom_pii_config)
        """
        if not enable:
            self._replace_pii_config = None
            return self

        cfg = None
        match config:
            case PiiReplacerConfig() as m:
                cfg = m.model_copy(update=kwargs, deep=True)
            case dict() as d:
                cfg = PiiReplacerConfig.model_validate(d).model_copy(update=kwargs, deep=True)
            case None:
                cfg = PiiReplacerConfig.get_default_config().model_copy(update=kwargs, deep=True)
            case _:
                raise ValueError(f"Config must be a PiiReplacerConfig, dict, or None, got {config!r}")

        self._replace_pii_config = cfg
        return self

    def with_evaluate(self, config: EvaluationParameters | ParamDict | None = None, **kwargs) -> Self:
        """Configure evaluation settings.

        Args:
            config: Evaluation configuration object or dict.
            **kwargs: Field-level overrides (e.g. ``enabled``).

        Returns:
            This builder instance with evaluation settings applied.
        """
        self._evaluation_config: EvaluationParameters | None = self._resolve_config(
            values=config, cls=EvaluationParameters, **kwargs
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

        Iterates over ``_nss_inputs``, maps each ``_*_config`` attribute
        to its ``SafeSynthesizerParameters`` field name, and constructs
        the unified config.  Also injects ``_classify_model_provider``
        into the PII replacer config when set.
        """
        params_map: dict = {k: k.split("_")[1] for k in self._nss_inputs}
        params_map["_replace_pii_config"] = "replace_pii"
        params_map["_time_series_config"] = "time_series"
        params_to_use: dict = {k: None for k in params_map.values()}

        for pg, name in params_map.items():
            param: NSSParameters | None = getattr(self, pg, None)
            match param:
                case BaseModel() as c:
                    params_to_use[name] = c
                case dict() as d:
                    params_to_use[name] = d
                case None:
                    logger.debug(f"Using default values for {pg}")
                case _:
                    raise ValueError(f"Input must be a BaseModel, dictionary, or None: {type(param)}")
        self._nss_config = SafeSynthesizerParameters(**params_to_use)

        # Inject classify_model_provider into PII replacer config if set
        if self._classify_model_provider and self._nss_config.replace_pii:
            self._nss_config.replace_pii.globals.classify.classify_model_provider = self._classify_model_provider
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
