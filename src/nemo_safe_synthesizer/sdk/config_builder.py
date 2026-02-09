# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Mapping, Self, TypeAlias, TypeVar, Union

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

NSSParameters = Union[
    DataParameters,
    EvaluationParameters,
    GenerateParameters,
    DifferentialPrivacyHyperparams,
    TimeSeriesParameters,
    TrainingHyperparams,
    SafeSynthesizerParameters,
    PiiReplacerConfig,
]

NSSParametersT = Union[
    type[DataParameters],
    type[EvaluationParameters],
    type[GenerateParameters],
    type[DifferentialPrivacyHyperparams],
    type[TimeSeriesParameters],
    type[TrainingHyperparams],
    type[SafeSynthesizerParameters],
    type[PiiReplacerConfig],
]


ParamT = TypeVar("ParamT", bound=NSSParameters)
DataSource = pd.DataFrame | str
ParamDict: TypeAlias = dict[str, Union[str, int, float, bool, None, Mapping[KT, VT]]]


class ConfigBuilder(object):
    def __init__(self, config: SafeSynthesizerParameters | None = None) -> None:
        self._nss_config: SafeSynthesizerParameters | None = config
        if self._nss_config is not None:
            self._evaluation_config = self._nss_config.evaluation
            self._enable_synthesis = self._nss_config.enable_synthesis
            self._enable_replace_pii = self._nss_config.enable_replace_pii
            self._replace_pii_config = self._nss_config.replace_pii
            self._privacy_config: DifferentialPrivacyHyperparams | None = self._nss_config.privacy
            self._training_config = self._nss_config.training
            self._generation_config = self._nss_config.generation
            self._data_config = self._nss_config.data
            self._time_series_config = self._nss_config.time_series
        else:
            self._enable_synthesis = None
            self._enable_replace_pii = None
            self._data_config: DataParameters = DataParameters()
            self._evaluation_config: EvaluationParameters = EvaluationParameters()
            self._generation_config: GenerateParameters = GenerateParameters()
            self._replace_pii_config: PiiReplacerConfig | None = None
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
        Args:
            values: Configuration values as a dictionary or a BaseModel instance.
            cls: The BaseModel class to validate against.
            **overrides: Additional configuration parameters to override.
        Returns:
            An instance of the specified BaseModel class with the resolved configuration.
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
            The current Safe Synthesizer builder instance.
        """
        self._data_source = df_source
        return self

    def synthesize(self) -> Self:
        """Enables synthesis for the job run.

        Use if not setting training or generation parameters directly
        """
        self._enable_synthesis = True
        return self

    def with_data(self, config: DataParameters | ParamDict | None = None, **kwargs) -> Self:
        """Configure  settings."""
        self._data_config: DataParameters | None = self._resolve_config(values=config, cls=DataParameters, **kwargs)
        return self

    def with_train(self, config: TrainingHyperparams | ParamDict | None = None, **kwargs) -> Self:
        """Configure training settings."""
        self._training_config: TrainingHyperparams | None = self._resolve_config(
            values=config, cls=TrainingHyperparams, **kwargs
        )
        self._enable_synthesis = True
        return self

    def with_generate(self, config: GenerateParameters | ParamDict | None = None, **kwargs) -> Self:
        """Configure generation settings."""
        self._generation_config: GenerateParameters | None = self._resolve_config(
            values=config, cls=GenerateParameters, **kwargs
        )
        self._enable_synthesis = True
        return self

    def with_time_series(self, config: TimeSeriesParameters | ParamDict | None = None, **kwargs) -> Self:
        """Configure time-series settings."""
        self._time_series_config: TimeSeriesParameters | None = self._resolve_config(
            values=config, cls=TimeSeriesParameters, **kwargs
        )
        return self

    def with_differential_privacy(
        self, config: DifferentialPrivacyHyperparams | ParamDict | None = None, **kwargs
    ) -> Self:
        """Configure privacy settings."""
        self._privacy_config: DifferentialPrivacyHyperparams | None = self._resolve_config(
            values=config, cls=DifferentialPrivacyHyperparams, **kwargs
        )
        return self

    def with_replace_pii(self, config: PiiReplacerConfig | ParamDict | None = None, **kwargs) -> Self:
        """Configure PII replacement settings. Will use default PII replacement settings if none are provided, which can be found in
        nemo_safe_synthesizer.config.replace_pii.PiiReplacerConfig.

        If you pass a keyword argument with a config object, overlapping keyword arguments will take precedence.

        Args:
            config: PII replacement configuration or dictionary containing PII replacement parameters.
            **kwargs: Configuration parameters for PII replacement.

        Returns:
            The current Safe Synthesizer builder instance.


        Raises:
            ValueError: If the provided config is not a PiiReplacerConfig, dictionary, or None/unset.

        Examples:
            ```python
            >>> from nemo_microservices import NeMoMicroservices
            >>> from nemo_microservices.beta.safe_synthesizer.sdk.library_builder import SafeSynthesizer
            >>> from nemo_safe_synthesizer.config.replace_pii import PiiReplacerConfig
            >>> # Using default PII replacement settings
            >>> builder = (
            >>>     SafeSynthesizer()
            >>>     .with_data_source(your_dataframe)
            >>>     .with_replace_pii(config=custom_pii_config, **{"classify": {"enable_classify": False}}))
            ```
        """
        cfg = None
        match config:
            case PiiReplacerConfig() as m:
                cfg = m.model_copy(update=kwargs, deep=True)
            case dict() as d:
                cfg = PiiReplacerConfig.model_validate(d).model_copy(update=kwargs, deep=True)
            case None:
                cfg = PiiReplacerConfig.get_default_config().model_copy(update=kwargs, deep=True)
            case _:
                raise ValueError("Config must be a PiiReplacerConfig, dictionary, or None")

        self._replace_pii_config = cfg
        self._enable_replace_pii = True
        return self

    def with_evaluate(self, config: EvaluationParameters | ParamDict | None = None, **kwargs) -> Self:
        """Configure evaluation settings.

        Args:
            config: Evaluation configuration or dictionary containing evaluation parameters.
            **kwargs: Configuration parameters for evaluation.

        Returns:
            The current Safe Synthesizer builder instance.
        """
        self._evaluation_config: EvaluationParameters | None = self._resolve_config(
            values=config, cls=EvaluationParameters, **kwargs
        )
        return self

    def resolve(self) -> Self:
        self._resolve_nss_config()
        self._resolve_datasource()
        return self

    def _resolve_nss_config(self) -> None:
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
        params_to_use["enable_replace_pii"] = (
            self._enable_replace_pii if self._enable_replace_pii is not None else False
        )
        params_to_use["enable_synthesis"] = self._enable_synthesis if self._enable_synthesis is not None else False
        self._nss_config = SafeSynthesizerParameters(**params_to_use)

        # Inject classify_model_provider into PII replacer config if set
        if self._classify_model_provider and self._nss_config.replace_pii:
            self._nss_config.replace_pii.globals.classify.classify_model_provider = self._classify_model_provider
            logger.debug(f"Injected classify model provider into PII config: {self._classify_model_provider}")

    def _resolve_datasource(self, **kwargs) -> None:
        match self._data_source:
            case pd.DataFrame():
                pass
            case str(url):
                self._data_source: pd.DataFrame = pd.read_csv(url, **kwargs)
            case _:
                raise ValueError("Data source must be a pandas DataFrame or a URL")
