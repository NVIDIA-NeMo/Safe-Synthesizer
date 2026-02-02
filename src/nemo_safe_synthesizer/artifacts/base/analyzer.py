# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict, Field

from nemo_safe_synthesizer.artifacts.base.name_anonymizer import NoopNameAnonymizer

if TYPE_CHECKING:
    from nemo_safe_synthesizer.artifacts.base.data_checks import (
        DataCheckResult,
    )
    from nemo_safe_synthesizer.artifacts.base.fields import (
        FieldFeatures,
        FieldFeaturesInfo,
    )
    from nemo_safe_synthesizer.artifacts.base.metadata import (
        DatasetMetadata,
    )
    from nemo_safe_synthesizer.artifacts.base.name_anonymizer import NameAnonymizer


class AnalyzerContext(BaseModel):
    data_frame: DataFrame

    model_config = ConfigDict(arbitrary_types_allowed=True)

    field_name_anonymizer: NameAnonymizer = Field(default=NoopNameAnonymizer())
    """
    To be used by analyzers where field name needs to be anonymized.
    """

    ner_metadata: DatasetMetadata | None = Field(default=None)
    field_features: list[FieldFeatures] = Field(default=list())
    field_info: FieldFeaturesInfo | None = Field(default=None)
    """
    Can be used by other libraries to get general field information without having to parse the manifest data
    """
    data_check_results: list[DataCheckResult] = Field(default=list())


class ArtifactAnalyzer(ABC):
    @abstractmethod
    def analyze(self, context: AnalyzerContext) -> None: ...


class DataFrameBasicAnalyzer(ArtifactAnalyzer):
    def analyze(self, context: AnalyzerContext) -> None:
        # just did manifest stuff. stripped it out.
        return
