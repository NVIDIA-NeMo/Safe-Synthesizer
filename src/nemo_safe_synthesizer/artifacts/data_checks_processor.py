# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

from pandas import DataFrame

from ..observability import get_logger
from .analyzers.data_checks import (
    create_all_checks,
)
from .analyzers.data_checks.base import DataChecksAnalyzer
from .analyzers.field_features import FieldFeaturesAnalyzer
from .base.analyzer import AnalyzerContext, ArtifactAnalyzer, DataFrameBasicAnalyzer
from .base.data_checks import DataCheckResults
from .base.name_anonymizer import NoopNameAnonymizer

logger = get_logger(__name__)


class DataChecksProcessor:
    """
    Manages process of running data checks on a given DataFrame:
    - prepares the context,
    - runs required analyzers, so the context is populated
    - runs data checks.
    """

    def __init__(self):
        self._analyzers = _create_analyzers()

    def process(self, data_source: str, df: DataFrame) -> DataCheckResults:
        start_ns = time.monotonic_ns()
        context = AnalyzerContext(data_frame=df, field_name_anonymizer=NoopNameAnonymizer())

        for analyzer in self._analyzers:
            analyzer.analyze(context)

        return DataCheckResults(
            elapsed_time_ms=(time.monotonic_ns() - start_ns) // 1_000_000,
            check_results=context.data_check_results,
        )


def _create_analyzers() -> list[ArtifactAnalyzer]:
    data_checks = create_all_checks()
    return [
        DataFrameBasicAnalyzer(),
        FieldFeaturesAnalyzer(),
        # This one has to go last.
        DataChecksAnalyzer(data_checks),
    ]
