# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import time

import pandas as pd

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import SafeSynthesizerResults, SafeSynthesizerSummary, SafeSynthesizerTiming
from nemo_safe_synthesizer.config.parameters import (
    SafeSynthesizerParameters,
)
from nemo_safe_synthesizer.evaluation.reports.multimodal.multimodal_report import MultimodalReport
from nemo_safe_synthesizer.generation.results import GenerateJobResults
from nemo_safe_synthesizer.observability import get_logger
from nemo_safe_synthesizer.pii_replacer.transform_result import ColumnStatistics

logger = get_logger(__name__)


class Evaluator:
    summary: SafeSynthesizerSummary
    report: MultimodalReport
    timing: SafeSynthesizerTiming
    evaluation_time: float
    total_time: float
    results: SafeSynthesizerResults
    workdir: Workdir | None

    def __init__(
        self,
        config: SafeSynthesizerParameters,
        generate_results: GenerateJobResults | pd.DataFrame,
        pii_replacer_time: float | None = None,
        column_statistics: dict[str, ColumnStatistics] | None = None,
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
        workdir: Workdir | None = None,
    ):
        self.config = config
        self.generate_results = generate_results
        self.pii_replacer_time = pii_replacer_time
        self.column_statistics = column_statistics
        self.train_df = train_df
        self.test_df = test_df
        self.workdir = workdir

    def evaluate(self):
        logger.info("Performing Evaluation.")
        evaluation_start = time.monotonic()
        output = self.generate_results if isinstance(self.generate_results, pd.DataFrame) else self.generate_results.df
        report = MultimodalReport.from_dataframes(
            reference=self.train_df,
            output=output,
            test=self.test_df,
            config=self.config,
            column_statistics=self.column_statistics,
        )
        evaluation_time_sec = time.monotonic() - evaluation_start
        logger.info("Evaluation complete.")
        self.evaluation_time = evaluation_time_sec
        self.report = report
