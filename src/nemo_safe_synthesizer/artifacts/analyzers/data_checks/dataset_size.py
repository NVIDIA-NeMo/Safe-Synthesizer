# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from ...base.data_checks import (
    DataCheckResult,
    DataCheckWarning,
)
from .base import DataCheck

if TYPE_CHECKING:
    from ...base.analyzer import AnalyzerContext

_RECORD_COUNT_LIMIT = 50_000
_FIELD_COUNT_LIMIT = 30


class DatasetSizeCheck(DataCheck):
    """
    Generates warnings for datasets that are likely too large to create a good model.
    """

    @property
    def check_id(self) -> str:
        return "dataset_size"

    def run_check(self, context: AnalyzerContext) -> DataCheckResult:
        record_count, field_count = context.data_frame.shape

        warnings = []
        if record_count >= _RECORD_COUNT_LIMIT and field_count >= _FIELD_COUNT_LIMIT:
            warnings.append(DataCheckWarning(_explain()))

        return DataCheckResult.completed(self.check_id, warnings)


def _explain() -> str:
    return "Complex dataset: Your dataset has more than 50K rows and 30 columns."
