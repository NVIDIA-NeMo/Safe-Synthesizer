# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_safe_synthesizer.artifacts.analyzers.data_checks.base import (
    DataCheck,
    plural_verb,
    warning_explain_prefix,
)
from nemo_safe_synthesizer.artifacts.base.data_checks import (
    DataCheckResult,
    DataCheckWarning,
)
from nemo_safe_synthesizer.artifacts.base.fields import FieldType

if TYPE_CHECKING:
    from nemo_safe_synthesizer.artifacts.base.analyzer import AnalyzerContext
    from nemo_safe_synthesizer.artifacts.base.fields import FieldFeatures

_FLOAT_PRECISION_LIMIT = 6


class HighFloatPrecisionCheck(DataCheck):
    """
    Generates warnings for fields that contain high precision float numbers.
    What counts here is number of digits after the dot (e.g. 1.1234 has precision 4).

    It requires FieldFeaturesAnalyzer to run before, as it uses field_features from
    the context.
    """

    @property
    def check_id(self) -> str:
        return "high_float_precision"

    def run_check(self, context: AnalyzerContext) -> DataCheckResult:
        if not context.field_features:
            return DataCheckResult.skipped(self.check_id, "Missing dataset metadata.")

        warning_fields: list[FieldFeatures] = []
        for field in context.field_features:
            if _high_float_precision_cond(field):
                warning_fields.append(field)

        if warning_fields:
            field_names = [f.name for f in warning_fields]
            return DataCheckResult.completed(self.check_id, [DataCheckWarning(_explain(field_names), field_names)])

        return DataCheckResult.completed(self.check_id, [])


def _explain(field_names: list[str]) -> str:
    count = len(field_names)
    template = "{prefix} {have} floats with precision greater than {limit} digits. Try reducing precision to 4 digits."
    return template.format(
        prefix=warning_explain_prefix(field_names, "high floating point precision"),
        have=plural_verb(("has", "have"), count),
        limit=_FLOAT_PRECISION_LIMIT,
    )


def _high_float_precision_cond(field: FieldFeatures) -> bool:
    if field.type != FieldType.NUMERIC:
        return False

    return field.max_precision and field.max_precision > _FLOAT_PRECISION_LIMIT
