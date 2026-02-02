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

if TYPE_CHECKING:
    from nemo_safe_synthesizer.artifacts.base.analyzer import AnalyzerContext
    from nemo_safe_synthesizer.artifacts.base.fields import FieldFeatures

_MISSING_PERCENTAGE_LIMIT = 50


class MissingDataCheck(DataCheck):
    """
    Generates warnings for fields that have more than 50% values missing.

    It requires FieldFeaturesAnalyzer to run before, as it uses field_features from
    the context.
    """

    @property
    def check_id(self) -> str:
        return "missing_data"

    def run_check(self, context: AnalyzerContext) -> DataCheckResult:
        if not context.field_features:
            return DataCheckResult.skipped(self.check_id, "Missing dataset metadata.")

        warning_fields: list[FieldFeatures] = []
        for field in context.field_features:
            if _missing_field_cond(field):
                warning_fields.append(field)

        if warning_fields:
            field_names = [f.name for f in warning_fields]
            return DataCheckResult.completed(self.check_id, [DataCheckWarning(_explain(field_names), field_names)])

        return DataCheckResult.completed(self.check_id, [])


def _explain(field_names: list[str]) -> str:
    count = len(field_names)
    template = "{prefix} {have} more than {limit}% missing data. "
    return template.format(
        prefix=warning_explain_prefix(field_names, "missing data"),
        have=plural_verb(("has", "have"), count),
        limit=_MISSING_PERCENTAGE_LIMIT,
    )


def _missing_field_cond(field: FieldFeatures) -> bool:
    return field.missing_percent >= _MISSING_PERCENTAGE_LIMIT
