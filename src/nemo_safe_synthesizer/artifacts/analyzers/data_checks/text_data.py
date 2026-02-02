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

_TEXT_FIELDS_COUNT_LIMIT = 2


class TextDataCheck(DataCheck):
    """
    Generates a warning if dataset contains more than 3 text fields.

    It requires FieldFeaturesAnalyzer to run before, as it uses field_features from
    the context.
    """

    @property
    def check_id(self) -> str:
        return "text_data"

    def run_check(self, context: AnalyzerContext) -> DataCheckResult:
        if not context.field_features:
            return DataCheckResult.skipped(self.check_id, "Missing dataset metadata.")

        text_fields = [f for f in context.field_features if f.type == FieldType.TEXT]
        if len(text_fields) >= _TEXT_FIELDS_COUNT_LIMIT:
            field_names = [f.name for f in text_fields]
            return DataCheckResult.completed(self.check_id, [DataCheckWarning(_explain(field_names), field_names)])

        return DataCheckResult.completed(self.check_id, [])


def _explain(field_names: list[str]) -> str:
    count = len(field_names)
    template = (
        "{prefix} {contains} unstructured text. Try using the Gretel GPT model on those for higher quality results."
    )
    return template.format(
        prefix=warning_explain_prefix(field_names, "text data"),
        contains=plural_verb(("contains", "contain"), count),
    )
