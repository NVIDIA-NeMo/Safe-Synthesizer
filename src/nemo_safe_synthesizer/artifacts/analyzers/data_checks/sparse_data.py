# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...base.data_checks import (
    DataCheckResult,
    DataCheckWarning,
)
from ...base.fields import (
    FieldType,
)
from .base import (
    DataCheck,
    plural_verb,
    warning_explain_prefix,
)

if TYPE_CHECKING:
    from ...base.analyzer import (
        AnalyzerContext,
    )
    from ...base.fields import (
        FieldFeatures,
    )

_BINARY_FIELDS_COUNT_LIMIT = 3


class SparseDataCheck(DataCheck):
    """
    Generates a warning if dataset contains more than 4 binary fields.

    It requires FieldFeaturesAnalyzer to run before, as it uses field_features from
    the context.
    """

    @property
    def check_id(self) -> str:
        return "sparse_data"

    def run_check(self, context: AnalyzerContext) -> DataCheckResult:
        if not context.field_features:
            return DataCheckResult.skipped(self.check_id, "Missing dataset metadata.")

        # NSS NB - there  was a warning array here that was a noop
        fields = context.field_features
        df = context.data_frame

        zero_or_one_fields: list[FieldFeatures] = []
        for field in fields:
            if field.type != FieldType.BINARY:
                continue
            if df[field.name].isin([np.nan, 0, 1]).all():
                zero_or_one_fields.append(field)

        if len(zero_or_one_fields) >= _BINARY_FIELDS_COUNT_LIMIT:
            field_names = [f.name for f in zero_or_one_fields]
            return DataCheckResult.completed(self.check_id, [DataCheckWarning(_explain(field_names), field_names)])

        return DataCheckResult.completed(self.check_id, [])


def _explain(field_names: list[str]) -> str:
    count = len(field_names)
    template = "{prefix} {contains} binary values. Try reverting back to a dense representation. "
    return template.format(
        prefix=warning_explain_prefix(field_names, "sparse data"),
        contains=plural_verb(("contains", "contain"), count),
    )
