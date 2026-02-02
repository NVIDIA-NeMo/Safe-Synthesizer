# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from pandas import Series
from pandas.core.dtypes.common import is_string_or_object_np_dtype

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

_VALUES_WITH_WHITESPACE_PERCENTAGE_LIMIT = 20


class SurroundingWhitespacesCheck(DataCheck):
    """
    Generates warnings for fields that contain leading or trailing whitespaces in more
    than 20% of its values.
    """

    @property
    def check_id(self) -> str:
        return "surrounding_whitespaces"

    def run_check(self, context: AnalyzerContext) -> DataCheckResult:
        field_names = []
        df = context.data_frame
        for field_name in df.columns:
            if _has_surrounding_whitespaces(df[field_name]):
                field_names.append(field_name)

        if field_names:
            return DataCheckResult.completed(self.check_id, [DataCheckWarning(_explain(field_names), field_names)])

        return DataCheckResult.completed(self.check_id, [])


def _explain(field_names: list[str]) -> str:
    count = len(field_names)
    template = (
        "{prefix} {contains} whitespaces at the beginning or the end "
        "for at least {limit}% of its values. "
        "Leading and trailing whitespaces reduce model performance."
    )
    return template.format(
        prefix=warning_explain_prefix(field_names, "surrounding whitespaces"),
        contains=plural_verb(("contains", "contain"), count),
        limit=_VALUES_WITH_WHITESPACE_PERCENTAGE_LIMIT,
    )


def _has_surrounding_whitespaces(data: Series) -> bool:
    if not is_string_or_object_np_dtype(data.dtype):
        return False

    whitespaces = (" ", "\t", "\n")
    result = data.apply(
        lambda value: value is not None and (str(value).startswith(whitespaces) or str(value).endswith(whitespaces))
    )

    return sum(result) / len(result) * 100 >= _VALUES_WITH_WHITESPACE_PERCENTAGE_LIMIT
