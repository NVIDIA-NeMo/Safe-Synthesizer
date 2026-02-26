# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from math import copysign, floor, log10
from typing import TYPE_CHECKING, Any

from pandas import Series
from pandas.core.dtypes.common import is_float_dtype, is_numeric_dtype

from ..base.analyzer import (
    AnalyzerContext,
    ArtifactAnalyzer,
)
from ..base.fields import (
    FieldFeatures,
    FieldFeaturesInfo,
    FieldType,
)
from ..base.metrics import timed

if TYPE_CHECKING:
    from ..base.metadata import (
        DatasetMetadata,
    )


_TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD = 2


class FieldFeaturesAnalyzer(ArtifactAnalyzer):
    @timed("FieldFeaturesDetectionTime")
    def analyze(self, context: AnalyzerContext) -> None:
        df = context.data_frame

        fields = [describe_field(field_name, df[field_name]) for field_name in df.columns]

        # add classification info if available
        if context.ner_metadata:
            _add_ner_to_fields(context.ner_metadata, fields)

        context.field_features = fields
        context.field_info = FieldFeaturesInfo(fields)

        _add_info_to_manifest(context, fields)


def _add_info_to_manifest(context: AnalyzerContext, fields: list[FieldFeatures]) -> None:
    manifest = context.manifest

    manifest.add_feature("types", _type_summary(fields))

    for key, value in _calculate_aggregates(fields, context).items():
        manifest.add_feature(key, value)

    # serialize to dict and anonymize field names
    fields_dict = [field.to_dict() for field in fields]
    for field_dict in fields_dict:
        field_dict["name"] = context.field_name_anonymizer.anonymize(field_dict["name"])

    manifest.fields = fields_dict

    # Attach the actual list of field features so we can interact with
    # them in other libraries if need be
    manifest.field_features = fields


def _add_ner_to_fields(ner_metadata: DatasetMetadata, fields: list[FieldFeatures]) -> None:
    fields_meta = ner_metadata.data.fields

    for field in fields:
        field_meta = next(filter(lambda meta: meta.field == field.name, fields_meta))
        if field_meta and field_meta.entities:
            meta_dict = field_meta.dict()
            # only take subset of keys
            field.classification = {key: meta_dict[key] for key in ["entities", "field_labels"]}


def _type_summary(fields: list[FieldFeatures]) -> list[dict]:
    field_type_counter = Counter(field.type.value for field in fields)

    return [{"type": field_type.value, "count": field_type_counter.get(field_type, 0)} for field_type in FieldType]


def _calculate_aggregates(fields: list[FieldFeatures], context: AnalyzerContext) -> dict[str, Any]:
    aggregates = {}
    if not fields:
        return aggregates

    max_unique = _max_aggregate(
        fields,
        cmp_key=lambda field: field.unique_count,
        item_fn=lambda field: {
            "unique_count": field.unique_count,
            "unique_percent": field.unique_percent,
            "field": context.field_name_anonymizer.anonymize(field.name),
        },
    )
    if max_unique:
        aggregates["max_unique"] = max_unique

    total_missing_count = sum(field.missing_count or 0 for field in fields)
    aggregates["total_missing_count"] = total_missing_count

    if total_missing_count > 0:
        # Only emit "max_missing" if there are any missing values, otherwise
        # "max_missing" would contain all the fields with "missing_percent" = 0.0
        max_missing = _max_aggregate(
            fields,
            cmp_key=lambda field: field.missing_count,
            item_fn=lambda field: {
                "missing_count": field.missing_count,
                "missing_percent": field.missing_percent,
                "field": context.field_name_anonymizer.anonymize(field.name),
            },
        )
        if max_missing:
            aggregates["max_missing"] = max_missing

    max_avg_str_length = _max_aggregate(
        fields,
        cmp_key=lambda field: field.avg_str_length,
        item_fn=lambda field: {
            "avg_str_length": field.avg_str_length,
            "field": context.field_name_anonymizer.anonymize(field.name),
        },
    )
    if max_avg_str_length:
        aggregates["max_avg_str_length"] = max_avg_str_length

    aggregates["highly_unique_field_count"] = sum(
        field.unique_percent > 80 for field in fields if field.unique_percent is not None
    )
    aggregates["high_missing_values_field_count"] = sum(
        field.missing_percent > 20 for field in fields if field.missing_percent is not None
    )

    return aggregates


def _max_aggregate(fields: list[FieldFeatures], cmp_key: callable, item_fn: callable) -> list[dict[str, Any]] | None:
    """
    Calculates MAX() aggregate.

    Args:
        fields: list of fields for calculation
        cmp_key: function that extracts key based on which fields are compared
        item_fn: function that creates an item with information about the field that has max value
    """
    max_field = max(fields, key=lambda field: cmp_key(field) or 0)
    if cmp_key(max_field) is not None:
        max_value = cmp_key(max_field)

        # return array of all of the fields that have max_value
        return [item_fn(field) for field in fields if cmp_key(field) == max_value]

    return None


def float_precision(data: Series) -> tuple[int | None, int | None]:
    """
    Converts a number to a string representation and counts number
    of decimal digits and returns (min, max) of these counts.

    Returns:
        Tuple with min and max number of decimal digits in values of the series or
        `None` if there are no floating point numbers in the series.
    """
    if not is_float_dtype(data.dtype):
        return None, None

    def _value_precision(value: float) -> int | None:
        parts = str(value).split(".")
        if len(parts) > 1 and parts[1] != "0":
            return len(parts[1])
        return 0

    precision_values = list(_filter_nan(_value_precision(item) for item in data))
    if len(precision_values) == 0:
        return None, None

    precision = min(precision_values), max(precision_values)
    if precision == (0, 0):
        return None, None

    return precision


def _filter_nan(iterable: Iterable[Any]) -> Iterable[Any]:
    return [x for x in iterable if x is not None]


def describe_field(field_name: str, data: Series) -> FieldFeatures:
    total_count = len(data)

    non_na_data = data.dropna()
    non_na_count = int(non_na_data.count())
    unique_values_list = non_na_data.unique().tolist()
    unique_count = len(unique_values_list)
    missing_count = total_count - non_na_count

    lengths = [len(str(entry)) for entry in non_na_data]
    features = FieldFeatures(
        name=field_name,
        type=FieldType.OTHER,
        count=non_na_count,
        unique_values_list=unique_values_list,
        unique_count=unique_count,
        unique_percent=(round(unique_count / non_na_count * 100, 4) if non_na_count > 0 else 0),
        missing_count=missing_count,
        missing_percent=(round(missing_count / total_count * 100, 4) if total_count > 0 else 0),
        min_str_length=min(lengths) if non_na_count > 0 else 0,
        max_str_length=max(lengths) if non_na_count > 0 else 0,
        avg_str_length=round(sum(lengths) / non_na_count, 4) if non_na_count > 0 else 0,
    )

    if non_na_count == 0:
        features.type = FieldType.EMPTY
        return features

    if unique_count == 2:
        features.type = FieldType.BINARY
        return features

    data_type = non_na_data.dtype
    if is_float_dtype(data_type):
        features.type = FieldType.NUMERIC
        features.min_value = floor_power_of_10(float(non_na_data.min()))
        features.max_value = floor_power_of_10(float(non_na_data.max()))
        features.min_precision, features.max_precision = float_precision(non_na_data)
        return features

    if is_numeric_dtype(data_type):
        min_value = floor_power_of_10(int(non_na_data.min()))
        if unique_count <= 10 and min_value >= 0:
            features.type = FieldType.CATEGORICAL
            return features

        features.type = FieldType.NUMERIC
        features.min_value = min_value
        features.max_value = floor_power_of_10(int(non_na_data.max()))
        return features

    # See
    # - https://github.com/Gretellabs/text_research/blob/main/column_detector.py
    # - https://jeffreymorgan.io/articles/identifying-categorical-data/
    diff = non_na_count - unique_count
    diff_percent = diff / non_na_count

    # ENGPROD-5: ``entry`` needs to be a string here regardless
    # in order to get the space count
    space_count = sum(str(entry).count(" ") for entry in non_na_data)
    features.space_count = space_count

    if diff_percent >= 0.9 or (diff_percent >= 0.7 and len(non_na_data) <= 50):
        features.type = FieldType.CATEGORICAL
        return features

    if space_count / non_na_count > _TEXT_FIELD_AVG_SPACE_COUNT_THRESHOLD:
        features.type = FieldType.TEXT
        return features

    return features


def floor_power_of_10(value: int | float) -> int | float:
    """Returns the biggest power of 10 that is smaller than the provided value."""
    if value == 0:
        return 1

    if math.isinf(value):
        return value

    if math.isnan(value):
        return value

    floor_log = floor(log10(abs(value)))
    if value < 0:
        floor_log += 1

    return copysign(10**floor_log, value)
