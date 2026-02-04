# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""
Named Entity Recognition
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from typing import TYPE_CHECKING, List, Optional, Set, Tuple, Union

from ...data_processing.records.json_record import JSONRecord
from ...data_processing.records.value_path import (
    ValuePath,
    value_path_to_json_path,
)
from ...observability import get_logger
from .utils import InData, input_to_json_records

if TYPE_CHECKING:
    from .pipeline import Pipeline
else:
    Pipeline = None


DEFAULT_CACHE_SIZE = 500000


logger = get_logger(__name__)


class NERError(Exception):
    pass


@dataclass(eq=True, frozen=True, order=True)
class NERPrediction:
    text: str
    start: int
    end: int
    label: str
    source: str
    score: Optional[float]
    field: Optional[str] = None
    value_path: Optional[ValuePath] = None
    # Alternative, so it's not included in __eq__ dataclasses_field(default=None, compare=False)

    substring_match: Optional[bool] = None
    """Set to true if the prediction was made on a substring of the
    original text. If the value is None, this just means we aren't
    tracking this property for the specific prediction.
    """

    @property
    def as_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, source: dict):
        return cls(**source)

    @property
    def json_path(self):
        return value_path_to_json_path(self.value_path)

    def get_dedupe_key_by_label(self, record: JSONRecord) -> Tuple:
        """Returns a string that can be used to dedupe
        predictions from different sources.

        Args:
            record: Used to locate the original value used to generate
                the prediction.

        Returns:
            A a tuple of unique values that can be used to dedupe
            similar predictions by label.
        """
        if self.value_path:
            value = record.value_for_value_path(self.value_path)
            if value is None:
                raise ValueError(f"Could not find value for record using '{value_path_to_json_path(self.value_path)}'")
        else:
            value = str(record.original)

        return (
            self.label,
            self.start,
            self.end,
            self.text,
            self.value_path,
            value,
        )


"""Represents the prediction results of a pipeline"""
PipelineResult = List[List[Union[NERPrediction, dict]]]


PRECISION = 4


@dataclass
class PredictorTimings:
    total_time: float = 0.0
    total_time_avg: float = 0.0

    def to_dict(self):
        return {
            "total_time_ms": round(self.total_time * 1000, PRECISION),
            "total_time_ms_avg": round(self.total_time_avg * 1000, PRECISION),
        }


@dataclass
class Timings:
    records: int = 0
    total_predictions: int = 0
    total_time: float = 0.0
    total_time_avg: float = 0.0
    predictors: dict = dataclasses_field(default_factory=lambda: defaultdict(PredictorTimings))

    def to_dict(self):
        ret = {
            "records": self.records,
            "total_predictions": self.total_predictions,
            "total_time_ms": round(self.total_time * 1000, PRECISION),
            "total_time_ms_avg": round(self.total_time_avg * 1000, PRECISION),
            "time_per_prediction_ms": round((self.total_time / self.total_predictions) * 1000, PRECISION),
        }
        timings_tuples = list(self.predictors.items())
        timings_tuples = sorted(timings_tuples, key=lambda t: t[1].total_time, reverse=True)
        ret["predictors"] = {p: t.to_dict() for p, t in timings_tuples}
        return ret

    def join(self, other: Timings):
        self.records += other.records
        self.total_predictions += other.total_predictions
        self.total_time += other.total_time

        keys_to_join = set(self.predictors.keys()).union(other.predictors.keys())
        for key in keys_to_join:
            _other_timing = other.predictors.get(key)
            _other_total_time = 0 if not _other_timing else _other_timing.total_time

            self.predictors[key].total_time += _other_total_time

    def set_avg(self, num_cpu: int = 1):
        self.total_time_avg = self.total_time / num_cpu
        for _, t in self.predictors.items():
            t.total_time_avg = t.total_time / num_cpu


class NER:
    """Entity Recognition Pipeline"""

    pipeline: Pipeline

    def __init__(self, predictor_cache_size=DEFAULT_CACHE_SIZE, pipeline: Pipeline = None):
        self.predictor_cache_size = predictor_cache_size
        self.pipeline = pipeline

    def predict(
        self,
        in_data: InData,
        *,
        pipeline: Pipeline = None,
        dict_result: bool = False,
        min_score: float = 0.0,
        timings_only: bool = False,
        include_labels: Optional[Set[str]] = None,
    ) -> PipelineResult:
        """
        Predict entities from a string, list or dictionary object.

        By default, every predictor that is available on the python
        installation will be used. To customize what predictors
        should run the `predictors` argument can be used.

        Args:
            - in_data: input data to run predictions against
            - pipeline: an assembled pipeline of predictors
            - dict_results: determines whether a dictionary or NERPrediction
                object is returned.
            - include_labels: a list of labels to filter for. If a prediction
                is not in this list, that prediction will be omitted from
                the result set.

        Returns:
            - a list of NERPrediction tuples.
        """
        if not pipeline and not self.pipeline:
            raise NERError("Pipeline is empty")

        # if we did not get a pipeline as a param, we'll use
        # the instance configured pipeline
        if not pipeline:
            pipeline = self.pipeline

        processed_in_data = input_to_json_records(in_data)  # type: List[JSONRecord]

        # create an empty array for each item
        # that we are predicting
        slots = [[] for _ in processed_in_data]

        # for each predictor we need to add all
        # predictions for each item to the right slot
        result_set = []

        timings = Timings(records=len(processed_in_data))

        input_record: JSONRecord
        for input_record in processed_in_data:
            timings.total_predictions += len(input_record.kv_pairs)
            record_predictions = []
            for predictor in pipeline.iter_predictors():
                start = time.perf_counter()
                predictions = predictor.evaluate(input_record)
                stop = time.perf_counter()
                took = stop - start
                timings.predictors[predictor.source].total_time += took
                timings.total_time += took

                match: NERPrediction
                for match in predictions:
                    if include_labels and match.label not in include_labels:
                        continue
                    # NOTE(jm): score can be None from Spacy
                    if match.score is None or (match.score is not None and match.score >= min_score):
                        record_predictions.append(match)
            result_set.append(record_predictions)

        logger.info(
            f"Analyzed {timings.total_predictions} values in {timings.records} records (took {timings.total_time} secs)"
        )
        if timings_only:
            return timings

        for slot, preds in zip(slots, result_set):
            results = preds
            if dict_result:
                results = [p.as_dict for p in preds]
            slot.extend(results)

        return slots if isinstance(in_data, list) else slots[0]
