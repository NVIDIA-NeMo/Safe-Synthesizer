# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import multiprocessing
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional

from ...data_processing.records.base import normalize_labels
from ...observability import get_logger
from .metadata import FieldLabelCondition
from .ner import NER
from .ner_mp import NERParallel
from .nlp import SpacyPredictor
from .pipeline import Pipeline, regex_pipeline
from .predictor import Predictor
from .regex import RegexPredictor

logger = get_logger(__name__)


class PredictorFilter(ABC):
    """Used to filter predictors that should be included in the NER."""

    @abstractmethod
    def should_include(self, predictor: Predictor): ...


class NERPipelineType(StrEnum):
    DEFAULT = "default"
    REGEX = "regex"
    ML = "ml"

    @classmethod
    def from_flags(cls, *, use_nlp: bool, regex_only: bool):
        if use_nlp:
            return NERPipelineType.ML

        if regex_only:
            return NERPipelineType.REGEX

        return NERPipelineType.DEFAULT

    def create_pipeline(self) -> Pipeline:
        return regex_pipeline()


class LabelSetPredictorFilter(PredictorFilter):
    def __init__(self, included_labels: set[str]):
        self._included_labels = normalize_labels(included_labels)

    def should_include(self, predictor: Predictor) -> bool:
        if isinstance(predictor, RegexPredictor):
            # For Regex predictor, we filter based on ``entity.tag`` (that's what is visible to the user)
            label = predictor.entity.tag if predictor.entity else predictor.source
            return label in self._included_labels

        # the spacy predictor must be explicitly configured,
        # so, if it appears in the list we can assume it should
        # be included.
        if isinstance(predictor, SpacyPredictor):
            return True

        return predictor.source in self._included_labels or predictor.default_name in self._included_labels


def _create_base_pipeline(predictor_filter: Optional[PredictorFilter], pipeline_type: NERPipelineType) -> Pipeline:
    # TODO(pm): For now this is good enough - we first create a pipeline with all predictors and
    #  then apply the filter. Better option -> create only the ones that are needed.
    pipeline = pipeline_type.create_pipeline()

    if not predictor_filter:
        logger.info("No predictor filters are defined, returning default pipeline.")
        return pipeline

    pipeline.predictors = [
        predictor for predictor in pipeline.iter_predictors() if predictor_filter.should_include(predictor)
    ]
    return pipeline


class NERFactoryBase(ABC):
    @abstractmethod
    def create(
        self,
        *,
        predictor_filter: Optional[PredictorFilter] = None,
        record_count: Optional[int] = None,
    ) -> NER | NERParallel: ...


class NERFactory(NERFactoryBase):
    def __init__(
        self,
        custom_predictors: Optional[list[Predictor]] = None,
        *,
        parallel: bool = True,
        use_nlp: bool = False,
        regex_only: bool = False,
        ner_max_runtime_seconds: int = None,
    ):
        if custom_predictors is None:
            custom_predictors = []

        self._custom_predictors = custom_predictors
        self._parallel = parallel
        self._ner_pipeline_type = NERPipelineType.from_flags(use_nlp=use_nlp, regex_only=regex_only)
        self._ner_max_runtime_seconds = ner_max_runtime_seconds

    def create(
        self,
        *,
        predictor_filter: Optional[PredictorFilter] = None,
        record_count: Optional[int] = None,
    ) -> NER | NERParallel:
        if self._parallel:
            return self._create_parallel_ner(predictor_filter, record_count)

        return self._create_ner(predictor_filter)

    def _create_parallel_ner(
        self,
        predictor_filter: Optional[PredictorFilter] = None,
        record_count: Optional[int] = None,
    ) -> NER | NERParallel:
        """
        Determines an optimal number of NER workers and creates NERParallel.

        Args:
            predictor_filter: Filter that will be used when creating predictors, NER will only use subset
                of predictors that pass the filter.
            record_count: Estimated number or records to label, which is used to optimize size of
                the worker pool.
        """
        # leave 1 CPU free for other work
        num_proc = max(1, multiprocessing.cpu_count() - 1)
        if record_count:
            # make sure there is at least 1000 records per worker to make it worth it
            num_proc = min(num_proc, max(1, record_count // 1000))

        # NOTE(jm): I was getting workers that were dying due to memory consumption when
        # testing on certain systems (or containers). If there are too many CPUs on the system and the
        # system tries to run another process and not enough memory workers will
        # start getting killed. So I'm setting an env here that allows an override
        # of the num CPUs when we need to explicitly control it.
        num_proc_env = os.getenv("SAFE_SYNTHESIZER_CPU_COUNT")
        if num_proc_env:
            try:
                num_proc = int(num_proc_env)
            except ValueError:
                ...

        if num_proc == 1:
            logger.info("Number of processes for NER is 1 - creating a standard NER.")
            return self._create_ner(predictor_filter)

        logger.info(f"Creating NERParallel with {num_proc} workers.")

        return NERParallel(
            pipeline_factory=lambda: self._create_predictor_pipeline(predictor_filter),
            num_proc=num_proc,
            ner_max_runtime_seconds=self._ner_max_runtime_seconds,
        )

    def _create_ner(self, predictor_filter: Optional[PredictorFilter] = None) -> NER:
        return NER(pipeline=self._create_predictor_pipeline(predictor_filter))

    def _create_predictor_pipeline(self, predictor_filter: Optional[PredictorFilter] = None) -> Pipeline:
        logger.info("Creating NER pipeline.")

        pipeline = _create_base_pipeline(predictor_filter, self._ner_pipeline_type)
        for custom_predictor in self._custom_predictors:
            if not predictor_filter or predictor_filter.should_include(custom_predictor):
                pipeline.add_predictors(custom_predictor)

        logger.info(f"Pipeline created with {len(pipeline.predictors)} predictors.")

        return pipeline


class StaticNERFactory(NERFactoryBase):
    def __init__(self, ner: NER | NERParallel):
        self._ner = ner

    def create(
        self,
        *,
        predictor_filter: Optional[PredictorFilter] = None,
        record_count: Optional[int] = None,
    ) -> NER | NERParallel:
        return self._ner


@dataclass(frozen=True)
class NERBundle:
    """Bundle that contains all of the NER-related information that the code running NER may need."""

    factory: NERFactoryBase = field(default_factory=NERFactory)

    custom_labels: list[str] = field(default_factory=list)

    field_label_condition: FieldLabelCondition = field(default_factory=FieldLabelCondition)
