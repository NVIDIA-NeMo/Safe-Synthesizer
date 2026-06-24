# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures as futures
import multiprocessing as mp
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from itertools import islice
from threading import BoundedSemaphore
from typing import Any, Optional

import joblib.externals.loky as loky
from typing_extensions import TypeIs

from ...observability import get_logger
from . import pipeline
from .ner import NER, NERPrediction, PipelineResult, Prediction, PredictionList, Timings
from .utils import InData, input_to_json_records

logger = get_logger(__name__)

CHUNK_SIZE = 500
MAX_CHUNKS = 250


@dataclass
class _ProcPayload:
    seq: int
    in_data: InData
    out_data: Timings | PipelineResult | None = None


_ner_predictor: NER | None = None
"""This global var should only ever be init'd to an
NER instance by a process that is part of the process
pool. By making it a global it becomes more stable to
init at the module level at the time the process pool
is created.
"""


def _is_prediction(value: object) -> TypeIs[Prediction]:
    return isinstance(value, NERPrediction) or (isinstance(value, dict) and all(isinstance(key, str) for key in value))


def _is_prediction_list(value: object) -> TypeIs[PredictionList]:
    return isinstance(value, list) and all(_is_prediction(item) for item in value)


def _extend_single_predictions(target: PredictionList, out_data: object) -> None:
    """Append worker output for a single input record to ``target``."""
    if _is_prediction_list(out_data):
        target.extend(out_data)
        return
    if isinstance(out_data, list):
        for row in out_data:
            if _is_prediction_list(row):
                target.extend(row)


def _set_ner_predictor(pipeline_factory: Callable[[], pipeline.Pipeline]):
    """This is the init routine when forking the process
    pool and should be passed into the pool constructor
    exclusively.
    """
    try:
        logger.info("Initializing NER in the worker...")
        global _ner_predictor
        _ner_predictor = NER(pipeline=pipeline_factory())
        logger.info("NER initialized.")

    except Exception as e:
        logger.error("Error during NER initialization!", exc_info=e)
        raise e


def _predict(payload: _ProcPayload, **kwargs) -> _ProcPayload:
    """This is the entrypoint for workers in the process
    pool and should be used there exclusively. The global
    NER predictor object should be created by the time
    this is ever called
    """
    try:
        if _ner_predictor is None:
            raise RuntimeError("NER worker was not initialized")
        payload.out_data = _ner_predictor.predict(payload.in_data, **kwargs)
        return payload
    except Exception as e:
        logger.error("Something went wrong with NER.predict.", exc_info=e)
        raise e


def iter_record_chunks(it: Iterator[Any], batch_size: int) -> Iterator[list[Any]]:
    while True:
        chunk = islice(it, batch_size)
        chunk_list = list(chunk)
        if not chunk_list:
            break
        yield chunk_list


@dataclass
class _ResultData:
    results: list[_ProcPayload] = field(default_factory=list)
    lock: BoundedSemaphore = field(default_factory=lambda: BoundedSemaphore(value=MAX_CHUNKS))
    # progress_callback: logging.ProgressCallback = field(
    #     default_factory=lambda: logging.get_progress_callback("Classifying data... ")
    # )
    """Only allow at max this number of items into the queue that is managed
    by the worker pool"""

    def handle_results(self, future: futures.Future):
        result = future.result()  # type: _ProcPayload
        logger.info(f"Chunk number {result.seq + 1} has completed")
        self.lock.release()
        self.results.append(result)
        # self.progress_callback.update_inc(len(result.out_data))


@dataclass
class _ChunkInfo:
    seq: int
    chunk_size: int


class NERParallel:
    num_proc: int
    pool: loky.ProcessPoolExecutor
    pipeline_factory: Callable[[], pipeline.Pipeline]

    def _initialize_pool(self):
        self.pool = loky.ProcessPoolExecutor(
            max_workers=self.num_proc,
            initializer=_set_ner_predictor,
            initargs=(self.pipeline_factory,),
        )

    def __init__(
        self,
        pipeline_factory: Callable[[], pipeline.Pipeline],
        *,
        num_proc: Optional[int] = None,
        ner_max_runtime_seconds: Optional[int] = None,
    ):
        if num_proc is None:
            num_proc = mp.cpu_count()
        self.num_proc = num_proc
        self.pipeline_factory = pipeline_factory
        self._initialize_pool()
        self.ner_max_runtime_seconds = ner_max_runtime_seconds

    def predict(self, in_data: InData, **kwargs) -> Timings | PipelineResult:
        """
        Runs NER prediction on the input data.

        This method will return one of 3 things:
        - If `timings_only=True` kwarg is passed in -> a single `Timings` object.
          E.g. `predict(data, timings_only=True)`
        - if `in_data` is a str, dict or JSONRecord -> a list with a single result.
        - if `in_data` is a list -> a list of results, of the same length as `in_data`.
          In a case where item from `in_data` had no entities in it, the result will be an empty list.
          E.g. `result = [[], [NERPrediction(), NERPrediction()], [], []]`.
          In this example, there were 4 records in `in_data`, 3 of them had no entities,
          and the second one had 2.
        """
        if "pipeline" in kwargs:
            raise ValueError("A pipeline object cannot be passed to predict() in MP mode!")

        logger.info(f"Starting NER prediction, using {self.num_proc} workers.")

        # _futures = []

        timings_only = kwargs.get("timings_only", False)

        list_input = isinstance(in_data, list)
        if list_input:
            # Send pure dicts to NER, as it's much faster to pickle/unpickle
            # pure dicts that JSONRecords (and that's what multiprocessing is doing)
            records = [record.original for record in input_to_json_records(in_data)]

            chunks: list[tuple[InData, int]] = [
                (chunk, len(chunk)) for chunk in iter_record_chunks(iter(records), CHUNK_SIZE)
            ]
        else:
            # We need to handle the case where a non-list is sent in
            # as our prediction object and we need to makesure the payload
            # we send to the worker is the raw input, not a list
            chunks = [(in_data, 1)]

        result_data_tracker = _ResultData()

        total_chunks = 0
        submitted_chunks = []
        for i, (data, chunk_size) in enumerate(chunks):
            result_data_tracker.lock.acquire()
            logger.info(f"Submitting chunk number {i + 1} to NER workers.")

            payload = _ProcPayload(seq=i, in_data=data)
            submitted_chunks.append(_ChunkInfo(seq=i, chunk_size=chunk_size))

            self.pool.submit(_predict, payload, **kwargs).add_done_callback(result_data_tracker.handle_results)

            logger.info(f"Chunk number {i + 1} has been submitted successfully.")
            total_chunks += 1
            # _futures.append(future)

        logger.info("All chunks submitted, waiting for work to complete...")

        # When we get here, all chunks have been submitted, and we just
        # want to wait for them to finish up before moving on
        start = time.time()
        while len(result_data_tracker.results) < total_chunks:
            time_passed_seconds = time.time() - start
            if self.ner_max_runtime_seconds is not None and self.ner_max_runtime_seconds < time_passed_seconds:
                logger.error(
                    "NER took more than %d seconds to finish, ending early",
                    self.ner_max_runtime_seconds,
                )
                self.pool.shutdown(wait=False, kill_workers=True)
                self._initialize_pool()  # in case it's reused
                break

        result_data = result_data_tracker.results

        logger.info("NER prediction completed.")

        if timings_only:
            result_timings = [result.out_data for result in result_data if isinstance(result.out_data, Timings)]
            timings = result_timings[0] if result_timings else Timings()
            for other_timings in result_timings[1:]:
                timings.join(other_timings)
            timings.set_avg(num_cpu=self.num_proc)
            return timings

        completed_chunks: dict[int, _ProcPayload] = {r.seq: r for r in result_data}
        all_chunks: dict[int, _ChunkInfo] = {ci.seq: ci for ci in submitted_chunks}

        # Restore the predictions to the order they would
        # have been if predicting on a single worker
        if list_input:
            batch_preds: list[PredictionList] = []
        else:
            single_preds: PredictionList = []
        for seq in sorted(list(all_chunks.keys())):
            if (payload := completed_chunks.get(seq, None)) is not None:
                if isinstance(payload.out_data, list):
                    if list_input:
                        for row in payload.out_data:
                            if _is_prediction_list(row):
                                batch_preds.append(row)
                    else:
                        _extend_single_predictions(single_preds, payload.out_data)

            else:
                # Add an empty spot for each record in the chunk that wasn't completed
                logger.warning(f"NER for chunk number {seq + 1} did not complete.")
                if list_input:
                    batch_preds.extend([[]] * all_chunks[seq].chunk_size)

        return batch_preds if list_input else single_preds

    def __exit__(self, exc_type, exc_value, traceback):
        logger.info("Shutting down NER worker pool.")
        self.pool.shutdown()
