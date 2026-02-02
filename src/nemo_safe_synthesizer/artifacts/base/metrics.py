# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from functools import wraps


class Metric(ABC):
    _name: str
    _dimensions: dict[str, str] | None

    def __init__(self, name: str, dimensions: dict[str, str] | None = None):
        self._name = name
        self._dimensions = dimensions

    @property
    def name(self):
        return self._name

    @property
    @abstractmethod
    def value(self): ...


class Timer(Metric):
    _start: float
    _duration_ms: float

    def __enter__(self):
        self._start = time.perf_counter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._duration_ms = round((time.perf_counter() - self._start) * 1000, 2)

    @property
    def value(self):
        return self._duration_ms


class Metrics:
    _metrics: list[Metric]

    def __init__(self):
        self._metrics = []

    def timer(self, metric_name: str, dimensions: dict[str, str] | None = None):
        metric = Timer(metric_name, dimensions)
        self._metrics.append(metric)
        return metric

    def print(self):
        return json.dumps([{"name": metric.name, "value": metric.value} for metric in self._metrics])


class Unit(StrEnum):
    MILLISECOND = "ms"


_GLOBAL_REGISTRY: Metrics = None


def init_global_registry() -> Metrics:
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = Metrics()

    return _GLOBAL_REGISTRY


def timer(metric_name: str, dimensions: dict[str, str] | None = None):
    if not _GLOBAL_REGISTRY:
        init_global_registry()

    return _GLOBAL_REGISTRY.timer(metric_name, dimensions)


def timed(metric_name: str, dimensions: dict[str, str] | None = None):
    def with_timer(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with timer(metric_name, dimensions):
                return func(*args, **kwargs)

        return wrapped

    return with_timer
