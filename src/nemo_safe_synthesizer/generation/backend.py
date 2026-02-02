# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from typing import Callable

from .. import utils
from ..cli.artifact_structure import Workdir
from ..config import SafeSynthesizerParameters
from ..generation.results import GenerateJobResults
from ..llm.metadata import ModelMetadata
from ..observability import get_logger

logger = get_logger(__name__)


class GeneratorBackend(metaclass=abc.ABCMeta):
    gen_method: Callable | None = None
    gen_results: GenerateJobResults
    config: SafeSynthesizerParameters
    model_metadata: ModelMetadata
    remote: bool
    elapsed_time: float
    workdir: Workdir

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "prepare_args")
            and callable(subclass.prepare_params)
            and hasattr(subclass, "load")
            and callable(subclass.initialize)
            and hasattr(subclass, "generate")
            and callable(subclass.generate)
            or NotImplemented
        )

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def prepare_params(self, **kwargs):
        pass

    @abc.abstractmethod
    def generate(
        self,
        keep_llm_state: bool = True,
        data_actions_fn: utils.DataActionsFn | None = None,
    ) -> GenerateJobResults:
        pass

    @abc.abstractmethod
    def teardown(self):
        pass
