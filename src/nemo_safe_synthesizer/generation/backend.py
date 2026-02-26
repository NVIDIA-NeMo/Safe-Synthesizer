# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract generator backend."""

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
    """Abstract base class for generation backends.

    Lifecycle: ``initialize`` -> ``prepare_params`` -> ``generate``
    [-> ``generate`` ...] -> ``teardown``.

    ``teardown`` must be idempotent and safe to call multiple times.
    Callers should use ``try/finally`` to guarantee ``teardown`` runs
    even if ``generate`` raises.

    Subclasses must implement :meth:`initialize`, :meth:`prepare_params`,
    :meth:`generate`, and :meth:`teardown`.

    Attributes:
        gen_method: Callable used internally for LLM generation.
        gen_results: Results from the most recent generation run.
        config: Pipeline configuration.
        model_metadata: Metadata for the fine-tuned model (prompt template,
            sequence length, adapter path, etc.).
        remote: Whether the backend calls a remote inference endpoint.
        elapsed_time: Wall-clock duration of the last generation run in
            seconds.
        workdir: Working directory containing model artifacts.
    """

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
        """Load the model into memory and prepare for generation."""

    @abc.abstractmethod
    def prepare_params(self, **kwargs):
        """Parse sampling parameters and configure the generation method."""

    @abc.abstractmethod
    def generate(
        self,
        keep_llm_state: bool = True,
        data_actions_fn: utils.DataActionsFn | None = None,
    ) -> GenerateJobResults:
        """Run the generation loop and return results.

        Args:
            keep_llm_state: If ``True``, keep the model in memory after
                generation for potential reuse.
            data_actions_fn: Optional post-processing / validation
                function applied to each batch of generated records.

        Returns:
            Results containing the generated DataFrame and statistics.
        """

    @abc.abstractmethod
    def teardown(self):
        """Release all resources held by this backend."""
