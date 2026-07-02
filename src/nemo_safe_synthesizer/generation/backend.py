# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Abstract generator backend."""

from __future__ import annotations

import abc
from collections.abc import Callable

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
    even if ``generate`` raises.  Each cleanup step should be isolated
    so one failure doesn't prevent the next from running.

    Subclasses must implement ``initialize``, ``prepare_params``,
    ``generate``, and ``teardown``.  The ``_torn_down`` guard flag
    pattern is recommended for teardown implementations.
    """

    gen_method: Callable | None = None
    """Callable used internally for LLM generation."""

    gen_results: GenerateJobResults
    """Results from the most recent generation run."""

    config: SafeSynthesizerParameters
    """Pipeline configuration."""

    model_metadata: ModelMetadata
    """Metadata for the fine-tuned model (prompt template, sequence length, adapter path, etc.)."""

    remote: bool
    """Whether the backend calls a remote inference endpoint."""

    elapsed_time: float
    """Wall-clock duration of the last generation run in seconds."""

    workdir: Workdir
    """Working directory containing model artifacts."""

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
    def initialize(self) -> None:
        """Load the model and any required resources into memory.

        Called once before the first ``generate()`` invocation.
        Implementations should allocate GPU memory, instantiate the
        inference engine (e.g. vLLM), load LoRA adapters, and configure
        backend-specific settings such as attention backends or
        structured-output support.

        After this method returns, the backend must be ready to accept
        ``prepare_params()`` and ``generate()`` calls.
        """

    @abc.abstractmethod
    def prepare_params(self, **kwargs) -> None:
        """Translate caller-supplied sampling parameters into a backend-native form.

        Resolves, validates, and transforms high-level generation
        parameters (temperature, top-p, max tokens, structured-output
        constraints, etc.) into the format expected by the underlying
        inference engine.  The result is stored internally so that
        subsequent ``generate()`` calls use these settings.

        Must be called after ``initialize()`` and before ``generate()``.

        Args:
            **kwargs: Sampling parameters such as ``temperature``,
                ``top_p``, ``max_new_tokens``, ``repetition_penalty``,
                and backend-specific options.
        """

    @abc.abstractmethod
    def generate(
        self,
        data_actions_fn: utils.DataActionsFn | None = None,
    ) -> GenerateJobResults:
        """Run the batch generation loop and return aggregated results.

        Repeatedly prompts the model, processes each batch through the
        configured
        [`Processor`][nemo_safe_synthesizer.generation.processors.Processor],
        and accumulates valid records until the target count is reached
        or a stopping condition fires (e.g. too many consecutive invalid
        batches).  Progress and error statistics are logged after each
        batch.

        Args:
            data_actions_fn: Optional post-processing / validation
                function applied to each batch of generated records.
                Typically reverses training-time preprocessing and
                enforces user-specified data constraints.

        Returns:
            Results containing the generated DataFrame, validity
            statistics, and timing information.
        """

    @abc.abstractmethod
    def teardown(self) -> None:
        """Release all resources held by this backend.

        Frees GPU memory, destroys distributed process groups, and
        cleans up any temporary state.  Must be idempotent -- safe to
        call multiple times.  Implementations should use the
        ``_torn_down`` guard flag and isolate each cleanup step so one
        failure doesn't prevent subsequent cleanup.

        Callers should wrap ``generate()`` in ``try/finally`` to
        guarantee this runs even when generation raises.
        """
