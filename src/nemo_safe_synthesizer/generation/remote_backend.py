# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation backend that calls a remote vLLM OpenAI-compatible server."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx

from .. import utils
from ..cli.artifact_structure import Workdir
from ..config import SafeSynthesizerParameters
from ..config.generate import resolve_structured_generation_schema_method
from ..errors import GenerationError, InternalError, ParameterError
from ..llm.metadata import ModelMetadata
from ..observability import get_logger
from ..utils import load_json
from .backend import GeneratorBackend
from .batch import Batch
from .processors import Processor, create_processor
from .regex_manager import build_json_based_regex

logger = get_logger(__name__)

_COMPLETIONS_PATH = "/completions"


class RemoteBackend(GeneratorBackend):
    """Generation backend that calls an external vLLM OpenAI-compatible server.

    Unlike [`VllmBackend`][nemo_safe_synthesizer.generation.vllm_backend.VllmBackend],
    this backend never loads a model locally: it issues HTTP requests to a
    server that already serves the base model with the fine-tuned LoRA adapter
    attached (registered under ``config.generation.remote.model``). It reuses
    the shared batch loop in
    [`GeneratorBackend.generate`][nemo_safe_synthesizer.generation.backend.GeneratorBackend.generate]
    and implements only the HTTP-specific pieces, so no GPU, vLLM engine, or
    CUDA dependency is required at generation time.

    Each record is one ``/v1/completions`` request with ``n=1``; the server's
    reported ``usage.completion_tokens`` is therefore the exact per-completion
    token count. Requests within a batch are issued concurrently up to
    ``config.generation.remote.max_concurrency``.

    Structured generation maps to vLLM's OpenAI-server guided-decoding
    extensions: ``regex`` -> ``guided_regex`` and ``json_schema`` ->
    ``guided_json``. The XGrammar ``structural_tag`` method has no OpenAI-API
    equivalent and is rejected in ``prepare_params``.

    Args:
        config: Pipeline configuration. ``config.generation.remote`` must be set.
        model_metadata: Model metadata (prompt template, instruction, schema).
        workdir: Working directory containing the dataset schema.
        **kwargs: Additional options. ``use_detailed_logs`` (bool) enables
            verbose per-record error messages (disabled by default to avoid
            leaking sensitive data).
    """

    def __init__(
        self,
        config: SafeSynthesizerParameters,
        model_metadata: ModelMetadata,
        workdir: Workdir,
        **kwargs,
    ):
        self.config = config
        self.model_metadata = model_metadata
        self.workdir = workdir
        self.remote = True
        self.use_detailed_logs = kwargs.pop("use_detailed_logs", False)

        self.schema = load_json(workdir.schema_file)
        self.columns = list(self.schema["properties"].keys())
        self.prompt = utils.create_schema_prompt(
            self.columns,
            instruction=model_metadata.instruction,
            prompt_template=model_metadata.prompt_config.template,
        )
        # No local tokenizer in the remote setup: per-record token counts come
        # from the server's usage report, so the processor stays tokenizer-less.
        self.processor: Processor = create_processor(self.schema, model_metadata, config)

        self._client: httpx.Client | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._request_body: dict[str, Any] | None = None
        self._torn_down = False

    @property
    def _remote(self):
        """Remote connection config, validated to be present."""
        remote = self.config.generation.remote
        if remote is None:
            raise InternalError("RemoteBackend requires `config.generation.remote` to be configured.")
        return remote

    def initialize(self) -> None:
        """Create the HTTP client and worker pool for the remote server.

        Does not contact the server; connection errors surface on the first
        ``generate()`` request instead, with per-request context.
        """
        self._torn_down = False
        remote = self._remote

        headers: dict[str, str] = {}
        if remote.api_key_env:
            api_key = os.environ.get(remote.api_key_env)
            if not api_key:
                raise ParameterError(
                    f"Remote endpoint API key env var {remote.api_key_env!r} is not set or is empty."
                )
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=remote.endpoint_url.rstrip("/"),
            headers=headers,
            timeout=remote.timeout_seconds,
        )
        self._pool = ThreadPoolExecutor(max_workers=remote.max_concurrency)
        logger.info(
            "RemoteBackend ready: endpoint=%s model=%s max_concurrency=%d",
            remote.endpoint_url,
            remote.model,
            remote.max_concurrency,
        )

    def _get_prompt_token_count(self) -> int:
        """Return ``0`` -- no local tokenizer, so the prompt-length clamp is disabled.

        The server enforces its own context window; the per-sample ``max_tokens``
        budget from
        [`generation_max_tokens_for`][nemo_safe_synthesizer.llm.metadata.ModelMetadata.generation_max_tokens_for]
        is sized from the training-time example length and stays well within it.
        """
        return 0

    def _build_guided_params(self) -> dict[str, Any]:
        """Map structured-generation config to vLLM guided-decoding request fields.

        Returns an empty dict when structured generation is disabled.

        Raises:
            ParameterError: If the resolved schema method is ``structural_tag``,
                which has no OpenAI-API equivalent.
        """
        gen = self.config.generation
        if not gen.use_structured_generation:
            return {}

        method = resolve_structured_generation_schema_method(
            gen.structured_generation_schema_method,
            gen.structured_generation_backend,
        )
        if method == "regex":
            pc = self.model_metadata.prompt_config
            logger.info("Structured generation enabled; constraining output with guided_regex")
            return {
                "guided_regex": build_json_based_regex(
                    self.schema, self.config, bos_token=pc.bos_token, eos_token=pc.eos_token
                )
            }
        if method == "json_schema":
            logger.info("Structured generation enabled; constraining output with guided_json")
            return {"guided_json": self.schema}

        raise ParameterError(
            "Remote generation does not support "
            "`structured_generation_schema_method='structural_tag'`: the XGrammar Structural "
            "Tag has no OpenAI-API equivalent. Use 'regex' or 'json_schema' instead."
        )

    def prepare_params(self, **kwargs) -> None:
        """Build the reusable ``/v1/completions`` request body from sampling params.

        Standard OpenAI fields (``temperature``, ``top_p``, ``max_tokens``) are
        sent alongside vLLM's protocol extensions (``repetition_penalty``,
        ``top_k``, ``min_p``, ``skip_special_tokens``, ``include_stop_str_in_output``,
        ``ignore_eos``), which a vLLM OpenAI server accepts as top-level fields.
        The per-request ``prompt`` is added in ``_generate_batch``.
        """
        body: dict[str, Any] = {
            "model": self._remote.model,
            "n": 1,
            "temperature": kwargs["temperature"],
            "top_p": kwargs["top_p"],
            "max_tokens": kwargs["max_tokens"],
            # vLLM OpenAI-server protocol extensions.
            "repetition_penalty": kwargs["repetition_penalty"],
            "top_k": kwargs["top_k"],
            "min_p": kwargs["min_p"],
            "skip_special_tokens": kwargs["skip_special_tokens"],
            "include_stop_str_in_output": kwargs["include_stop_str_in_output"],
            "ignore_eos": kwargs["ignore_eos"],
        }
        body |= self._build_guided_params()
        self._request_body = body

    def _complete_one(self) -> tuple[str, int, str | None]:
        """Issue one completion request and return ``(text, completion_tokens, finish_reason)``."""
        if self._client is None or self._request_body is None:
            raise InternalError("RemoteBackend._complete_one() called before initialize()/prepare_params().")
        try:
            response = self._client.post(_COMPLETIONS_PATH, json={**self._request_body, "prompt": self.prompt})
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"Remote endpoint returned {exc.response.status_code} for {self._remote.endpoint_url}: "
                f"{exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(f"Remote endpoint request to {self._remote.endpoint_url} failed: {exc}") from exc

        try:
            data = response.json()
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            return choice.get("text", ""), int(usage.get("completion_tokens", 0)), choice.get("finish_reason")
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            # Truncate the body so a large or sensitive payload isn't echoed in full.
            raise GenerationError(
                f"Remote endpoint returned an unexpected response shape: {exc}. Body: {response.text[:500]}"
            ) from exc

    def _generate_batch(
        self,
        num_prompts_per_batch: int,
        batch: Batch,
        **_sampling_kwargs,
    ) -> Batch:
        """Issue ``num_prompts_per_batch`` concurrent completions and process the responses.

        Sampling parameters are already baked into the request body by
        ``prepare_params``; the trailing kwargs the shared loop forwards are
        accepted and ignored.
        """
        if self._pool is None:
            raise InternalError("RemoteBackend._generate_batch() called before initialize().")

        results = list(self._pool.map(lambda _: self._complete_one(), range(num_prompts_per_batch)))
        for idx, (text, completion_tokens, finish_reason) in enumerate(results):
            batch.finish_reasons[str(finish_reason or "unknown")] += 1
            batch.process(idx, text, completion_tokens=completion_tokens)
        return batch

    def teardown(self) -> None:
        """Close the HTTP client and worker pool. Idempotent."""
        if self._torn_down:
            return
        self._torn_down = True

        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False)
            except Exception:
                logger.debug("RemoteBackend pool shutdown failed during teardown", exc_info=True)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.debug("RemoteBackend client close failed during teardown", exc_info=True)
        self._pool = None
        self._client = None

    def __del__(self) -> None:
        """Clean up resources on garbage collection."""
        try:
            self.teardown()
        except Exception:
            logger.debug("RemoteBackend teardown failed during garbage collection", exc_info=True)
