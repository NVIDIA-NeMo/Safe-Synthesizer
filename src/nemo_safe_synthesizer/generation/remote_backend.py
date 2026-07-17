# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generation backend that calls a remote vLLM OpenAI-compatible server."""

from __future__ import annotations

import json
import os
import random
import time
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
from .regex_manager import build_json_based_regex, build_json_structural_tag

logger = get_logger(__name__)

_COMPLETIONS_PATH = "/completions"

# Status codes worth retrying: request-timeout/conflict/too-early, rate limiting,
# and the transient 5xx family. Other 4xx (400/401/403/404) are permanent for a
# fixed request body and fail fast instead.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 30.0


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Return the ``Retry-After`` delay in seconds, or ``None`` if absent/unparseable.

    Only the numeric-seconds form is honored; the rarely-used HTTP-date form is
    ignored so the caller falls back to computed backoff.
    """
    value = response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _backoff_delay(attempt: int, retry_after: float | None) -> float:
    """Seconds to wait before retry ``attempt`` (0-indexed).

    Honors a server ``Retry-After`` when given; otherwise uses full-jitter
    exponential backoff (``random in [0, base * 2**attempt]``), capped at
    ``_BACKOFF_MAX_SECONDS``. Jitter spreads retries from the concurrent worker
    pool so they don't stampede the server in lockstep.
    """
    if retry_after is not None:
        return min(retry_after, _BACKOFF_MAX_SECONDS)
    capped = min(_BACKOFF_BASE_SECONDS * (2**attempt), _BACKOFF_MAX_SECONDS)
    return random.uniform(0.0, capped)


def _coerce_token_count(value: object) -> int:
    """Coerce a server-reported ``completion_tokens`` to a non-negative int.

    The ``usage`` block is advisory, so a missing, null, boolean, or malformed
    value must not fail an otherwise-valid completion -- it degrades to ``0``.
    """
    match value:
        case bool():  # bool is an int subclass; treat as "not a real count"
            return 0
        case int():
            return max(0, value)
        case float() | str():
            try:
                return max(0, int(float(value)))
            except ValueError:
                return 0
        case _:
            return 0


def _compact_json_completion(text: str) -> str:
    """Collapse a pretty-printed JSON object onto a single line for the JSONL processor.

    Servers constrained with ``structured_outputs: {"json": schema}`` may
    pretty-print the object across multiple lines. The line-oriented record
    extractor matches ``{.+?}`` with ``.`` *not* spanning newlines, so a
    multi-line object yields zero records. Re-encoding the parsed object with
    no insignificant whitespace produces the compact single-line shape the
    processor expects.

    Returns the input unchanged when it does not parse as a single JSON object
    (e.g. already-compact JSONL, multiple objects, or non-JSON text), so this
    is a safe no-op outside the pretty-printed ``json_schema`` case.
    """
    stripped = text.strip()
    if not stripped:
        return text
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        return text
    if not isinstance(obj, dict):
        return text
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


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

    Structured generation maps to vLLM's ``structured_outputs`` request field
    (``regex`` / ``json`` / ``structural_tag``), mirroring the offline
    ``StructuredOutputsParams``, so all three schema methods -- including the
    ``auto`` default -- are supported.

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
        self._prompt_token_count: int | None = None
        # Set in ``_build_structured_outputs`` when the resolved schema method is
        # ``json_schema``: such servers may pretty-print JSON, so completions are
        # compacted to single-line JSONL before the processor sees them.
        self._compact_json = False
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
                raise ParameterError(f"Remote endpoint API key env var {remote.api_key_env!r} is not set or is empty.")
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
        """Return the templated prompt's token length, or ``0`` when no tokenizer is local.

        Uses ``model_metadata.tokenizer`` opportunistically -- it is present
        on the train-then-generate path (loaded from the HF cache) but ``None``
        on the resume path (``from_metadata_json`` excludes it) and the typical
        offline-remote setup where the model was never downloaded locally. The
        remote backend never *forces* a tokenizer load, so it stays GPU- and
        download-free.

        When the count is ``0`` the prompt-length clamp in
        [`generation_max_tokens_for`][nemo_safe_synthesizer.llm.metadata.ModelMetadata.generation_max_tokens_for]
        is disabled; the server enforces its own context window and the
        per-sample ``max_tokens`` budget is sized from the training-time example
        length, so it stays well within that window regardless. The count is
        cached after the first call.
        """
        if self._prompt_token_count is not None:
            return self._prompt_token_count
        tokenizer = self.model_metadata.tokenizer
        if tokenizer is None:
            return 0
        self._prompt_token_count = len(tokenizer.encode(self.prompt))
        return self._prompt_token_count

    def _build_structured_outputs(self) -> dict[str, Any]:
        """Map structured-generation config to a vLLM ``structured_outputs`` request field.

        Mirrors the offline
        [`StructuredOutputsParams`][vllm.sampling_params.StructuredOutputsParams]
        the local backend builds, so all three schema methods are supported:
        ``regex`` -> ``{"regex": ...}``, ``json_schema`` -> ``{"json": schema}``,
        and ``structural_tag`` -> ``{"structural_tag": ...}`` (the XGrammar tag
        is sent as its JSON-encoded string, which yields multi-record JSONL).
        Returns an empty dict when structured generation is disabled.

        The legacy top-level ``guided_regex`` / ``guided_json`` fields are
        silently ignored by vLLM 0.20+ servers, so the nested
        ``structured_outputs`` field is used instead.

        For ``json_schema`` the constraint is compacted at two layers: on the
        ``vllm`` dialect ``disable_any_whitespace`` makes xgrammar emit
        single-line JSON at the source (saving completion tokens), and
        ``self._compact_json`` is set so ``_generate_batch`` also collapses any
        residual multi-line output -- a portable net for the ``openai`` dialect,
        where that vLLM-only field can't be sent. The ``regex`` and
        ``structural_tag`` methods already enforce the single-line shape, so no
        post-processing is needed and ``auto`` (-> ``structural_tag``) sidesteps
        the issue entirely.
        """
        gen = self.config.generation
        self._compact_json = False
        structured_generation = gen.structured_generation
        if not structured_generation.enabled:
            return {}

        method = resolve_structured_generation_schema_method(
            structured_generation.schema_method,
            structured_generation.backend,
        )
        pc = self.model_metadata.prompt_config
        if method == "regex":
            logger.info("Structured generation enabled; constraining output with a regex")
            regex = build_json_based_regex(self.schema, self.config, bos_token=pc.bos_token, eos_token=pc.eos_token)
            return {"structured_outputs": {"regex": regex}}
        if method == "json_schema":
            self._compact_json = True
            json_constraint: dict[str, Any] = {"json": self.schema}
            if self._remote.dialect == "vllm":
                # Source fix for vLLM: xgrammar (its default backend) emits compact JSON
                # when whitespace is disabled, so the server never pretty-prints across
                # lines and no completion tokens are wasted on whitespace. This is a vLLM
                # protocol extension -- strict OpenAI servers (the "openai" dialect, e.g.
                # NIM/TRT-LLM) 400 on it -- so it is gated like the other vLLM extensions.
                # ``_compact_json`` stays armed regardless as a portable safety net.
                json_constraint["disable_any_whitespace"] = True
            logger.info(
                "Structured generation enabled; constraining output with a JSON schema "
                "(remote completions compacted to single-line JSONL)"
            )
            return {"structured_outputs": json_constraint}
        if method == "structural_tag":
            logger.info("Structured generation enabled; constraining output with an XGrammar structural tag")
            tag = build_json_structural_tag(self.schema, self.config, bos_token=pc.bos_token, eos_token=pc.eos_token)
            return {"structured_outputs": {"structural_tag": tag}}

        raise InternalError(f"Unhandled structured-generation schema method: {method!r}")

    def prepare_params(self, **kwargs) -> None:
        """Build the reusable ``/v1/completions`` request body from sampling params.

        The request fields depend on ``config.generation.remote.dialect``:

        - ``"vllm"`` (default): the universal OpenAI fields (``temperature``,
          ``top_p``, ``max_tokens``, ``n``) plus vLLM's protocol extensions
          (``repetition_penalty``, ``top_k``, ``min_p``, ``skip_special_tokens``,
          ``include_stop_str_in_output``, ``ignore_eos``).
        - ``"openai"``: only the universal fields, for stricter servers (e.g.
          NIM / TensorRT-LLM) that reject the vLLM extensions with a 400.

        The resolved sampling values are identical across dialects; only which
        fields go on the wire differs. The prompt is constant across every
        request in a run, so it is baked into the body here once rather than
        merged per request.
        """
        body: dict[str, Any] = {
            "model": self._remote.model,
            "prompt": self.prompt,
            "n": 1,
            "temperature": kwargs["temperature"],
            "top_p": kwargs["top_p"],
            "max_tokens": kwargs["max_tokens"],
        }
        if self._remote.dialect == "vllm":
            body |= {
                "repetition_penalty": kwargs["repetition_penalty"],
                "top_k": kwargs["top_k"],
                "min_p": kwargs["min_p"],
                "skip_special_tokens": kwargs["skip_special_tokens"],
                "include_stop_str_in_output": kwargs["include_stop_str_in_output"],
                "ignore_eos": kwargs["ignore_eos"],
            }
        body |= self._build_structured_outputs()
        self._request_body = body

    def _complete_one(self) -> tuple[str, int, str | None]:
        """Issue one completion (with transient-failure retries) and parse the result.

        Returns ``(text, completion_tokens, finish_reason)``.
        """
        return self._parse_completion(self._post_completion())

    def _post_completion(self) -> httpx.Response:
        """POST one completion request, retrying transient failures with backoff.

        Connection errors, timeouts, and retryable status codes
        (``_RETRYABLE_STATUS``) are retried up to ``remote.max_retries`` times
        with full-jitter exponential backoff, honoring a ``Retry-After`` header
        when present. A non-retryable status (e.g. 400/401/404) fails
        immediately via ``raise_for_status`` -- it would fail identically for
        every record. ``GenerationError`` is raised once retries are exhausted.
        """
        if self._client is None or self._request_body is None:
            raise InternalError("RemoteBackend._post_completion() called before initialize()/prepare_params().")

        endpoint = self._remote.endpoint_url
        max_retries = self._remote.max_retries
        last_error = "no attempts made"

        for attempt in range(max_retries + 1):
            retry_after: float | None = None
            try:
                response = self._client.post(_COMPLETIONS_PATH, json=self._request_body)
            except httpx.HTTPError as exc:
                last_error = f"request failed: {exc}"
            else:
                if response.status_code not in _RETRYABLE_STATUS:
                    return self._raise_for_status(response)
                last_error = f"status {response.status_code}: {response.text[:200]}"
                retry_after = _parse_retry_after(response)

            if attempt == max_retries:
                break
            delay = _backoff_delay(attempt, retry_after)
            logger.warning(
                "Remote endpoint %s transient failure (%s); retry %d/%d in %.1fs",
                endpoint,
                last_error,
                attempt + 1,
                max_retries,
                delay,
            )
            time.sleep(delay)

        raise GenerationError(f"Remote endpoint {endpoint} failed after {max_retries + 1} attempt(s): {last_error}")

    def _raise_for_status(self, response: httpx.Response) -> httpx.Response:
        """Return ``response`` if OK, else raise ``GenerationError`` with a truncated body."""
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise GenerationError(
                f"Remote endpoint {self._remote.endpoint_url} returned {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            ) from exc
        return response

    def _parse_completion(self, response: httpx.Response) -> tuple[str, int, str | None]:
        """Extract ``(text, completion_tokens, finish_reason)`` from a completion response.

        The ``usage`` token count is advisory and coerced defensively; only a
        missing/empty ``choices`` array is fatal, since it means no completion
        was produced.
        """
        try:
            data = response.json()
            choice = data["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            # Truncate the body so a large or sensitive payload isn't echoed in full.
            raise GenerationError(
                f"Remote endpoint {self._remote.endpoint_url} returned an unexpected response shape: {exc}. "
                f"Body: {response.text[:500]}"
            ) from exc

        usage = data.get("usage") or {}
        return (
            choice.get("text", ""),
            _coerce_token_count(usage.get("completion_tokens")),
            choice.get("finish_reason"),
        )

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

        futures = [self._pool.submit(self._complete_one) for _ in range(num_prompts_per_batch)]
        for idx, future in enumerate(futures):
            text, completion_tokens, finish_reason = future.result()
            if self._compact_json:
                text = _compact_json_completion(text)
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
