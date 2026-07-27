# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-based generation backend for tabular data synthesis."""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
from vllm import LLM as vLLM
from vllm import RequestOutput
from vllm.config import AttentionConfig, StructuredOutputsConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.inputs.llm import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from .. import utils
from ..cli.artifact_structure import Workdir
from ..cli.wandb_setup import log_observability_event
from ..config import SafeSynthesizerParameters
from ..config.generate import (
    resolve_structured_generation_schema_method,
    structural_tag_backend_error_message,
)
from ..data_processing.dataset import relax_numeric_bounds
from ..defaults import DEFAULT_SAMPLING_PARAMETERS, FIXED_RUNTIME_GENERATE_ARGS
from ..errors import InternalError, ParameterError
from ..generation.backend import GeneratorBackend
from ..generation.batch import Batch
from ..generation.processors import EncodeOnlyTokenizer, Processor, TabularDataProcessor, create_processor
from ..generation.regex_manager import build_json_based_regex, build_json_structural_tag
from ..generation.results import GenerateJobResults, GenerationBatches, GenerationStatus
from ..generation.vllm_observability import (
    GenerationObservability,
    NvmlPeakSampler,
    probe_engine_runtime_config,
    read_loadavg,
    read_vllm_runtime_metrics,
)
from ..llm.metadata import ModelMetadata
from ..llm.utils import ModelRef, cleanup_memory, get_max_vram
from ..observability import get_logger, heartbeat
from ..utils import all_equal_type, load_json

logger = get_logger(__name__)

if torch.cuda.is_available():
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
else:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

# vLLM 0.20+ runs a deep_gemm FP8 kernel warmup during engine init that crashes
# when the optional `deep_gemm` package isn't installed. We don't use FP8 kernels
# for the supported NSS models, so default the warmup off; users can still opt
# in by exporting VLLM_USE_DEEP_GEMM=1.
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

# CVE-2025-69872: diskcache (pulled in transitively by outlines and used by
# vLLM's optional on-disk outlines cache) deserializes cached values with
# pickle/cloudpickle and is therefore RCE-vulnerable if another principal can
# write into the cache directory. Neither library exposes a way to swap the
# serializer, so we mitigate at the boundary:
#   1. Keep vLLM's opt-in diskcache off (its default is an in-memory LRUCache).
#      Hard-set (not setdefault) so a user env can't silently flip on a
#      pickle-deserializing code path.
#   2. Pin OUTLINES_CACHE_DIR to a per-user path and chmod it to 0700, since
#      outlines always uses diskcache for its FSM/index cache.
os.environ["VLLM_V1_USE_OUTLINES_CACHE"] = "0"


def _build_rope_hf_overrides(model_metadata: ModelMetadata) -> dict[str, Any] | None:
    """Return vLLM ``hf_overrides`` needed for NSS RoPE context extension."""
    rope_scaling = model_metadata.rope_scaling
    if rope_scaling is None or rope_scaling.factor <= 1.0:
        return None

    rope_parameters = dict(rope_scaling.rope_parameters)
    rope_type = rope_parameters.get("rope_type", rope_scaling.rope_type)
    if rope_type == "default":
        rope_type = "linear"

    rope_parameters.update(
        {
            "rope_type": rope_type,
            "factor": float(rope_scaling.factor),
            "original_max_position_embeddings": model_metadata.base_max_seq_length,
            "rope_theta": float(rope_scaling.theta),
        }
    )

    # vLLM 0.24 expects only native RoPE config overrides here. The effective
    # context length still belongs on the top-level ``LLM(max_model_len=...)``.
    return {
        "rope_parameters": rope_parameters,
    }


def _tokens_prompt(prompt_token_ids: list[int]) -> TokensPrompt:
    """Build a vLLM token prompt for pre-tokenized generation."""
    return TokensPrompt(prompt_token_ids=prompt_token_ids)


def _secure_outlines_cache_dir() -> None:
    """Pin ``OUTLINES_CACHE_DIR`` to a per-user path and tighten permissions.

    Respects an explicit ``OUTLINES_CACHE_DIR`` set by the operator (so CI and
    multi-tenant deployments can choose their own private location), but always
    creates the directory with 0700 permissions to prevent co-tenants from
    poisoning the diskcache (CVE-2025-69872).

    When unset, picks a per-user path under ``$XDG_CACHE_HOME`` or
    ``$HOME/.cache`` and falls back to a UID-scoped subdir of the system temp
    dir for distroless/rootless containers where ``$HOME`` is ``/``.
    """
    cache_dir_env = os.environ.get("OUTLINES_CACHE_DIR")
    if cache_dir_env:
        cache_dir = Path(cache_dir_env)
    else:
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        home_dir = os.path.normpath(os.path.expanduser("~"))
        if xdg_cache_home:
            cache_root = Path(xdg_cache_home)
        elif home_dir != "/" and Path(home_dir).is_dir():
            cache_root = Path(home_dir) / ".cache"
        else:
            uid = getattr(os, "getuid", lambda: "default")()
            cache_root = Path(tempfile.gettempdir()) / f".cache-{uid}"
        cache_dir = cache_root / "nemo-safe-synthesizer" / "outlines"
        os.environ["OUTLINES_CACHE_DIR"] = str(cache_dir)

    try:
        # Set the umask to 077 to prevent other principals from writing to the
        # cache directory between the mkdir and chmod calls.
        old_umask = os.umask(0o077)
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        finally:
            os.umask(old_umask)
        # Also explicitly set permissions to 0700 for the situation where the
        # directory already exists and is not 0700.
        cache_dir.chmod(0o700)
    except OSError as exc:
        logger.warning(
            "Could not enforce 0700 permissions on outlines cache dir %s: %s. "
            "If this path is shared with other principals, set OUTLINES_CACHE_DIR "
            "to a private location (CVE-2025-69872).",
            cache_dir,
            exc,
        )


_secure_outlines_cache_dir()


def _is_redis_available() -> bool:
    """Return True if the ``redis`` package is importable."""
    try:
        import redis  # noqa: F401  # ty:ignore[unresolved-import]

        return True
    except ImportError:
        return False


class _NoopRemoteCacheBackend:
    """No-op stand-in for ``RedisRemoteCacheBackend``.

    All reads return ``None``; all writes are silently dropped.
    """

    def get(self, key: str) -> bytes | None:
        return None

    def put(self, key: str, data: bytes) -> None:
        pass


def _install_noop_remote_cache_backends() -> None:
    """Replace the Inductor ``RemoteAutotuneCache`` backend with a no-op.

    ``torch.compile`` uses ``RemoteAutotuneCache`` (backed by Redis) to share
    autotuning results across processes.  When the ``redis`` package is not
    installed the default backend raises at construction time, which breaks
    ``torch.compile`` in environments that never intended to run Redis.

    This function patches *only* ``RemoteAutotuneCache`` -- the single
    Redis-backed cache that surfaces errors during normal Safe-Synthesizer
    runs.  Other Redis caches (``RemoteFxGraphCache``,
    ``RemoteBundledAutotuneCache``, etc.) are left untouched so they keep
    working if a future dependency pulls in ``redis``.

    The override is skipped entirely when ``redis`` *is* importable, leaving
    the default backend intact and avoiding any ``torch.compile`` performance
    regression.
    """
    if _is_redis_available():
        return

    try:
        from torch._inductor.remote_cache import RemoteAutotuneCache

        RemoteAutotuneCache.backend_override_cls = _NoopRemoteCacheBackend  # ty: ignore[invalid-assignment]
        logger.debug("Installed no-op backend for RemoteAutotuneCache (redis unavailable)")
    except ImportError:
        logger.debug("RemoteAutotuneCache is unavailable; skipping no-op backend patch", exc_info=True)


_install_noop_remote_cache_backends()


class VllmBackend(GeneratorBackend):
    """Generation backend using vLLM for high-throughput inference.

    Loads the base model with a LoRA adapter via vLLM and generates
    synthetic records in batches.  Supports optional structured
    generation (regex or JSON schema) to constrain outputs.

    ``LoRARequest("lora", 1, str(adapter_path))`` is passed to
    ``llm.generate`` when an adapter is available. The vLLM engine uses
    ``config.training.lora_r`` as ``max_lora_rank``.

    Args:
        config: Pipeline configuration.
        model_metadata: Model metadata (prompt template, adapter path,
            sequence length, etc.).
        workdir: Working directory containing the adapter and schema.
        **kwargs: Additional options.  ``use_detailed_logs`` (bool)
            enables verbose error messages (disabled by default to
            avoid leaking sensitive data).
    """

    def __init__(
        self,
        config: SafeSynthesizerParameters,
        model_metadata: ModelMetadata,
        workdir: Workdir,
        **kwargs,
    ):
        self.model_metadata = model_metadata
        self.config = config
        self.remote = False
        self.workdir = workdir
        self.schema = load_json(self.workdir.schema_file)
        if not self.config.generation.validation.enforce_numeric_range:
            # Drop float range bounds so out-of-range floating-point values are
            # accepted instead of rejected in post-generation validation. Integer
            # and enum constraints (which the grammar enforces) are unaffected.
            self.schema = relax_numeric_bounds(self.schema)
        self.columns = list(self.schema["properties"].keys())
        self.prompt = utils.create_schema_prompt(
            self.columns,
            instruction=self.model_metadata.instruction,
            prompt_template=self.model_metadata.prompt_config.template,
        )
        self.llm: vLLM | None = None
        self._prompt_token_count: int | None = None
        # Populated in ``initialize()`` after engine build; pre-declared
        # here so ``generate()`` can always read it (tests that mock the
        # backend without calling ``initialize()`` see the empty dict).
        self._engine_runtime_config: dict[str, Any] = {}

        # Do not generate detailed error messages in production to avoid leaking sensitive data.
        self.use_detailed_logs = kwargs.pop("use_detailed_logs", False)
        self.gen_method: partial | None = None
        self._gen_method: partial | None = None
        # Initial processor without a tokenizer; replaced in ``initialize()`` with a
        # tokenizer-aware processor once the vLLM engine (and its tokenizer) exists.
        # This lets callers introspect the processor type before ``initialize()``,
        # at the cost of token counts being zero until the tokenizer is attached.
        self.processor: Processor = create_processor(self.schema, self.model_metadata, self.config)
        adapter_path = self.workdir.adapter_path if self.workdir.adapter_path else self.model_metadata.adapter_path
        self.lora_req = LoRARequest("lora", 1, str(adapter_path)) if adapter_path else None
        self._torn_down = False

    def teardown(self) -> None:
        """Release GPU memory and distributed resources. Idempotent -- safe to call multiple times."""
        if self._torn_down:
            return
        self._torn_down = True

        try:
            cleanup_dist_env_and_memory()
        except Exception:
            logger.debug("cleanup_dist_env_and_memory failed during teardown", exc_info=True)

        self.llm = None
        self._gen_method = None
        self.gen_method = None

        try:
            cleanup_memory()
        except Exception:
            logger.debug("cleanup_memory failed during teardown", exc_info=True)

    def __del__(self) -> None:
        """Clean up resources on garbage collection."""
        try:
            self.teardown()
        except Exception:
            logger.debug("VllmBackend teardown failed during garbage collection", exc_info=True)

    def initialize(self, **kwargs) -> None:
        """Initialize and load the model into memory.

        Creates the vLLM engine and then builds the record processor
        with the engine's tokenizer so that exact token counts are
        available during generation.
        """
        self._torn_down = False

        # vLLM 0.12+ accepts attention_config as a constructor arg (replaces the
        # VLLM_ATTENTION_BACKEND env var used in 0.11.x).
        attn_backend = self.config.generation.attention_backend
        attention_config = (
            AttentionConfig(backend=attn_backend)  # ty: ignore[invalid-argument-type] -- vLLM validates backend strings.
            if attn_backend not in (None, "auto")
            else None
        )

        max_vram = get_max_vram()
        # note this only works for single GPU setups
        max_vram = max_vram.get(0, 0.8)

        # vllm requires this "config" to set the backend ahead of time.
        structured_outputs_config = StructuredOutputsConfig(
            backend=self.config.generation.structured_generation.backend,
        )
        model_ref = ModelRef.parse(self.config.training.pretrained_model)
        hf_overrides = _build_rope_hf_overrides(self.model_metadata)

        with heartbeat("Model loading", logger_name=__name__, model=self.config.training.pretrained_model):
            self.llm = vLLM(
                model=model_ref.target(),
                gpu_memory_utilization=max_vram,
                max_model_len=self.model_metadata.max_seq_length,
                enable_lora=True,
                max_lora_rank=self.config.training.lora_r,
                structured_outputs_config=structured_outputs_config,
                attention_config=attention_config,
                trust_remote_code=model_ref.trust_remote_code,
                hf_overrides=hf_overrides,
            )

        # Cache the engine's *effective* runtime config once at init. Read by
        # ``generate()`` to populate the generation-complete observability
        # event; used by the benchmark harness (and any other intent-comparing
        # caller) to detect "flag didn't engage" mismatches against what was
        # asked for.
        self._engine_runtime_config = probe_engine_runtime_config(self.llm)

        tokenizer: EncodeOnlyTokenizer = self.llm.get_tokenizer()
        self.processor = create_processor(
            self.schema,
            self.model_metadata,
            self.config,
            tokenizer=tokenizer,
        )

    def _get_prompt_token_count(self) -> int:
        """Return the templated prompt's tokenized length, cached after first call.

        Uses the loaded vLLM tokenizer so encoding matches what the engine
        will see at runtime. Returns ``0`` when the engine has not yet been
        initialized -- callers fall back to a prompt-agnostic ceiling in
        that case.
        """
        if self._prompt_token_count is not None:
            return self._prompt_token_count
        if self.llm is None:
            return 0
        tokenizer = self.llm.get_tokenizer()
        self._prompt_token_count = len(tokenizer.encode(self.prompt))
        return self._prompt_token_count

    def _build_structured_output_params(self) -> StructuredOutputsParams | None:
        """Build structured output parameters based on generation config.

        Returns:
            StructuredOutputsParams if structured generation is enabled, None otherwise.
        """
        if not self.config.generation.structured_generation.enabled:
            return None

        params: dict[str, Any] = {}
        schema_method = resolve_structured_generation_schema_method(
            self.config.generation.structured_generation.schema_method,
            self.config.generation.structured_generation.backend,
        )

        if schema_method == "regex":
            logger.info("Structured generation is enabled, using a regex to enforce the schema")
            pc = self.model_metadata.prompt_config
            regex = build_json_based_regex(
                self.schema,
                self.config,
                bos_token=pc.bos_token,
                eos_token=pc.eos_token,
            )
            params["regex"] = regex
        elif schema_method == "json_schema":
            params["json"] = self.schema
        elif schema_method == "structural_tag":
            backend = self.config.generation.structured_generation.backend
            if message := structural_tag_backend_error_message(backend):
                raise ParameterError(message)
            logger.info("Structured generation is enabled, using an XGrammar Structural Tag")
            pc = self.model_metadata.prompt_config
            params["structural_tag"] = build_json_structural_tag(
                self.schema,
                self.config,
                bos_token=pc.bos_token,
                eos_token=pc.eos_token,
            )

        return StructuredOutputsParams(**params)

    def _resolve_temperature(self, kwargs: dict[str, Any]) -> float:
        """Resolve temperature value based on sampling settings.

        Args:
            kwargs: Dictionary containing sampling parameters.

        Returns:
            The resolved temperature value.

        Raises:
            ValueError: If do_sample is False but temperature is nonzero.
        """
        match kwargs:
            case {
                "do_sample": bool(samp),
                "temperature": float(temp),
                **rest,  # noqa: F841
            } if samp is False and temp > 0.0:
                raise ValueError(
                    f"Invalid arguments - Cannot set a nonzero temperature (`temperature=={temp}`) for `do_sample=={samp}`"
                )

            case {
                "do_sample": bool(samp),
                **rest,  # noqa: F841
            } if samp is False:
                logger.warning(f"do_sample={samp}. Setting temperature=0.0 for greedy decoding.")
                return 0.0

            case {"temperature": float(val), **rest}:  # noqa: F841
                return val

            case _:
                logger.warning(
                    f"Temperature undefined; Setting temperature={DEFAULT_SAMPLING_PARAMETERS['temperature']}."
                )
                return DEFAULT_SAMPLING_PARAMETERS["temperature"]

    def _get_api_param_mapping(self, resolved_temperature: float) -> dict[str, Any]:
        """Get the mapping from our API parameters to vLLM parameters.

        Args:
            resolved_temperature: The resolved temperature value to use.

        Returns:
            Dictionary mapping parameter names to transformation functions.
        """
        return {
            "max_new_tokens": lambda x: ("max_tokens", x),
            "eos_token_id": lambda x: (
                "stop_token_ids",
                x if isinstance(x, list) else [x],
            ),
            "temperature": lambda x: ("temperature", resolved_temperature),
            "num_beams": lambda x: (None, None),
            "early_stopping": lambda x: (None, None),
        }

    def _transform_kwargs_to_sampling_params(
        self, kwargs: dict[str, Any], api_mapping: dict[str, Any]
    ) -> dict[str, Any]:
        """Transform kwargs using the API mapping to vLLM sampling parameters.

        Args:
            kwargs: Dictionary containing our API parameters.
            api_mapping: Mapping from our parameter names to vLLM parameters.

        Returns:
            Dictionary of vLLM-compatible sampling parameters.
        """
        sampling_params = {}

        for param, val in kwargs.items():
            if action := api_mapping.get(param):
                new_param, new_val = action(val)
                if new_param != param or new_val != val:
                    logger.info(f"remapped {param}={val} -> {new_param}={new_val}")
                param, val = new_param, new_val

            # Skip parameters that were mapped to None (signals exclusion)
            if param is not None:
                sampling_params[param] = val

        return sampling_params

    def prepare_params(self, **kwargs) -> None:
        """Parse parameters and configure the generation method.

        Parses a dictionary of parameters into ``SamplingParams``, applying
        necessary transformations from the Safe Synthesizer API to vLLM's API.
        ``num_beams`` is omitted because vLLM 0.24 no longer accepts the old
        ``beam_width`` sampling parameter.

        Args:
            **kwargs: Sampling parameters to configure.
        """
        structured_output_params = self._build_structured_output_params()
        kwargs |= {"structured_outputs": structured_output_params}

        resolved_temperature = self._resolve_temperature(kwargs)
        api_mapping = self._get_api_param_mapping(resolved_temperature)
        sampling_params = self._transform_kwargs_to_sampling_params(kwargs, api_mapping)

        real_params = SamplingParams(**sampling_params)
        logger.debug(f"SamplingParams: {real_params!r}")

        # Create a partially parametrized version of the underlying vllm.LLM.generate
        # method that is immediately callable downstream.
        if self.llm is None:
            raise InternalError(
                "VllmBackend._configure_sampling_params() called before initialize() -- self.llm is None."
            )
        self._gen_method = partial(
            self.llm.generate,
            sampling_params=real_params,
            lora_request=self.lora_req,
            # Show vLLM's tqdm progress bar only when debug logging is enabled.
            use_tqdm=logger.isEnabledFor(logging.DEBUG),
        )

    def _generate(
        self,
        prompts: str | list[str] | None = None,
        input_ids: torch.TensorType | list[int] | list[list[int]] | None = None,
        **kwargs,
    ) -> list[RequestOutput]:
        """Dispatch a generation call to the underlying vLLM engine.

        Exactly one of ``prompts`` or ``input_ids`` must be provided.

        Args:
            prompts: Text prompts to generate from.
            input_ids: Pre-tokenized prompt IDs (tensor, flat list for a
                single prompt, or nested list for multiple prompts).

        Returns:
            List of vLLM ``RequestOutput`` objects.

        Raises:
            ValueError: If both or neither of ``prompts`` / ``input_ids``
                are provided, or if the generation method is not configured.
        """
        if prompts is None and input_ids is None:
            raise ValueError("Either prompts or input_ids must be provided.")

        if (prompts is not None) and (input_ids is not None):
            raise ValueError("Only one of prompts or input_ids should be provided.")

        if self._gen_method is None:
            raise ValueError("gen_method must be provided.")

        match self._gen_method.keywords:
            case {"sampling_params": _, **rest_}:  # noqa: F841
                result = None
                match input_ids:
                    case torch.Tensor():
                        token_ids = input_ids.tolist()
                        logger.debug("vllm generate: token prompts (torch.Tensor)")
                        if all_equal_type(token_ids, int, flatten_iter=False):
                            result = self._gen_method(prompts=_tokens_prompt(token_ids))
                        else:
                            result = self._gen_method(prompts=[_tokens_prompt(ids) for ids in token_ids])
                    case [[*_inner], *_] if all_equal_type(input_ids, int):  # ty: ignore[invalid-argument-type]
                        assert isinstance(input_ids, list)
                        token_ids_batch = cast(list[list[int]], input_ids)
                        logger.debug(f"vllm generate: token prompts ({len(input_ids)} prompts)")
                        result = self._gen_method(prompts=[_tokens_prompt(ids) for ids in token_ids_batch])
                    case [*ids] if all_equal_type(ids, int, flatten_iter=False):
                        logger.debug("vllm generate: token prompts (single flat list)")
                        result = self._gen_method(prompts=_tokens_prompt(ids))
                    case None:
                        logger.debug(
                            f"vllm generate: processing {len(prompts) if isinstance(prompts, list) else 1} prompts"
                        )
                        return self._gen_method(prompts=prompts)
                    case _:
                        raise ValueError("input_ids are not a tensor, list, or None!")

                if result is None:
                    raise ValueError("input_ids are not a tensor, list, or None!")
                return result
            case _:
                raise ValueError("input ids are not a tensor or list!")

    def _generate_batch(
        self,
        num_prompts_per_batch: int,
        batch: Batch,
        **sampling_kwargs,
    ) -> Batch:
        """Run generation on a batch of prompts.

        Args:
            num_prompts_per_batch: Number of prompts to run per batch.
            batch: Batch object, which contains a processor for extracting
                records from the generated text.

        Returns:
            Batch object that contains the generated records and associated statistics.
        """
        logger.debug(f"generation prompt ({len(self.prompt)} chars):\n{self.prompt}")
        prompt_list = [self.prompt] * num_prompts_per_batch

        # `n` is the number of output sequences per prompt.
        # Subsequent processing assumes `n=1`, so we hardcode it here.
        sampling_kwargs.update({"n": 1})

        outputs = self._generate(prompts=prompt_list, **sampling_kwargs)

        for idx, output in enumerate(outputs):
            out = output.outputs[0]
            batch.finish_reasons[str(out.finish_reason or "unknown")] += 1
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"prompt {idx}: {len(out.token_ids)} tokens, "
                    f"finish_reason={out.finish_reason}, "
                    f"stop_reason={out.stop_reason}"
                )
            batch.process(idx, out.text, completion_tokens=len(out.token_ids))

        return batch

    def _log_batch_timing_and_progress(
        self,
        batch: Batch,
        duration: float,
        num_records: int,
        num_valid_records: int,
        batches: GenerationBatches,
    ) -> None:
        """Log batch timing and progress as a structured Rich table.

        Emits structured data via ``logger.user.info`` that is rendered
        as a Rich ASCII table on the console and as key/value pairs in
        JSON logs.
        """
        records_per_second = 0 if duration == 0 else batch.num_valid_records / duration

        # Build structured data - processor renders as table for console
        progress_data: dict[str, int | float] = {
            "records_per_second": round(records_per_second, 2),
            "duration_seconds": round(duration, 2),
            "valid_records_generated": batches.num_valid_records,
            "target_records": self.config.generation.num_records,
            "progress_fraction": round(batches.num_valid_records / self.config.generation.num_records, 4),
        }
        if batch.total_completion_tokens > 0 and duration > 0:
            progress_data["tokens_per_second"] = round(batch.total_completion_tokens / duration, 1)
            progress_data["valid_tokens_per_second"] = round(batch.total_valid_record_tokens / duration, 1)

        # Pass structured data - processor renders for console, JSON keeps as-is
        logger.user.info(
            "",
            extra={
                "ctx": {
                    "render_table": True,
                    "tabular_data": progress_data,
                    "title": "Batch Progress",
                }
            },
        )

    def generate(
        self,
        data_actions_fn: utils.DataActionsFn | None = None,
    ) -> GenerateJobResults:
        """Generate synthetic tabular data in batches until the target count is reached.

        Iterates over generation batches, applying the processor to each
        LLM output, until the configured ``num_records`` target is met or
        a stopping condition fires.

        Non-tabular processors need BOS/EOS delimiters in the raw text, so
        generation keeps special tokens for those processors and strips them
        only for ``TabularDataProcessor``. Native EOS stopping remains enabled
        through ``ignore_eos=False``.

        Wrapped in an :class:`NvmlPeakSampler` context so a
        generation-complete observability event is emitted at end of
        the call carrying peak device VRAM, host loadavg pre/post, vLLM's
        kv_cache_usage_perc / prefix_cache_hit_rate / spec_accept_rate
        (read at end-of-generation), and the engine's effective runtime
        config probed at engine-init time. Sampler is degraded-mode when
        NVML is unavailable; the event still emits with ``peak_vram_gb=None``.

        Args:
            data_actions_fn: Optional post-processing / validation function
                applied to each batch of generated records.

        Returns:
            Results containing the generated DataFrame and statistics.
        """
        loadavg_pre = read_loadavg()
        sampler = NvmlPeakSampler()
        try:
            with sampler:
                self._run_generation(data_actions_fn)
        finally:
            # Emit regardless of success: the sampler thread has been joined
            # by ``with`` exit, so ``sampler.peak_gb`` is the final peak, and
            # the event carries whatever measurements were captured up to a
            # failure point.
            self._emit_generation_observability(sampler, loadavg_pre)

        return self.gen_results

    def _run_generation(self, data_actions_fn: utils.DataActionsFn | None) -> None:
        """Run the batch-generation loop until the target or a stop condition fires.

        Populates ``self.gen_results`` and ``self.elapsed_time``. Extracted
        from :meth:`generate` so that method stays a thin observability
        bracket around the loop.
        """
        generation_start = time.monotonic()
        need_special_token_outputs = not isinstance(self.processor, TabularDataProcessor)
        sampling_kwargs = dict(
            temperature=self.config.generation.temperature,
            repetition_penalty=self.config.generation.repetition_penalty,
            top_p=self.config.generation.top_p,
            top_k=FIXED_RUNTIME_GENERATE_ARGS["top_k"],
            min_p=FIXED_RUNTIME_GENERATE_ARGS["min_p"],
            max_tokens=self.model_metadata.generation_max_tokens_for(self._get_prompt_token_count()),
            skip_special_tokens=not need_special_token_outputs,
            include_stop_str_in_output=need_special_token_outputs,
            ignore_eos=False,
        )

        self.prepare_params(**sampling_kwargs)

        # The batches object collects batches and keeps track of the stopping condition.
        batches = GenerationBatches(
            target_num_records=self.config.generation.num_records,
            invalid_fraction_threshold=self.config.generation.invalid_fraction_threshold,
            patience=self.config.generation.patience,
            data_actions_fn=data_actions_fn,
        )

        with heartbeat(
            "Generation",
            logger_name=__name__,
            target_records=self.config.generation.num_records,
            progress_note=("Long stretches with no new records are normal."),
        ):
            while batches.num_valid_records < self.config.generation.num_records:
                # Generate a batch from prompts and process the responses.
                num_prompts = batches.get_next_num_prompts()
                start_time = time.perf_counter()
                batch: Batch = self._generate_batch(
                    num_prompts_per_batch=num_prompts,
                    batch=Batch(processor=self.processor),
                    **sampling_kwargs,
                )
                duration = time.perf_counter() - start_time
                batches.add_batch(batch)

                # Log generation summary and progress.
                batch.log_summary(detailed_errors=self.use_detailed_logs)
                self._log_batch_timing_and_progress(
                    batch=batch,
                    duration=duration,
                    num_records=self.config.generation.num_records,
                    num_valid_records=batches.num_valid_records,
                    batches=batches,
                )
                # Check if the generation job should stop.
                if batches.status in [
                    GenerationStatus.STOP_NO_RECORDS,
                    GenerationStatus.STOP_METRIC_REACHED,
                ]:
                    break

        batches.job_complete()
        batches.log_status()

        max_num_records = (
            self.config.generation.num_records
            if self.config.data.group_training_examples_by is None and batches.status == GenerationStatus.COMPLETE
            else None
        )

        self.elapsed_time = time.monotonic() - generation_start
        self.gen_results = GenerateJobResults.from_batches(
            batches=batches,
            columns=self.columns,
            max_num_records=max_num_records,
            elapsed_time=self.elapsed_time,
        )

    def _emit_generation_observability(
        self, sampler: NvmlPeakSampler, loadavg_pre: tuple[float, float, float] | None
    ) -> None:
        """Assemble and route the generation-complete observability event.

        Reads end-of-generation vLLM metrics plus the NVML peak captured by
        ``sampler``, builds a :class:`GenerationObservability`, and routes it to
        structured logs and (when a run is active) wandb. Best-effort: any
        failure here is logged and swallowed so observability never masks a
        generation error propagating through :meth:`generate`'s ``finally``.
        """
        try:
            vllm_metrics = read_vllm_runtime_metrics(self.llm)
            gen_event = GenerationObservability(
                peak_vram_gb=sampler.peak_gb,
                kv_cache_usage_perc=vllm_metrics["kv_cache_usage_perc"],
                prefix_cache_hit_rate=vllm_metrics["prefix_cache_hit_rate"],
                spec_accept_rate=vllm_metrics["spec_accept_rate"],
                loadavg_pre=loadavg_pre,
                loadavg_post=read_loadavg(),
                engine_runtime_config=self._engine_runtime_config,
                # ``flag_did_not_engage`` is the consuming caller's job to
                # compute. Production generation has no intended-overrides
                # dict to compare against; the benchmark harness probes the
                # engine config against its candidate's intended settings and
                # sets the bit when a knob silently fails to engage.
                flag_did_not_engage=False,
            )
            logger.runtime.info("vLLM generation complete", extra={"ctx": gen_event.model_dump()})
            # Mirror to the active wandb run when one exists (no-op when
            # ``WANDB_MODE=disabled`` or ``initialize_wandb_run`` was never
            # called). Best-effort; wandb failures are swallowed downstream.
            log_observability_event(gen_event, prefix="vllm_gen")
        except Exception as exc:  # noqa: BLE001 — observability must never break generation
            # Guard the warning itself: a faulty logger handler must not turn a
            # swallowed observability failure into a generation failure.
            with contextlib.suppress(Exception):
                logger.runtime.debug(
                    "vLLM generation observability emit failed",
                    extra={"ctx": {"error": f"{exc!r}"}},
                    exc_info=True,
                )
