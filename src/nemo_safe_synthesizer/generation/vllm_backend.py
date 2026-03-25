# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM-based generation backend for tabular data synthesis."""

import logging
import os
import time
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
from transformers import TypicalLogitsWarper
from vllm import LLM as vLLM
from vllm import RequestOutput
from vllm.config import StructuredOutputsConfig
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from .. import utils
from ..cli.artifact_structure import Workdir
from ..config import SafeSynthesizerParameters
from ..defaults import DEFAULT_SAMPLING_PARAMETERS, FIXED_RUNTIME_GENERATE_ARGS
from ..generation.backend import GeneratorBackend
from ..generation.batch import Batch
from ..generation.processors import TabularDataProcessor, create_processor
from ..generation.regex_manager import build_json_based_regex
from ..generation.results import GenerateJobResults, GenerationBatches, GenerationStatus
from ..llm.metadata import ModelMetadata
from ..llm.utils import cleanup_memory, get_max_vram
from ..observability import get_logger, heartbeat
from ..utils import all_equal_type, load_json

logger = get_logger(__name__)

if torch.cuda.is_available():
    _gpu_count = torch.cuda.device_count()
    if _gpu_count <= 1:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    else:
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
else:
    # When CUDA is unavailable, avoid triggering CUDA initialization and
    # default to disabling vLLM v1 multiprocessing.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _is_redis_available() -> bool:
    """Return True if the ``redis`` package is importable."""
    try:
        import redis  # noqa: F401 # type: ignore[unresolved-import]

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

        RemoteAutotuneCache.backend_override_cls = _NoopRemoteCacheBackend  # type: ignore[invalid-assignment]
        logger.debug("Installed no-op backend for RemoteAutotuneCache (redis unavailable)")
    except ImportError:
        pass


_install_noop_remote_cache_backends()


class VllmBackend(GeneratorBackend):
    """Generation backend using vLLM for high-throughput inference.

    Loads the base model with a LoRA adapter via vLLM and generates
    synthetic records in batches.  Supports optional structured
    generation (regex or JSON schema) to constrain outputs.

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
        self.columns = list(self.schema["properties"].keys())
        self.prompt = utils.create_schema_prompt(
            self.columns,
            instruction=self.model_metadata.instruction,
            prompt_template=self.model_metadata.prompt_config.template,
        )
        if self.model_metadata.prompt_config.add_eos_token_to_prompt:
            self.prompt += self.model_metadata.prompt_config.eos_token
        self.llm: vLLM | None = None
        self.logits_processors = []

        # Do not generate detailed error messages in production to avoid leaking sensitive data.
        self.use_detailed_logs = kwargs.pop("use_detailed_logs", False)
        self.gen_method: partial | None = None
        self._gen_method: partial | None = None
        self.processor = create_processor(self.schema, self.model_metadata, self.config)
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
            pass

    def initialize(self, **kwargs) -> None:
        """Initialize and load the model into memory."""
        self._torn_down = False

        # vLLM 0.12+ accepts attention_config as a constructor arg (replaces the
        # VLLM_ATTENTION_BACKEND env var used in 0.11.x).
        attn_backend = self.config.generation.attention_backend
        attention_config = {"backend": attn_backend} if attn_backend not in (None, "auto") else None

        max_vram = get_max_vram()
        # note this only works for single GPU setups
        max_vram = max_vram.get(0, 0.8)

        # vllm requires this "config" to set the backend ahead of time.
        structured_outputs_config = StructuredOutputsConfig(
            backend=self.config.generation.structured_generation_backend,
            disable_fallback=True,
        )
        # Unsloth patches model attention forward functions with torch.compiler.disable().
        # vLLM compiles TransformersForCausalLM with fullgraph=True via @support_torch_compile.
        # PyTorch >= 2.9.1 changed fullgraph=True to raise immediately on torch.compiler.disable()
        # rather than silently breaking the graph (pytorch#8e83e24). This combination produces:
        #   torch._dynamo.exc.Unsupported: Skip inlining `torch.compiler.disable()`d function
        # Passing enforce_eager=True skips vLLM's torch.compile pipeline entirely for these runs.
        # check this when updating unsloth in the future.
        enforce_eager = self.config.training.use_unsloth is True

        with heartbeat("Model loading", logger_name=__name__, model=self.config.training.pretrained_model):
            self.llm = vLLM(
                model=self.config.training.pretrained_model,
                gpu_memory_utilization=max_vram,
                enable_lora=True,
                max_lora_rank=self.config.training.lora_r,
                structured_outputs_config=structured_outputs_config,
                enforce_eager=enforce_eager,
                attention_config=attention_config,
            )

    def _build_structured_output_params(self) -> StructuredOutputsParams | None:
        """Build structured output parameters based on generation config.

        Returns:
            StructuredOutputsParams if structured generation is enabled, None otherwise.
        """
        if not self.config.generation.use_structured_generation:
            return None

        params: dict[str, Any] = {"disable_fallback": True}

        if self.config.generation.structured_generation_schema_method == "regex":
            logger.info("Structured generation is enabled, using a regex to enforce the schema")
            regex = build_json_based_regex(
                self.schema,
                self.config,
                self.model_metadata.prompt_config.bos_token,
                self.model_metadata.prompt_config.eos_token,
            )
            params["regex"] = regex
        elif self.config.generation.structured_generation_schema_method == "json_schema":
            params["json"] = self.schema

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
            "typical_p": lambda x: (
                "logits_processors",
                [TypicalLogitsWarperWrapper(x)],
            ),
            "temperature": lambda x: ("temperature", resolved_temperature),
            "num_beams": lambda x: ("beam_width", x) if x > 1 else (None, None),
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

        Parses a dictionary of parameters into SamplingParameters,
        applying necessary transformations from our API to vLLM's API.

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
        if TYPE_CHECKING:
            assert self.llm is not None
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
                        logger.debug("vllm generate: prompt_token_ids (torch.Tensor)")
                        result = self._gen_method(prompt_token_ids=input_ids.tolist())
                    case [[*_inner], *_] if all_equal_type(input_ids, int):
                        assert isinstance(input_ids, list)
                        logger.debug(f"vllm generate: prompt_token_ids ({len(input_ids)} prompts)")
                        result = self._gen_method(prompt_token_ids=input_ids)
                    case [*ids] if all_equal_type(ids, int, flatten_iter=False):
                        logger.debug("vllm generate: prompt_token_ids (single flat list)")
                        result = self._gen_method(prompt_token_ids=[ids])
                    case None:
                        logger.debug(
                            f"vllm generate: processing {len(prompts) if isinstance(prompts, list) else 1} prompts"
                        )
                        return self._gen_method(prompts=prompts)
                    case _:
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
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"prompt {idx}: {len(out.token_ids)} tokens, "
                    f"finish_reason={out.finish_reason}, "
                    f"stop_reason={out.stop_reason}"
                )
            batch.process(idx, out.text)

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
        progress_data = {
            "records_per_second": round(records_per_second, 2),
            "duration_seconds": round(duration, 2),
            "valid_records_generated": batches.num_valid_records,
            "target_records": self.config.generation.num_records,
            "progress_fraction": round(batches.num_valid_records / self.config.generation.num_records, 4),
        }

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

        Args:
            data_actions_fn: Optional post-processing / validation function
                applied to each batch of generated records.

        Returns:
            Results containing the generated DataFrame and statistics.
        """
        generation_start = time.monotonic()

        need_special_token_outputs = not isinstance(self.processor, TabularDataProcessor)
        sampling_kwargs = dict(
            temperature=self.config.generation.temperature,
            repetition_penalty=self.config.generation.repetition_penalty,
            top_p=self.config.generation.top_p,
            top_k=FIXED_RUNTIME_GENERATE_ARGS["top_k"],
            min_p=FIXED_RUNTIME_GENERATE_ARGS["min_p"],
            logits_processors=self.logits_processors,
            max_tokens=self.model_metadata.max_seq_length,
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

        generation_time_sec = time.monotonic() - generation_start
        self.elapsed_time = generation_time_sec
        self.gen_results = GenerateJobResults.from_batches(
            batches=batches,
            columns=self.columns,
            max_num_records=max_num_records,
            elapsed_time=self.elapsed_time,
        )

        return self.gen_results


class TypicalLogitsWarperWrapper:
    """Adapter enabling locally typical sampling in vLLM.

    Wraps the HuggingFace ``TypicalLogitsWarper`` to match the vLLM
    logits-processor signature.  See
    `vllm#1444 <https://github.com/vllm-project/vllm/issues/1444>`_.

    Args:
        mass: Probability mass for typical sampling.
    """

    def __init__(self, mass: float):
        self.warper = TypicalLogitsWarper(mass=mass)

    def __call__(self, token_ids: list[int], logits: torch.FloatTensor) -> torch.FloatTensor:
        return self.warper(input_ids=None, scores=logits.reshape((1, -1)))
