# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from functools import partial
from typing import TYPE_CHECKING, Any, cast

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
from ..observability import get_logger
from ..utils import all_equal_type, load_json

logger = get_logger(__name__)

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"


class VllmBackend(GeneratorBackend):
    def __init__(self, config: SafeSynthesizerParameters, model_metadata: ModelMetadata, workdir: Workdir, **kwargs):
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
        self.llm: vLLM | None = None
        self.logits_processors = []

        # Do not generate detailed error messages in production to avoid leaking sensitive data.
        self.use_detailed_logs = kwargs.pop("use_detailed_logs", False)
        self.gen_method: partial | None = None
        self._gen_method: partial | None = None
        self.processor = create_processor(self.schema, self.model_metadata, self.config)
        adapter_path = self.workdir.adapter_path if self.workdir.adapter_path else self.model_metadata.adapter_path
        self.lora_req = LoRARequest("lora", 1, str(adapter_path)) if adapter_path else None

    def teardown(self) -> None:
        """Clear the LLM state to free up GPU memory. Unloads the model from memory and cleans up any distributed resources."""
        self._clear_llm_state()

    def _clear_llm_state(self) -> None:
        """Delete LLM state to free up GPU memory."""
        cleanup_dist_env_and_memory()
        # destroy_model_parallel()
        self.llm = None
        logger.debug("Cleaned up LLM")
        cleanup_memory()
        logger.debug("Cleaned up memory")

    def __del__(self) -> None:
        """Cleanup resources when the object is garbage collected, which prevents warnings during forced shutdowns."""
        try:
            self._clear_llm_state()
        except Exception:
            # Suppress errors during garbage collection to avoid masking other exceptions
            pass

    def initialize(self, **kwargs) -> None:
        """Initialize and load the model into memory."""
        # vLLM 0.11.x uses an environment variable for attention backend selection.
        # When vLLM is upgraded to 0.12+, migrate to the attention_backend constructor arg.
        if self.config.generation.attention_backend not in [None, "auto"]:
            os.environ["VLLM_ATTENTION_BACKEND"] = self.config.generation.attention_backend

        max_vram = get_max_vram()
        # note this only works for single GPU setups
        max_vram = max_vram.get(0, 0.8)

        # vllm requires this "config" to set the backend ahead of time.
        structured_outputs_config = StructuredOutputsConfig(
            backend=self.config.generation.structured_generation_backend,
            disable_fallback=True,
        )
        self.llm = vLLM(
            model=self.config.training.pretrained_model,
            gpu_memory_utilization=max_vram,
            enable_lora=True,
            max_lora_rank=self.config.training.lora_r,
            structured_outputs_config=structured_outputs_config,
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
                self.model_metadata.prompt_config.bos_token,
                self.model_metadata.prompt_config.eos_token,
                group_by=self.config.data.group_training_examples_by is not None,
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
                logger.info(f"updating {param} from {val}")
                param, val = action(val)
                logger.info(f"updated {param} to {val}")

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

        # Create a partially parametrized version of the underlying vllm.LLM.generate
        # method that is immediately callable downstream.
        if TYPE_CHECKING:
            assert self.llm is not None
        self._gen_method = partial(
            self.llm.generate, sampling_params=real_params, lora_request=self.lora_req, use_tqdm=False
        )

    def _generate(
        self,
        prompts: str | list[str] | None = None,
        input_ids: torch.TensorType | list[list[int]] | None = None,
        **kwargs,
    ) -> list[RequestOutput]:
        # attention_mask is unnecessary in vLLM due to continuous batching.
        # leaving in here for compatibility.

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
                    case torch.Tensor(data=ids):
                        result = self._gen_method(prompt_token_ids=ids.tolist())
                    # read the below as:
                    # if the ids passed are a list of a list of ints
                    case [*ids] if all_equal_type(ids, int):
                        result = self._gen_method(prompt_token_ids=ids)
                    case None:
                        return self._gen_method(prompts=prompts)
                    case _:
                        raise ValueError("input_ids are not a tensor, list, or None!")

                return cast(list[RequestOutput], [r.outputs[0].text for r in result])
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
        logger.debug("prompt: ", self.prompt)
        prompt_list = [self.prompt] * num_prompts_per_batch

        # `n` is the number of output sequences per prompt.
        # Subsequent processing assumes `n=1`, so we hardcode it here.
        sampling_kwargs.update({"n": 1})

        for idx, output in enumerate(self._generate(prompts=prompt_list, **sampling_kwargs)):
            logger.debug(f"output: {output.outputs[0].text}")
            batch.process(idx, output.outputs[0].text)

        return batch

    def _log_batch_timing_and_progress(
        self, batch: Batch, duration: float, num_records: int, num_valid_records: int, batches: GenerationBatches
    ) -> None:
        """Log batch timing and progress information.

        Outputs:
            - Console: Automatically rendered as Rich ASCII table by structlog processor
            - JSON logs: Structured key/value pairs for machine parsing
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
        keep_llm_state: bool = True,
        data_actions_fn: utils.DataActionsFn | None = None,
    ) -> GenerateJobResults:
        """Generate tabular data using Nemo Safe Synthesizer.

        Args:
            keep_llm_state: If True, keep the model in memory after generation. Note, this will be cleared upon garbage collection of this object.
            data_actions_fn: Optional function that takes a DataFrame and returns a modified DataFrame.

        Returns:
            Generation results object, which includes a DataFrame of generated records.
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
            ignore_eos=need_special_token_outputs,
        )
        self.prepare_params(**sampling_kwargs)

        # The batches object collects batches and keeps track of the stopping condition.
        batches = GenerationBatches(
            target_num_records=self.config.generation.num_records,
            invalid_fraction_threshold=self.config.generation.invalid_fraction_threshold,
            patience=self.config.generation.patience,
            data_actions_fn=data_actions_fn,
        )

        while batches.num_valid_records < self.config.generation.num_records:
            # Generate a batch from prompts and process the responses.
            start_time = time.perf_counter()
            batch: Batch = self._generate_batch(
                num_prompts_per_batch=batches.get_next_num_prompts(),
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

        if not keep_llm_state:
            self._clear_llm_state()

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
    """
    A wrapper to enable locally typical sampling in vllm.
    See thread: https://github.com/vllm-project/vllm/issues/1444.
    """

    def __init__(self, mass: float):
        self.warper = TypicalLogitsWarper(mass=mass)

    def __call__(self, token_ids: list[int], logits: torch.FloatTensor) -> torch.FloatTensor:
        # transformers warpers assume tensors of shape (batch_size, vocab_size)
        # and the typical warper doesn't use input_ids
        return self.warper(input_ids=None, scores=logits.reshape((1, -1)))
