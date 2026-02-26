# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
This module contains the ModelMetadata class, which is used to store model-family-specific
information - prompt formats, and our program runtime information. the most problematic
part of this module is the rope_scaling logic, which is used to scale the context window
of the model based on the tokens in the dataset.

We should probably change how users specify this up front in the parameters, using
something like "context_window_size" or "max_context_window_size".  and determine if we
should use rope scaling or not from that, the model, and the data itself.

Currently we're using the global max sequence length of 2048 * 6 to prevent OOM errors
and underfitting errors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator
from pydantic.experimental.missing_sentinel import MISSING
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig

from ..cli.artifact_structure import Workdir
from ..config.parameters import SafeSynthesizerParameters
from ..defaults import (
    DEFAULT_INSTRUCTION,
    MAX_ROPE_SCALING_FACTOR,
    PROMPT_TEMPLATE,
)
from ..observability import get_logger
from ..utils import load_json, write_json

logger = get_logger(__name__)

DEFAULT_MAX_SEQ_LENGTH = 2048
GLOBAL_MAX_SEQ_LENGTH = 2048 * 6


class LLMPromptConfig(BaseModel):
    template: str
    add_bos_token_to_prompt: bool
    add_eos_token_to_prompt: bool

    bos_token: str
    bos_token_id: int
    eos_token: str
    eos_token_id: int

    @classmethod
    def from_tokenizer(cls, name: str, tokenizer: AutoTokenizer | None = None, **kwargs) -> LLMPromptConfig:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(name)
        bos_token = kwargs.get("bos_token", getattr(tokenizer, "bos_token", None))
        bos_token_id = kwargs.get("bos_token_id", getattr(tokenizer, "bos_token_id", None))
        eos_token = kwargs.get("eos_token", getattr(tokenizer, "eos_token", None))
        eos_token_id = kwargs.get("eos_token_id", getattr(tokenizer, "eos_token_id", None))
        template = kwargs.get("template", PROMPT_TEMPLATE)
        add_bos_token_to_prompt = kwargs.get("add_bos_token_to_prompt", True)
        add_eos_token_to_prompt = kwargs.get("add_eos_token_to_prompt", True)

        pc = {
            "template": template,
            "add_bos_token_to_prompt": add_bos_token_to_prompt,
            "add_eos_token_to_prompt": add_eos_token_to_prompt,
            "bos_token": bos_token,
            "bos_token_id": bos_token_id,
            "eos_token": eos_token,
            "eos_token_id": eos_token_id,
        }

        return cls(**pc)


def resolve_rope_scaling_factor(
    factor: float | int | RopeScaling | dict | None = None, autoconfig: PretrainedConfig | None = None
) -> RopeScaling | None:
    """Resolve the rope scaling factor from a variety of input types."""
    match factor, autoconfig:
        case None | 1 | 1.0, _:
            return None
        case RopeScaling() as r, _:
            return r
        case dict() as d, _:
            return RopeScaling(**d)
        case int(x) | float(x), PretrainedConfig() as c:
            return RopeScaling.from_autoconfig(config=c, factor=x)
        case int(x) | float(x), None:
            raise ValueError("autoconfig is required when factor is an int or float")
        case _, None:
            raise ValueError("autoconfig is required when factor is not a RopeScaling, dict, or int/float")
        case _, _:
            raise ValueError("Invalid input type for rope scaling factor")


class RopeScaling(BaseModel):
    """Parameters for rope scaling.

    Replace this with `from .modeling_rope_utils import RotaryEmbeddingConfigMixin`
    when that's ready in transformers v5 or something similar.
    """

    rope_type: Annotated[
        Literal["linear", "dynamic", "default", "yarn", "llama3"],
        Field(description="Type of rope scaling"),
    ] = "default"
    factor: Annotated[float, Field(description="Multiplier for rope scaling to extend context window")] = 1.0
    theta: Annotated[float, Field(description="Theta for rope scaling")] = 10000.0

    @field_validator("factor", mode="after")
    @classmethod
    def validate_factor(cls, v: float | int | None) -> float | int | None:
        if v is None or v <= MAX_ROPE_SCALING_FACTOR:
            return v
        logger.warning(
            f"Rope scaling factor {v} is greater than MAX_ROPE_SCALING_FACTOR: {MAX_ROPE_SCALING_FACTOR}, setting to {MAX_ROPE_SCALING_FACTOR}"
        )
        return MAX_ROPE_SCALING_FACTOR

    @classmethod
    def from_autoconfig(cls, config: PretrainedConfig, factor: float | int | None = None) -> "RopeScaling":
        """Create RopeScaling from a HuggingFace AutoConfig.

        Reads the model's native rope configuration and optionally applies a scaling factor.
        """
        # Try to get theta from config (different models use different attribute names)
        theta = getattr(config, "rope_theta", None) or 10000.0

        # Try to get rope_type from config
        rope_type = getattr(config, "rope_scaling", {})
        if isinstance(rope_type, dict):
            rope_type = rope_type.get("rope_type", "default")
        else:
            rope_type = "default"

        return cls(
            rope_type=rope_type,
            factor=factor or 1.0,
            theta=theta,
        )

    def to_hf_dict(self) -> dict | None:
        """
        Convert to HuggingFace rope_scaling dict format.

        This is used to set the rope_scaling parameter on the HuggingFace model config. Use the new `RotaryEmbeddingConfigMixin`
        or RopeParameters from HF when that's ready in transformers v5 or something similar.
        """
        """Convert to HuggingFace rope_scaling dict format."""
        if self.factor == 1.0:
            return None
        return {
            "rope_type": self.rope_type,
            "factor": self.factor,
            "theta": self.theta,
        }


class ModelMetadata(BaseModel):
    """
    Container to hold model-family-specific information - prompt formats,
    tokens, etc.
    """

    # Learning rate when training.learning_rate is "auto". Override in subclasses (e.g. Mistral).
    default_learning_rate: ClassVar[float] = 0.0005

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name_or_path: str
    prompt_config: LLMPromptConfig
    autoconfig: Annotated[PretrainedConfig, Field(description="PretrainedConfig object for the model", exclude=True)]
    base_max_seq_length: Annotated[
        int | None,
        Field(description="Supported context window for base model, before rope scaling factor adjustment"),
    ] = None
    rope_scaling: Annotated[
        RopeScaling | MISSING | None,  # type: ignore[invalid-type-form]
        Field(
            description="RoPE scaling configuration for context window extension. will be auto-populated if not provided. if an integer is provided, it will be used as the factor and theta will be set to 10000.0",
        ),
    ] = MISSING
    max_sequences_per_example: Annotated[
        int | None,
        Field(description="Maximum number of sequences per training example."),
    ] = None
    workdir: Workdir | None = None
    is_adapter: bool = False
    instruction: str = DEFAULT_INSTRUCTION
    rope_parameters_location: Literal["autoconfig", "automodel"] = "automodel"
    initial_prefill: dict[str, str] | str | None = None

    @model_validator(mode="before")
    @classmethod
    def populate_derived_fields(cls, data: dict) -> dict:
        """Populate autoconfig, rope_scaling, and base_max_seq_length if not provided."""
        if data.get("autoconfig") is None:
            data["autoconfig"] = AutoConfig.from_pretrained(data["model_name_or_path"])

        if data.get("base_max_seq_length") is None:
            data["base_max_seq_length"] = get_base_max_seq_length(data["autoconfig"])

        rsf = data.get("rope_scaling")
        data["rope_scaling"] = resolve_rope_scaling_factor(rsf, data["autoconfig"])

        return data

    @field_serializer("autoconfig")
    def serialize_autoconfig(self, config: PretrainedConfig) -> dict:
        """Serialize PretrainedConfig to dict for JSON export."""
        return config.to_dict()

    @property
    def adapter_path(self) -> Path:
        """Get the path where adapter model files are stored.

        Raises:
            ValueError: If workdir is not set.
        """
        if self.workdir is None:
            raise ValueError("Cannot get adapter_path: workdir is not set")
        return self.workdir.train.adapter.path.resolve()

    @property
    def metadata_path(self) -> Path:
        """Get the path to the metadata JSON file.

        Uses workdir.metadata_file which automatically resolves to the parent
        workdir's path when resuming for generation.

        Raises:
            ValueError: If workdir is not set.
        """
        if self.workdir is None:
            raise ValueError("Cannot get metadata_path: workdir is not set")
        return self.workdir.metadata_file

    @property
    def rope_scaling_factor(self) -> float:
        """Get the rope scaling factor for backwards compatibility."""
        return self.rope_scaling.factor if self.rope_scaling is not None else 1.0

    @property
    def max_seq_length(self) -> int:
        """Actual context window for training.

        Includes any adjustment for rope_scaling.factor.
        """
        rsf = 1.0
        if isinstance(self.rope_scaling, RopeScaling) and self.rope_scaling.factor > 1.0:
            rsf = self.rope_scaling.factor
        return int(self.base_max_seq_length * rsf)

    def save_metadata(self) -> None:
        """Save model metadata to JSON file.

        Raises:
            ValueError: If workdir is not set.
        """
        if self.workdir is None:
            raise ValueError("Cannot save metadata: workdir is not set")
        write_json(self.model_dump(mode="json"), path=self.workdir.train.adapter.metadata, indent=4)

    @classmethod
    def from_str_or_path(cls: type["ModelMetadata"], model_name_or_path: Path | str, **kwargs) -> ModelMetadata:
        classes = TinyLlama, Qwen, Llama32, SmolLM2, SmolLM3, Mistral, Nemotron, Granite
        for class_ in classes:
            if str(class_.__name__).lower() in str(model_name_or_path).lower():
                return class_(model_name_or_path=str(model_name_or_path), **kwargs)
        raise ValueError(f"Unknown model name or path: {model_name_or_path}")

    @classmethod
    def from_config(
        cls: type["ModelMetadata"],
        config: SafeSynthesizerParameters,
        workdir: Workdir | None = None,
    ) -> ModelMetadata:
        """Create ModelMetadata from SafeSynthesizerParameters.

        The config should have been resolved with AutoConfigResolver before this is called.

        Args:
            config: SafeSynthesizerParameters with model and training configuration.
            workdir: Workdir instance for artifact paths. Required for saving model artifacts.

        If rope_scaling_factor is set in config, it will be used to create a RopeScaling
        object with the model's native theta value.

        If max_sequences_per_example is set in config.data, it will be passed through
        to control how many sequences are packed per training example. This is critical
        for differential privacy where it must be 1.
        """
        kwargs: dict = {"workdir": workdir}

        if config.training.rope_scaling_factor is not None and config.training.rope_scaling_factor != "auto":
            # Pass the factor; the subclass will create the RopeScaling with proper theta
            kwargs["rope_scaling_factor"] = config.training.rope_scaling_factor

        # Pass max_sequences_per_example from data config - critical for DP training
        kwargs["max_sequences_per_example"] = config.data.max_sequences_per_example

        return ModelMetadata.from_str_or_path(config.training.pretrained_model, **kwargs)

    @classmethod
    def from_metadata_json(
        cls: type["ModelMetadata"],
        path: Path | str,
        workdir: Workdir | None = None,
    ) -> ModelMetadata:
        """Load ModelMetadata from a saved JSON file.

        Args:
            path: Path to the metadata JSON file.
            workdir: Workdir instance for artifact paths. If not provided, will be None.

        Returns:
            ModelMetadata instance with the loaded configuration.
        """
        path = Path(path).resolve()
        kwargs = load_json(path)
        if workdir is not None:
            kwargs["workdir"] = workdir
        return cls(**kwargs)


def get_base_max_seq_length(config: AutoConfig) -> int:
    """
    Get the base max sequence length for the model before rope scaling.
    In the future, we will use a more dynamic approach based on available VRAM and the tokens in your dataset.
    For now we have a global max for
    """
    if mpe := getattr(config, "max_position_embeddings", None):
        logger.info(f"Using max_position_embeddings from config: {mpe}")
        if mpe > GLOBAL_MAX_SEQ_LENGTH:
            msg = f"max_position_embeddings is greater than GLOBAL_MAX_SEQ_LENGTH: {mpe} > {GLOBAL_MAX_SEQ_LENGTH}"
            msg += "\n This is a temporary workaround to prevent OOM and underfitting errors"
            msg += "\n In the future, we will use a more dyanmic approach based on available VRAM and the tokens in your dataset."
            logger.warning(msg)
        return min(mpe, GLOBAL_MAX_SEQ_LENGTH)
    logger.info(f"Using default max_position_embeddings: {DEFAULT_MAX_SEQ_LENGTH}")
    return DEFAULT_MAX_SEQ_LENGTH


# idea: make these classes subclasses of the huggingface class, like `LLamaModel`, and add
# our ModelMetadata as the additional property.
# that way we can use the huggingface class's methods, like `from_pretrained`, `from_config`, etc.
# but this might not work bc of multiple inheritance with torch.nn.Module
class Granite(ModelMetadata):
    def __init__(
        self, model_name_or_path: str, tokenizer=None, rope_scaling_factor: float | None = None, **kwargs
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                name=model_name_or_path,
                template="user\n {instruction} {schema} \n assistant\n{prefill}",
                add_bos_token_to_prompt=False,
                add_eos_token_to_prompt=True,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=rope_scaling_factor,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class Llama32(ModelMetadata):
    def __init__(
        self, model_name_or_path: str, tokenizer=None, rope_scaling_factor: float | None = None, **kwargs
    ) -> None:
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                name=model_name_or_path,
                template="user\n {instruction} {schema} \n assistant\n{prefill}",
                bos_token="<|im_start|>",
                bos_token_id=151644,
                add_bos_token_to_prompt=False,
                add_eos_token_to_prompt=False,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=rope_scaling_factor,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class Mistral(ModelMetadata):
    """Mistral family. Uses lower default learning rate (0.0001) when training.learning_rate is auto."""

    default_learning_rate: ClassVar[float] = 0.0001

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: AutoTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
        config.learning_rate = Mistral.default_learning_rate
        if rope_scaling_factor:
            logger.warning(
                f"Rope scaling factor {rope_scaling_factor} is not supported for Mistral due to longer default context lengths. Ignoring."
            )

        template = "[INST] {instruction} \n\n {schema} [/INST]{prefill}"
        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                name=model_name_or_path,
                template=template,
                add_bos_token_to_prompt=True,
                add_eos_token_to_prompt=True,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=None,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class Nemotron(ModelMetadata):
    def __init__(
        self, model_name_or_path: str, tokenizer=None, rope_scaling_factor: float | None = None, **kwargs
    ) -> None:
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                template="[INST] {instruction} \n\n {schema} [/INST]{prefill}",
                add_bos_token_to_prompt=True,
                add_eos_token_to_prompt=True,
                tokenizer=tokenizer,
                name=model_name_or_path,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=rope_scaling_factor,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class Qwen(ModelMetadata):
    def __init__(
        self, model_name_or_path: str, tokenizer=None, rope_scaling_factor: float | None = None, **kwargs
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config = AutoConfig.from_pretrained(model_name_or_path)

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            # Matched with vllm prompt 2024-12-18
            prompt_config=LLMPromptConfig.from_tokenizer(
                template="user\n {instruction} {schema} \n assistant\n{prefill}",
                add_bos_token_to_prompt=True,
                add_eos_token_to_prompt=False,
                tokenizer=tokenizer,
                name=model_name_or_path,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=rope_scaling_factor,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class SmolLM2(ModelMetadata):
    """SmolLM2 models (e.g., HuggingFaceTB/SmolLM2-135M).
    Potentially used for testing."""

    def __init__(
        self, model_name_or_path: str, tokenizer=None, rope_scaling_factor: float | None = None, **kwargs
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config = AutoConfig.from_pretrained(model_name_or_path)
        if rope_scaling_factor:
            logger.warning(
                f"Rope scaling factor {rope_scaling_factor} is not supported for Mistral due to longer default context lengths. Ignoring."
            )

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                template="user\n {instruction} {schema} \n assistant\n{prefill}",
                add_bos_token_to_prompt=False,
                add_eos_token_to_prompt=False,
                tokenizer=tokenizer,
                bos_token="<|im_start|>",
                bos_token_id=151644,
                name=model_name_or_path,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=None,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class SmolLM3(ModelMetadata):
    def __init__(
        self, model_name_or_path: str, tokenizer=None, rope_scaling_factor: float | None = None, **kwargs
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config = AutoConfig.from_pretrained(model_name_or_path)

        # we use the bos token here explicitly for support during group-by SFT.
        # the groupby assumes there is a bos token at the start of the prompt.
        bos_token = "<|im_start|>"
        bos_token_id = 128011

        # SmolLM3 uses high theta values (1.5M-5M) so it's important to read from config
        if rope_scaling_factor:
            logger.warning(
                f"Rope scaling factor {rope_scaling_factor} is not supported for SmolLM3 due to longer default context lengths. Ignoring."
            )

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                template="user\n {instruction} {schema} <|im_end|> \n <|im_start|>assistant\n{prefill}",
                add_bos_token_to_prompt=True,
                add_eos_token_to_prompt=True,
                tokenizer=tokenizer,
                name=model_name_or_path,
                bos_token=bos_token,
                bos_token_id=bos_token_id,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=None,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class TinyLlama(ModelMetadata):
    def __init__(
        self, model_name_or_path: str, tokenizer=None, rope_scaling_factor: float | None = None, **kwargs
    ) -> None:
        tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name_or_path)
        config = AutoConfig.from_pretrained(model_name_or_path)

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                template=PROMPT_TEMPLATE,
                add_bos_token_to_prompt=True,
                add_eos_token_to_prompt=True,
                tokenizer=tokenizer,
                name=model_name_or_path,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=rope_scaling_factor,
            rope_parameters_location="autoconfig",
            **kwargs,
        )
