# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-family metadata for prompt formatting, RoPE scaling, and runtime bookkeeping.

Provides ``ModelMetadata`` and its per-family subclasses (``Llama32``,
``Mistral``, ``Qwen``, etc.) that capture prompt templates, special-token
settings, and context-window configuration.  The ``RopeScaling`` model
handles context-window extension via Rotary Position Embeddings.

A global maximum sequence length (``GLOBAL_MAX_SEQ_LENGTH = 2048 * 6``)
is applied as a safety cap to prevent OOM and underfitting errors.

Classes:
    LLMPromptConfig: Prompt template and special-token settings.
    RopeScaling: RoPE scaling parameters for context-window extension.
    ModelMetadata: Base container for model-family-specific metadata.
    Granite: IBM Granite family metadata.
    Llama32: Meta Llama 3.2 family metadata.
    Mistral: Mistral AI family metadata.
    Nemotron: NVIDIA Nemotron family metadata.
    Qwen: Alibaba Qwen family metadata.
    SmolLM2: HuggingFace SmolLM2 family metadata.
    SmolLM3: HuggingFace SmolLM3 family metadata.
    TinyLlama: TinyLlama family metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer

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
    """Prompt template and special-token settings for an LLM.

    Holds the Jinja-style prompt ``template`` together with flags and
    token values that control how BOS/EOS markers are injected during
    training and inference.
    """

    template: str
    """Prompt template with ``{instruction}``, ``{schema}``, and ``{prefill}`` placeholders.

    * ``{instruction}`` -- task directive telling the model what to generate
      (e.g. "Generate a JSONL dataset with the following columns: ").
    * ``{schema}`` -- column schema fragment listing expected output fields,
      typically formatted as ``"col":<unk>,"col2":<unk>``.
    * ``{prefill}`` -- optional text injected at the start of the model's
      response to steer generation, currently used for time series data.
    """

    add_bos_token_to_prompt: bool
    """Whether to prepend the BOS token to the prompt."""

    add_eos_token_to_prompt: bool
    """Whether to append the EOS token to the prompt."""

    bos_token: str
    """Beginning-of-sequence token string."""

    bos_token_id: int
    """Integer id for the BOS token."""

    eos_token: str
    """End-of-sequence token string."""

    eos_token_id: int
    """Integer id for the EOS token."""

    @classmethod
    def from_tokenizer(cls, name: str, tokenizer: PreTrainedTokenizer | None = None, **kwargs) -> LLMPromptConfig:
        """Create a prompt config by reading from settings of a tokenizer.

        If no ``tokenizer`` is supplied one is loaded from ``name``
        via ``AutoTokenizer.from_pretrained``.  Individual fields can
        be overridden through ``**kwargs`` (e.g. ``bos_token``,
        ``template``).

        Args:
            name: HuggingFace model identifier used to load the
                tokenizer when ``tokenizer`` is ``None``.
            tokenizer: Optional pre-loaded tokenizer instance.
            **kwargs: Overrides for any ``LLMPromptConfig`` field.

        Returns:
            A new ``LLMPromptConfig`` populated from the tokenizer.
        """
        _tokenizer: PreTrainedTokenizer = cast(
            PreTrainedTokenizer, AutoTokenizer.from_pretrained(name) if tokenizer is None else tokenizer
        )
        bos_token = kwargs.get("bos_token", getattr(_tokenizer, "bos_token", None))
        bos_token_id = kwargs.get("bos_token_id", getattr(_tokenizer, "bos_token_id", None))
        eos_token = kwargs.get("eos_token", getattr(_tokenizer, "eos_token", None))
        eos_token_id = kwargs.get("eos_token_id", getattr(_tokenizer, "eos_token_id", None))
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
    factor: float | int | RopeScaling | dict | None = None,
    autoconfig: PretrainedConfig | None = None,
) -> RopeScaling | None:
    """Normalize a rope-scaling specification into a ``RopeScaling`` or ``None``.

    Accepts several convenience representations and converts them into a
    canonical ``RopeScaling`` instance.

    Args:
        factor: The scaling specification.  Accepted forms:

            * ``None``, ``1``, or ``1.0`` — no scaling (returns ``None``).
            * ``RopeScaling`` — returned as-is.
            * ``dict`` — unpacked as ``RopeScaling(**factor)``.
            * ``int`` / ``float`` — used as the scaling factor; requires
              ``autoconfig`` to read ``rope_theta`` and ``rope_type``.
        autoconfig: A HuggingFace ``PretrainedConfig``.  Required when
            ``factor`` is a bare numeric value.

    Returns:
        A ``RopeScaling`` instance, or ``None`` when no scaling is needed.

    Raises:
        ValueError: If a numeric ``factor`` is given without
            ``autoconfig``, or if the input type is unsupported.
    """
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
    """Rotary Position Embedding (RoPE) scaling configuration.

    Encapsulates the parameters needed to extend a model's context
    window via RoPE scaling.  Will be superseded by
    ``RotaryEmbeddingConfigMixin`` when available in transformers v5.
    """

    rope_type: Literal["linear", "dynamic", "default", "yarn", "llama3"] = Field(
        default="default",
        description="Scaling algorithm: linear, dynamic, default, yarn, or llama3.",
    )

    factor: float = Field(
        default=1.0,
        description="Multiplier for RoPE scaling to extend the context window; values above MAX_ROPE_SCALING_FACTOR are clamped.",
    )

    theta: float = Field(default=10000.0, description="Theta for rope scaling.")

    @field_validator("factor", mode="after")
    @classmethod
    def validate_factor(cls, v: float | int | None) -> float | int | None:
        """Clamp ``factor`` to ``MAX_ROPE_SCALING_FACTOR`` and warn if exceeded."""
        if v is None or v <= MAX_ROPE_SCALING_FACTOR:
            return v
        logger.warning(
            f"Rope scaling factor {v} is greater than MAX_ROPE_SCALING_FACTOR: {MAX_ROPE_SCALING_FACTOR}, setting to {MAX_ROPE_SCALING_FACTOR}"
        )
        return MAX_ROPE_SCALING_FACTOR

    @classmethod
    def from_autoconfig(cls, config: PretrainedConfig, factor: float | int | None = None) -> "RopeScaling":
        """Create a ``RopeScaling`` from a HuggingFace ``PretrainedConfig``.

        Reads the model's native ``rope_theta`` and ``rope_type`` and
        optionally overrides the scaling ``factor``.

        Args:
            config: A loaded HuggingFace model config.
            factor: Scaling factor override.  Defaults to ``1.0``.

        Returns:
            A ``RopeScaling`` populated from the config.
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
        """Convert to the HuggingFace ``rope_scaling`` dict format.

        Returns ``None`` when ``factor`` is ``1.0`` (no scaling).

        Returns:
            A dict with keys ``rope_type``, ``factor``, and ``theta``,
            or ``None``.
        """
        if self.factor == 1.0:
            return None
        return {
            "rope_type": self.rope_type,
            "factor": self.factor,
            "theta": self.theta,
        }


class ModelMetadata(BaseModel):
    """Base container for model-family-specific metadata.

    Stores prompt formats, special tokens, RoPE scaling parameters, and
    runtime bookkeeping needed to load, fine-tune, and generate with a
    given LLM family.  Each supported model family has a concrete
    subclass (e.g. ``Llama32``, ``Mistral``) that sets the correct
    defaults.

    Use the factory methods [`from_str_or_path`][nemo_safe_synthesizer.llm.metadata.ModelMetadata.from_str_or_path],
    [`from_config`][nemo_safe_synthesizer.llm.metadata.ModelMetadata.from_config],
    or [`from_metadata_json`][nemo_safe_synthesizer.llm.metadata.ModelMetadata.from_metadata_json]
    to construct instances rather than calling the constructor directly.
    """

    # Learning rate when training.learning_rate is "auto". Override in subclasses.
    default_learning_rate: ClassVar[float] = 0.0005

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name_or_path: str = Field(description="HuggingFace model identifier or local path.")

    prompt_config: LLMPromptConfig = Field(description="Prompt template and token settings.")

    autoconfig: PretrainedConfig = Field(description="PretrainedConfig object for the model.", exclude=True)
    """HuggingFace ``PretrainedConfig`` (excluded from serialization)."""

    base_max_seq_length: int | None = Field(
        default=None,
        description="Supported context window for base model, before rope scaling factor adjustment.",
    )

    rope_scaling: RopeScaling | None = Field(
        default=None,
        description=(
            "RoPE scaling configuration for context window extension. "
            "Accepts a RopeScaling instance, a dict of RopeScaling fields, "
            "a numeric scale factor (requires autoconfig), or None."
        ),
    )

    max_sequences_per_example: int | None = Field(
        default=None, description="Maximum number of sequences per training example."
    )
    """Cap on sequences packed into one training example.

    Resolved by ``AutoConfigResolver`` to ``1`` when DP is enabled,
    ``10`` when DP is disabled and set to ``"auto"``, or a
    user-supplied integer.
    """

    workdir: Workdir | None = Field(default=None, description="Artifact directory layout.")

    is_adapter: bool = Field(default=False, description="Whether an adapter checkpoint is loaded.")

    instruction: str = Field(default=DEFAULT_INSTRUCTION, description="Default system instruction text.")

    rope_parameters_location: Literal["autoconfig", "automodel"] = Field(
        default="automodel",
        description="Where to read RoPE parameters from: autoconfig or automodel.",
    )

    initial_prefill: dict[str, str] | str | None = Field(
        default=None, description="Optional prefill text for generation."
    )
    """Currently used for time series data. May be a single string or a per-column dict."""

    @model_validator(mode="before")
    @classmethod
    def populate_derived_fields(cls, data: dict) -> dict:
        """Auto-populate ``autoconfig``, ``rope_scaling``, and ``base_max_seq_length``.

        Called by Pydantic before field validation.  Loads an
        ``AutoConfig`` from ``model_name_or_path`` when one is not
        already present, derives ``base_max_seq_length`` from that
        config, and resolves the ``rope_scaling`` specification into a
        ``RopeScaling`` instance (or ``None``).

        Args:
            data: Raw field values dict supplied to the constructor.

        Returns:
            The mutated ``data`` dict with derived fields populated.
        """
        if data.get("autoconfig") is None:
            data["autoconfig"] = AutoConfig.from_pretrained(data["model_name_or_path"])

        if data.get("base_max_seq_length") is None:
            data["base_max_seq_length"] = get_base_max_seq_length(data["autoconfig"])

        rsf = data.get("rope_scaling")
        data["rope_scaling"] = resolve_rope_scaling_factor(rsf, data["autoconfig"])

        return data

    @field_serializer("autoconfig")
    def serialize_autoconfig(self, config: PretrainedConfig) -> dict:
        """Serialize ``PretrainedConfig`` to a plain dict for JSON export.

        Args:
            config: The HuggingFace config to serialize.

        Returns:
            Dict representation of the config.
        """
        return config.to_dict()

    @property
    def adapter_path(self) -> Path:
        """The path where adapter model files are stored.

        Raises:
            ValueError: If workdir is not set.
        """
        if self.workdir is None:
            raise ValueError("Cannot get adapter_path: workdir is not set")
        return self.workdir.train.adapter.path.resolve()

    @property
    def metadata_path(self) -> Path:
        """The path to the metadata JSON file.

        Uses ``workdir.metadata_file`` which automatically resolves to the
        parent workdir's path when resuming for generation.

        Raises:
            ValueError: If workdir is not set.
        """
        if self.workdir is None:
            raise ValueError("Cannot get metadata_path: workdir is not set")
        return self.workdir.metadata_file

    @property
    def rope_scaling_factor(self) -> float:
        """The rope scaling factor for backwards compatibility."""
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
        write_json(
            self.model_dump(mode="json"),
            path=self.workdir.train.adapter.metadata,
            indent=4,
        )

    @classmethod
    def _resolve_model_class(cls: type["ModelMetadata"], model_name_or_path: Path | str) -> type["ModelMetadata"]:
        """Resolve model name or path to the matching ``ModelMetadata`` subclass (no instantiation)."""
        classes = TinyLlama, Qwen, Llama32, SmolLM2, SmolLM3, Mistral, Nemotron, Granite
        for class_ in classes:
            if class_.__name__.lower() in str(model_name_or_path).lower():
                return class_
        raise ValueError(f"Unknown model name or path: {model_name_or_path}")

    @classmethod
    def from_str_or_path(cls: type["ModelMetadata"], model_name_or_path: Path | str, **kwargs) -> ModelMetadata:
        """Instantiate the correct ``ModelMetadata`` subclass from a model name or path.

        Performs case-insensitive substring matching of each registered
        subclass name against ``model_name_or_path``.

        Args:
            model_name_or_path: HuggingFace model identifier or local
                filesystem path.
            **kwargs: Forwarded to the matched subclass constructor.

        Returns:
            An instance of the matched ``ModelMetadata`` subclass.

        Raises:
            ValueError: If no registered subclass matches.
        """
        return cls._resolve_model_class(model_name_or_path)(model_name_or_path=str(model_name_or_path), **kwargs)

    @classmethod
    def from_config(
        cls: type["ModelMetadata"],
        config: SafeSynthesizerParameters,
        workdir: Workdir | None = None,
    ) -> ModelMetadata:
        """Create ``ModelMetadata`` from ``SafeSynthesizerParameters``.

        The *config* should have been resolved with
        ``AutoConfigResolver`` before calling this method.

        If ``rope_scaling_factor`` is set, a ``RopeScaling`` object is
        created with the model's native theta.
        ``max_sequences_per_example`` is always forwarded from
        ``config.data`` -- ``AutoConfigResolver`` resolves it to ``1``
        when DP is enabled, ``10`` when set to ``"auto"`` with DP
        disabled, or the user-supplied integer.

        Args:
            config: Resolved parameters with model and training
                configuration.
            workdir: Artifact directory layout.  Required for saving
                model artifacts.

        Returns:
            A ``ModelMetadata`` subclass instance matching the
            configured pretrained model.
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
    """Derive the base max sequence length from a model config.

    Reads ``max_position_embeddings`` from the config and clamps it to
    ``GLOBAL_MAX_SEQ_LENGTH`` to prevent OOM and underfitting errors.
    Falls back to ``DEFAULT_MAX_SEQ_LENGTH`` when the attribute is
    absent.

    Args:
        config: A HuggingFace ``AutoConfig`` for the model.

    Returns:
        The effective base sequence length (before RoPE scaling).
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


class Granite(ModelMetadata):
    """Metadata for IBM Granite model family.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Optional RoPE scaling factor.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                name=model_name_or_path,
                tokenizer=tokenizer,
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
    """Metadata for Meta Llama 3.2 model family.

    Uses ``<|im_start|>`` as the BOS token and disables automatic
    BOS/EOS injection in prompts.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Optional RoPE scaling factor.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer

        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                name=model_name_or_path,
                tokenizer=tokenizer,
                template="user\n {instruction} {schema} \n assistant\n{prefill}",
                bos_token="<|im_start|>",
                bos_token_id=im_start_id,
                add_bos_token_to_prompt=False,
                add_eos_token_to_prompt=False,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=rope_scaling_factor,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class Mistral(ModelMetadata):
    """Metadata for Mistral AI model family.

    RoPE scaling is not supported for Mistral models. Any supplied
    ``rope_scaling_factor`` will be ignored with a warning.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Ignored with a warning if provided.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    default_learning_rate: ClassVar[float] = 0.0001

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config: PretrainedConfig = AutoConfig.from_pretrained(model_name_or_path)
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
                tokenizer=tokenizer,
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
    """Metadata for NVIDIA Nemotron model family.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Optional RoPE scaling factor.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
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
    """Metadata for Alibaba Qwen model family.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Optional RoPE scaling factor.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
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
    """Metadata for HuggingFace SmolLM2 model family (e.g. ``SmolLM2-135M``).

    RoPE scaling is not supported and any supplied ``rope_scaling_factor``
    will be ignored with a warning.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Ignored with a warning if provided.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config = AutoConfig.from_pretrained(model_name_or_path)
        if rope_scaling_factor:
            logger.warning(
                f"Rope scaling factor {rope_scaling_factor} is not supported for SmolLM2 due to longer default context lengths. Ignoring."
            )

        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                template="user\n {instruction} {schema} \n assistant\n{prefill}",
                add_bos_token_to_prompt=False,
                add_eos_token_to_prompt=False,
                tokenizer=tokenizer,
                bos_token="<|im_start|>",
                bos_token_id=im_start_id,
                name=model_name_or_path,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=None,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class SmolLM3(ModelMetadata):
    """Metadata for HuggingFace SmolLM3 model family.

    Uses ``<|im_start|>`` as the BOS token.  RoPE scaling is not
    supported. Any supplied ``rope_scaling_factor`` will be ignored
    with a warning.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Ignored with a warning if provided.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path) if tokenizer is None else tokenizer
        config = AutoConfig.from_pretrained(model_name_or_path)

        # Group-by SFT assumes a BOS token at the start of the prompt.
        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

        if rope_scaling_factor:
            logger.warning(
                f"Rope scaling factor {rope_scaling_factor} is not supported for SmolLM3 due to longer default context lengths. Ignoring."
            )

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                template="user\n {instruction} {schema} <|im_end|> \n assistant\n{prefill}",
                add_bos_token_to_prompt=True,
                add_eos_token_to_prompt=False,
                tokenizer=tokenizer,
                name=model_name_or_path,
                bos_token="<|im_start|>",
                bos_token_id=im_start_id,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=None,
            rope_parameters_location="autoconfig",
            **kwargs,
        )


class TinyLlama(ModelMetadata):
    """Metadata for the TinyLlama model family.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Optional RoPE scaling factor.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizer | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
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
