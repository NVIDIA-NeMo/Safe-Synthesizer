# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model-family metadata for prompt formatting, RoPE scaling, and runtime bookkeeping."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)
from transformers import AutoConfig, PretrainedConfig, PreTrainedTokenizerBase

from ..cli.artifact_structure import Workdir
from ..config.parameters import SafeSynthesizerParameters
from ..defaults import (
    DEFAULT_INSTRUCTION,
    MAX_ROPE_SCALING_FACTOR,
    PROMPT_TEMPLATE,
)
from ..errors import ParameterError
from ..observability import get_logger
from ..utils import load_json, write_json
from .model_policy import NEMOTRON3_NANO_POLICY, model_policy_for, model_policy_for_reference
from .utils import ModelRef, load_fast_tokenizer

logger = get_logger(__name__)

DEFAULT_MAX_SEQ_LENGTH = 2048
GLOBAL_MAX_SEQ_LENGTH = 2048 * 6

GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER = 1.2
"""Margin applied to ``max_tokens_per_example`` when sizing generation
``SamplingParams.max_tokens``. The stat is the actual tokenized max
observed during training, so a small jitter margin is sufficient."""


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

    use_chat_template: bool = False
    """Whether prompts are rendered through the tokenizer's chat template."""

    response_prefix_ids: list[int] = Field(default_factory=list)
    """Tokens inserted immediately before response content."""

    response_suffix_ids: list[int] = Field(default_factory=list)
    """Tokens inserted immediately after response content."""

    @classmethod
    def from_tokenizer(cls, name: str, tokenizer: PreTrainedTokenizerBase | None = None, **kwargs) -> LLMPromptConfig:
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
        if tokenizer is None:
            model_ref = ModelRef.parse(name)
            tokenizer = load_fast_tokenizer(
                model_ref.target(),
                trust_remote_code=model_ref.trust_remote_code,
            )
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
            "use_chat_template": kwargs.get("use_chat_template", False),
            "response_prefix_ids": kwargs.get("response_prefix_ids", []),
            "response_suffix_ids": kwargs.get("response_suffix_ids", []),
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


def _model_load_parameter_error(model_name_or_path: str, err: OSError) -> ParameterError:
    """Return user-facing guidance for model metadata load failures."""
    message = str(err)
    if "couldn't connect to 'https://huggingface.co'" in message or "outgoing traffic has been disabled" in message:
        return ParameterError(
            f"Could not load model metadata for '{model_name_or_path}' from the local Hugging Face cache. "
            "Hugging Face access appears to be offline or disabled, and the model files were not found locally. "
            "Either pre-download the model into the Hugging Face cache, pass a local model path, or unset "
            f"`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` to allow online lookup. Original error: {message}"
        )
    return ParameterError(
        f"Could not load model metadata for '{model_name_or_path}'. Ensure the model is a Transformers-compatible "
        f"causal language model, is accessible, and has config/tokenizer files available. Original error: {message}"
    )


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

    rope_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Native Transformers v5 RoPE parameters preserved from the model config.",
    )

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

        Reads the model's native Transformers v5 ``rope_parameters`` and
        optionally overrides the scaling ``factor``. Falls back to legacy
        top-level ``rope_theta`` and ``rope_scaling`` fields for older configs.

        Args:
            config: A loaded HuggingFace model config.
            factor: Scaling factor override.  Defaults to ``1.0``.

        Returns:
            A ``RopeScaling`` populated from the config.
        """
        rope_parameters = getattr(config, "rope_parameters", None)
        rope_parameters = dict(rope_parameters) if isinstance(rope_parameters, dict) else {}

        legacy_rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(legacy_rope_scaling, dict):
            rope_parameters = {**legacy_rope_scaling, **rope_parameters}

        theta = rope_parameters.get("rope_theta", rope_parameters.get("theta"))
        if not isinstance(theta, (int, float)):
            theta = getattr(config, "rope_theta", None) or 10000.0

        rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))

        return cls(
            rope_type=rope_type,
            factor=factor or 1.0,
            theta=theta,
            rope_parameters=rope_parameters,
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
        rope_parameters = dict(self.rope_parameters)
        rope_parameters.update(
            {
                "rope_type": self.rope_type,
                "factor": self.factor,
                "theta": self.theta,
            }
        )
        return {key: value for key, value in rope_parameters.items() if key != "rope_theta"}


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

    To add a model family, define a ``ModelMetadata`` subclass, configure its
    ``LLMPromptConfig`` from the tokenizer, override ``default_learning_rate``
    if needed, and add the subclass to ``_resolve_model_class`` in the intended
    match order.
    """

    # Learning rate when training.learning_rate is "auto". Override in subclasses.
    default_learning_rate: ClassVar[float] = 0.0005
    uses_rope: ClassVar[bool] = True
    automatic_lora_targets: ClassVar[tuple[str, ...]] = ("q_proj", "k_proj", "v_proj", "o_proj")

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

    max_tokens_per_example: int | None = Field(
        default=None,
        description="Maximum tokenized example length observed during training.",
    )
    """Populated by the training backend from the assembler's
    ``tokens_per_example`` running statistic. Consumed by
    ``generation_max_tokens_for`` to size ``SamplingParams.max_tokens`` so
    fine-tuned LoRAs that fail to emit EOS on short structured outputs do
    not decode wasted tokens to the full context-window cap."""

    max_records_per_group: int | None = Field(
        default=None,
        description="Maximum number of records in any single group observed during training.",
    )
    """Populated by the training backend from the grouped assembler's
    ``records_per_group`` running statistic. Consumed as the default bound on
    per-group record repetition in structured generation for grouped data:
    the grammar forces the group-closing ``eos_token`` after at most this many
    records, so a fine-tuned model that fails to emit the delimiter terminates
    and produces a parseable group instead of decoding to the token cap.
    ``None`` for non-grouped training (the bound does not apply)."""

    tokenizer: PreTrainedTokenizerBase | None = Field(default=None, exclude=True, repr=False)

    @field_validator("max_records_per_group", mode="after")
    @classmethod
    def validate_max_records_per_group(cls, v: int | None) -> int | None:
        """Reject non-positive persisted bounds (``None`` or ``>= 1`` only)."""
        if v is not None and v < 1:
            raise ValueError("max_records_per_group must be None or >= 1")
        return v

    @property
    def response_prefix_ids(self) -> list[int]:
        """Return model-owned token IDs inserted before response content."""
        if self.prompt_config.use_chat_template:
            return list(self.prompt_config.response_prefix_ids)
        return [self.prompt_config.bos_token_id]

    @property
    def response_suffix_ids(self) -> list[int]:
        """Return model-owned token IDs inserted after response content."""
        if self.prompt_config.use_chat_template:
            return list(self.prompt_config.response_suffix_ids)
        return [self.prompt_config.eos_token_id]

    def render_prompt(
        self,
        columns: list[str],
        *,
        prefill: str = "",
        exclude_columns: list[str] | None = None,
    ) -> str:
        """Render the schema and optional prefill with this model's prompt policy."""
        from ..utils import create_schema_prompt

        if not self.prompt_config.use_chat_template:
            return create_schema_prompt(
                columns,
                instruction=self.instruction,
                prompt_template=self.prompt_config.template,
                prefill=prefill,
                exclude_columns=exclude_columns,
            )

        tokenizer = self.tokenizer
        if tokenizer is None:
            model_ref = ModelRef.parse(self.model_name_or_path)
            tokenizer = load_fast_tokenizer(
                model_ref.target(),
                trust_remote_code=model_ref.trust_remote_code,
            )
        excluded = set(exclude_columns or [])
        schema = ",".join(f'"{column}":<unk>' for column in columns if column not in excluded)
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{self.instruction}{schema}"},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return f"{rendered}{prefill}"

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
            model_name_or_path = data["model_name_or_path"]
            model_ref = ModelRef.parse(model_name_or_path)
            try:
                data["autoconfig"] = AutoConfig.from_pretrained(
                    model_ref.target(),
                    trust_remote_code=model_ref.trust_remote_code,
                )
            except OSError as err:
                raise _model_load_parameter_error(model_name_or_path, err) from err

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
        return int((self.base_max_seq_length or DEFAULT_MAX_SEQ_LENGTH) * rsf)

    def generation_max_tokens_for(self, prompt_len: int, multiplier: float | None = None) -> int:
        """Per-sample ``max_tokens`` ceiling, prompt-aware.

        Returns the smaller of:

        1. ``int(max_tokens_per_example * multiplier)`` when the assembler stat
           is populated, else ``max_seq_length``.
        2. ``max_seq_length - prompt_len`` -- vLLM raises when
           ``len(prompt) + max_tokens > max_model_len``
           (`vllm#33418 <https://github.com/vllm-project/vllm/issues/33418>`_).

        ``max_tokens_per_example`` already includes the prompt: it is the
        tokenized length of ``prompt + packed records`` produced by the
        assembler, so the scaled value is an upper bound on any rollout
        rather than a tight output budget. The explicit ``- prompt_len``
        clamp is a defensive belt for legacy adapters where the assembler
        stat is missing and for prompts longer than those seen in training.

        The default ``multiplier`` (``GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER``)
        adds only a small jitter margin, which is enough for most tables but
        too tight for long, unbounded free-text columns: a model that
        over-generates slightly past the longest training example truncates
        mid-JSON and yields no parseable record. Callers wire the user-facing
        ``generation.max_tokens_multiplier`` knob through here to widen the
        budget (bounded by the context window) for such datasets.

        Args:
            prompt_len: Tokenized length of the prompt this sample will
                run against. Pass ``0`` to disable the prompt clamp.
            multiplier: Margin applied to ``max_tokens_per_example``. Defaults
                to ``GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER`` when ``None`` so
                non-generation callers (e.g. the training eval callback) keep
                the legacy sizing.

        Returns:
            Non-negative ``max_tokens`` value safe to feed to ``SamplingParams``.
        """
        if multiplier is None:
            multiplier = GENERATION_MAX_TOKENS_SAFETY_MULTIPLIER
        if self.max_tokens_per_example and self.max_tokens_per_example > 0:
            sized = int(self.max_tokens_per_example * multiplier)
        else:
            sized = self.max_seq_length
        return max(0, min(sized, self.max_seq_length - prompt_len))

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

    @staticmethod
    def _load_config_and_tokenizer(
        model_name_or_path: str,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> tuple[PretrainedConfig, PreTrainedTokenizerBase]:
        """Load ``PretrainedConfig`` and (optionally) ``AutoTokenizer`` for a model.

        Centralises the repeated boilerplate present in every subclass
        ``__init__``: loading the HuggingFace config and, when no
        pre-loaded tokenizer is supplied, fetching one via
        ``AutoTokenizer.from_pretrained``.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
            tokenizer: Pre-loaded tokenizer to reuse.  When ``None`` a new
                one is loaded from ``model_name_or_path``.

        Returns:
            A ``(config, tokenizer)`` tuple ready to pass to ``super().__init__``.
        """
        model_ref = ModelRef.parse(model_name_or_path)
        try:
            config: PretrainedConfig = AutoConfig.from_pretrained(
                model_ref.target(), trust_remote_code=model_ref.trust_remote_code
            )
            if tokenizer is None:
                tokenizer = load_fast_tokenizer(
                    model_ref.target(),
                    trust_remote_code=model_ref.trust_remote_code,
                )
        except OSError as err:
            raise _model_load_parameter_error(model_name_or_path, err) from err
        return config, tokenizer

    @classmethod
    def _resolve_model_class(cls: type["ModelMetadata"], model_name_or_path: Path | str) -> type["ModelMetadata"]:
        """Resolve model name or path to the matching metadata subclass.

        Uses case-insensitive substring matching over the registered subclass
        names. The returned class is not instantiated; callers such as
        ``AutoConfigResolver`` use it to inspect class-level metadata.

        Raises:
            ValueError: If no registered subclass matches.
        """
        model_name = str(model_name_or_path)
        if model_policy_for(model_name) is NEMOTRON3_NANO_POLICY:
            return Nemotron3Nano

        model_path = Path(model_name_or_path)
        if model_path.exists():
            model_ref = ModelRef.parse(model_path)
            if model_policy_for_reference(model_ref.repo_id, model_ref.local_path) is NEMOTRON3_NANO_POLICY:
                return Nemotron3Nano

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
    def stub(cls, config: "SafeSynthesizerParameters") -> "ModelMetadata":
        """Create a minimal ModelMetadata without network access.

        Used when ``check_only=True`` and ``from_config`` fails (e.g. model
        not cached, no network). The returned instance has ``tokenizer=None``,
        which causes ``check_token_budget`` to skip with a warning.
        """
        return cls.model_construct(
            model_name_or_path=config.training.pretrained_model,
            max_seq_length=DEFAULT_MAX_SEQ_LENGTH,
            tokenizer=None,
            autoconfig=None,
        )

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
        tokenizer=None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)

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
            rope_scaling=rope_scaling_factor,  # ty: ignore[invalid-argument-type] -- third-party stub
            rope_parameters_location="autoconfig",
            tokenizer=tokenizer,
            **kwargs,
        )


class Llama32(ModelMetadata):
    """Metadata for Meta Llama 3.2 model family.

    Uses ``<|im_start|>`` (id 151644) as the BOS token and disables
    automatic BOS/EOS injection in prompts.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Optional RoPE scaling factor.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer=None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                name=model_name_or_path,
                tokenizer=tokenizer,
                template="user\n {instruction} {schema} \n assistant\n{prefill}",
                bos_token="<|im_start|>",
                bos_token_id=151644,
                add_bos_token_to_prompt=False,
                add_eos_token_to_prompt=False,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=rope_scaling_factor,  # ty: ignore[invalid-argument-type] -- third-party stub
            rope_parameters_location="autoconfig",
            tokenizer=tokenizer,
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
        tokenizer: PreTrainedTokenizerBase | None = None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)
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
            tokenizer=tokenizer,  # ty: ignore[invalid-argument-type] -- re-annotated local shadows param type
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
        tokenizer=None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)

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
            rope_scaling=rope_scaling_factor,  # ty: ignore[invalid-argument-type] -- third-party stub
            rope_parameters_location="autoconfig",
            tokenizer=tokenizer,
            **kwargs,
        )


class Nemotron3Nano(ModelMetadata):
    """Metadata for the official NVIDIA Nemotron 3 Nano 4B BF16 checkpoint."""

    uses_rope: ClassVar[bool] = False
    automatic_lora_targets: ClassVar[tuple[str, ...]] = NEMOTRON3_NANO_POLICY.automatic_lora_targets

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer=None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        if rope_scaling_factor not in (None, 1, 1.0):
            raise ParameterError("Nemotron 3 Nano does not use RoPE; rope_scaling_factor must be 1")
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)
        empty_user = [
            {"role": "system", "content": ""},
            {"role": "user", "content": ""},
        ]
        dialogue = tokenizer.apply_chat_template(
            empty_user,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix = tokenizer.apply_chat_template(
            empty_user,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full = tokenizer.apply_chat_template(
            [*empty_user, {"role": "assistant", "content": ""}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        dialogue_ids = tokenizer.encode(dialogue, add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
        if prefix_ids[: len(dialogue_ids)] != dialogue_ids or full_ids[: len(prefix_ids)] != prefix_ids:
            raise ParameterError("Nemotron 3 chat-template prefix is not token-boundary stable")

        super().__init__(
            autoconfig=config,
            instruction=DEFAULT_INSTRUCTION,
            prompt_config=LLMPromptConfig.from_tokenizer(
                name=model_name_or_path,
                tokenizer=tokenizer,
                template="{instruction}{schema}{prefill}",
                add_bos_token_to_prompt=False,
                add_eos_token_to_prompt=False,
                use_chat_template=True,
                response_prefix_ids=prefix_ids[len(dialogue_ids) :],
                response_suffix_ids=full_ids[len(prefix_ids) :],
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=None,
            rope_parameters_location="autoconfig",
            tokenizer=tokenizer,
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
        tokenizer=None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)

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
            rope_scaling=rope_scaling_factor,  # ty: ignore[invalid-argument-type] -- third-party stub
            rope_parameters_location="autoconfig",
            tokenizer=tokenizer,
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
        tokenizer=None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)
        if rope_scaling_factor:
            logger.warning(
                f"Rope scaling factor {rope_scaling_factor} is not supported for SmolLM2 due to longer default context lengths. Ignoring."
            )

        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")  # ty: ignore[unresolved-attribute] -- third-party stub
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
            tokenizer=tokenizer,
            **kwargs,
        )


class SmolLM3(ModelMetadata):
    """Metadata for HuggingFace SmolLM3 model family.

    Uses ``<|im_start|>`` (id 128011) as the BOS token.  RoPE scaling
    is not supported. Any supplied ``rope_scaling_factor`` will be
    ignored with a warning.

    Args:
        model_name_or_path: HuggingFace model identifier or local path.
        tokenizer: Optional pre-loaded tokenizer.
        rope_scaling_factor: Ignored with a warning if provided.
        **kwargs: Forwarded to [`ModelMetadata`][nemo_safe_synthesizer.llm.metadata.ModelMetadata].
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer=None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)

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
                template="user\n {instruction} {schema} <|im_end|> \n assistant\n{prefill}",
                add_bos_token_to_prompt=True,
                add_eos_token_to_prompt=False,
                tokenizer=tokenizer,
                name=model_name_or_path,
                bos_token=bos_token,
                bos_token_id=bos_token_id,
            ),
            model_name_or_path=model_name_or_path,
            rope_scaling=None,
            rope_parameters_location="autoconfig",
            tokenizer=tokenizer,
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
        tokenizer=None,
        rope_scaling_factor: float | None = None,
        **kwargs,
    ) -> None:
        config, tokenizer = ModelMetadata._load_config_and_tokenizer(model_name_or_path, tokenizer)

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
            rope_scaling=rope_scaling_factor,  # ty: ignore[invalid-argument-type] -- third-party stub
            rope_parameters_location="autoconfig",
            tokenizer=tokenizer,
            **kwargs,
        )


# Pydantic 2.12 + transformers v5 + `from __future__ import annotations` interact
# such that forward references inside `PretrainedConfig | None` and
# `PreTrainedTokenizerBase | None` unions are only resolvable after the module
# has fully loaded. Transformers includes lazy `torch.*` references in that
# namespace, so provide it explicitly while rebuilding instead of relying on an
# otherwise-unused module import.
_model_rebuild_namespace = {"torch": importlib.import_module("torch")}
for _cls in (
    ModelMetadata,
    Granite,
    Llama32,
    Mistral,
    Nemotron,
    Nemotron3Nano,
    Qwen,
    SmolLM2,
    SmolLM3,
    TinyLlama,
):
    _cls.model_rebuild(_types_namespace=_model_rebuild_namespace)
del _cls, _model_rebuild_namespace
