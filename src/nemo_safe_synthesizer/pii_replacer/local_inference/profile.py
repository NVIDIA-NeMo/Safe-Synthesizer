# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned vLLM serving profiles for the managed local PII planning server."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from ...errors import ParameterError
from ..llm_client import RESERVED_REQUEST_FIELDS

__all__ = [
    "RESERVED_VLLM_OPTIONS",
    "LocalVllmProfile",
    "bundled_profile_for_model",
    "bundled_profile_names",
    "load_profile",
]

_BUNDLED_PROFILE_DIR = Path(__file__).with_name("profiles")
_PROFILE_SUFFIX = ".yaml"

RESERVED_VLLM_OPTIONS = frozenset(
    {
        "--api-key",
        "--enable-log-outputs",
        "--enable-log-requests",
        "--host",
        "--model",
        "--port",
        "--revision",
        "--served-model-name",
        "--tokenizer-revision",
        "--uds",
    }
)
"""``vllm serve`` options NSS sets itself or keeps disabled.

The launch owns the address and credential, the profile owns the model
identity, and request logging stays off because planning prompts contain raw
cell samples.
"""


def _is_reserved_option(argument: str) -> bool:
    """Return whether ``argument`` names a reserved option, including argparse abbreviations."""
    name = argument.split("=", 1)[0].replace("_", "-")
    if not name.startswith("--") or len(name) <= 2:
        return False
    return any(option.startswith(name) for option in RESERVED_VLLM_OPTIONS)


class LocalVllmProfile(BaseModel):
    """Pinned model and vLLM engine settings for one locally served planner model.

    Profiles describe only the model and engine. The listening address comes
    from ``NSS_INFERENCE_ENDPOINT`` or a free loopback port, and every launch
    generates its own API key.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str = Field(min_length=1, description="Hugging Face model repository or local model directory.")
    revision: str = Field(min_length=1, description="Pinned model revision, preferably a full commit SHA.")
    served_model_name: str | None = Field(
        default=None,
        min_length=1,
        description="Model name exposed by the server. Defaults to model_id.",
    )
    gpu_memory_utilization: float = Field(
        default=0.9,
        gt=0,
        le=1,
        description="Fraction of GPU memory vLLM may reserve. Must be in (0, 1].",
    )
    max_model_len: int | None = Field(
        default=None,
        gt=0,
        description="Maximum prompt plus completion tokens per request. Uses the model's limit when unset.",
    )
    max_num_seqs: int | None = Field(
        default=None,
        gt=0,
        description="Maximum concurrently scheduled requests. Uses the vLLM default when unset.",
    )
    tensor_parallel_size: int = Field(default=1, gt=0, description="Number of GPUs used for tensor parallelism.")
    extra_args: tuple[str, ...] = Field(
        default=(),
        description="Additional `vllm serve` options. Options NSS manages, such as --host, --port, "
        "--api-key, and --enable-log-requests, are rejected.",
    )
    request_options: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Chat-completions fields sent with every planning request, such as temperature, top_p, "
        "top_k, and thinking_token_budget. When set, they replace NSS's default temperature of 0. "
        "NSS_INFERENCE_REQUEST_OPTIONS overrides them.",
    )
    environment: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables for the vLLM server process, such as "
        "VLLM_USE_V2_MODEL_RUNNER=0, which thinking_token_budget needs in vLLM 0.27.",
    )
    request_timeout_seconds: float = Field(
        default=60,
        gt=0,
        description="Per-request timeout for planning calls to this model, including reasoning. "
        "NSS_INFERENCE_TIMEOUT overrides it.",
    )
    startup_timeout_seconds: float = Field(
        default=600,
        gt=0,
        description="Maximum seconds to wait for the server to become ready, including any model download.",
    )
    shutdown_timeout_seconds: float = Field(
        default=30,
        gt=0,
        description="Seconds to wait after SIGTERM before force-killing the server.",
    )

    @property
    def served_name(self) -> str:
        """Model name clients must send; ``served_model_name`` or ``model_id``."""
        return self.served_model_name or self.model_id

    @field_validator("request_options")
    @classmethod
    def _reject_reserved_request_fields(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if reserved := sorted(RESERVED_REQUEST_FIELDS.intersection(value)):
            raise ValueError(f"request_options must not set fields NSS manages: {', '.join(reserved)}")
        return value

    @field_validator("environment")
    @classmethod
    def _reject_managed_environment(cls, value: dict[str, str]) -> dict[str, str]:
        managed = sorted(key for key in value if key == "VLLM_API_KEY" or key.startswith("NSS_INFERENCE_"))
        if managed:
            raise ValueError(f"environment must not set variables NSS manages: {', '.join(managed)}")
        return value

    @field_validator("extra_args")
    @classmethod
    def _reject_reserved_options(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        reserved = [argument for argument in value if _is_reserved_option(argument)]
        if reserved:
            raise ValueError(f"extra_args must not set options managed by NSS: {', '.join(reserved)}")
        return value


def bundled_profile_names() -> tuple[str, ...]:
    """Return the names of profiles shipped with NSS."""
    return tuple(sorted(path.stem for path in _BUNDLED_PROFILE_DIR.glob(f"*{_PROFILE_SUFFIX}")))


def bundled_profile_for_model(model_id: str) -> LocalVllmProfile | None:
    """Return the bundled profile whose served model name is ``model_id``, if any."""
    for name in bundled_profile_names():
        profile = load_profile(name)
        if profile.served_name == model_id:
            return profile
    return None


def load_profile(reference: str | Path) -> LocalVllmProfile:
    """Load a bundled profile by name, or a profile YAML file by path.

    Args:
        reference: Bundled profile name, such as ``"gpt-oss-120b"``, or a path
            to a profile YAML file.

    Returns:
        The validated profile.

    Raises:
        ParameterError: If the reference matches neither a bundled profile nor
            a file, or the file is not a valid profile.
    """
    if isinstance(reference, str) and reference in bundled_profile_names():
        path = _BUNDLED_PROFILE_DIR / f"{reference}{_PROFILE_SUFFIX}"
    else:
        path = Path(reference).expanduser()
        if not path.is_file():
            raise ParameterError(
                f"Local inference profile {str(reference)!r} is neither a bundled profile "
                f"({', '.join(bundled_profile_names())}) nor an existing file"
            )

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ParameterError(f"Could not read local inference profile {str(reference)!r}") from exc
    if not isinstance(data, dict):
        raise ParameterError(f"Local inference profile {str(reference)!r} must be a YAML mapping")
    try:
        return LocalVllmProfile.model_validate(data)
    except ValidationError as exc:
        raise ParameterError(f"Invalid local inference profile {str(reference)!r}: {exc}") from exc
