# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU memory management, quantization, device mapping, and tokenizer helpers for LLM loading.

Optional LLM dependencies are imported inside the helpers that need them so
lightweight utilities such as ``trust_remote_code_for_model`` remain usable
without installing the full training or inference stack.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

from ..observability import get_logger

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import AutoConfig, BitsAndBytesConfig, PreTrainedTokenizer

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Resolved model reference for local cache and trust policy decisions."""

    original: str | Path
    repo_id: str | None = None
    revision: str = "main"
    local_path: Path | None = None
    cache_root: Path | None = None

    trusted_orgs: ClassVar[frozenset[str]] = frozenset({"nvidia"})

    @classmethod
    def parse(
        cls,
        model_name: str | Path,
        *,
        revision: str = "main",
        cache_root: str | Path | None = None,
    ) -> Self:
        """Parse a model identifier or path without contacting Hugging Face."""
        cache_root_path = Path(cache_root) if cache_root is not None else cls._default_hf_cache_root()
        model_path = Path(model_name)
        if model_path.exists():
            repo_id = cls._repo_id_from_hf_cache_path(model_path, cache_root_path)
            return cls(
                original=model_name,
                repo_id=repo_id,
                revision=revision,
                local_path=model_path,
                cache_root=cache_root_path,
            )

        model_ref = str(model_name)
        repo_id = cls._repo_id_from_hub_identifier(model_ref)
        local_path = cls._cached_snapshot_for_repo(repo_id, revision, cache_root_path) if repo_id else None
        return cls(
            original=model_name,
            repo_id=repo_id,
            revision=revision,
            local_path=local_path,
            cache_root=cache_root_path,
        )

    @staticmethod
    def _default_hf_cache_root() -> Path:
        """Return Hugging Face's configured Hub cache root. Their implementation
        reads and sets this from several environment variables - HF_HOME, HF_HUB_CACHE, etc.
        """
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)

    @staticmethod
    def _repo_id_from_hub_identifier(model_ref: str) -> str | None:
        """Return a valid Hugging Face model repository ID, if ``model_ref`` is one."""
        if not model_ref or model_ref.startswith(("/", ".")):
            return None

        from huggingface_hub.errors import HFValidationError
        from huggingface_hub.utils import validate_repo_id

        try:
            validate_repo_id(model_ref)
        except HFValidationError:
            return None
        return model_ref

    @staticmethod
    def _repo_id_from_hf_cache_path(path: Path, cache_root: Path) -> str | None:
        path_resolved = path.resolve(strict=False)
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound

        try:
            repos = scan_cache_dir(cache_root).repos
        except (CacheNotFound, ValueError):
            return None

        for repo in repos:
            if repo.repo_type != "model":
                continue
            for revision in repo.revisions:
                snapshot_path = revision.snapshot_path.resolve(strict=False)
                if path_resolved.is_relative_to(snapshot_path):
                    return repo.repo_id
        return None

    @staticmethod
    def _cached_snapshot_for_repo(repo_id: str, revision: str, cache_root: Path) -> Path | None:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError

        try:
            snapshot_path = Path(
                snapshot_download(
                    repo_id,
                    revision=revision,
                    cache_dir=cache_root,
                    local_files_only=True,
                )
            )
        except LocalEntryNotFoundError:
            return None
        if not ModelRef._snapshot_has_model_artifacts(snapshot_path, cache_root):
            return None
        return snapshot_path

    @classmethod
    def _snapshot_has_model_artifacts(cls, snapshot_path: Path, cache_root: Path) -> bool:
        """Return whether HF Hub's cache index reports weight artifacts in ``snapshot_path``."""
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound

        artifact_patterns = cls._model_artifact_patterns()
        snapshot_resolved = snapshot_path.resolve(strict=False)

        try:
            repos = scan_cache_dir(cache_root).repos
        except (CacheNotFound, ValueError):
            return False

        for repo in repos:
            if repo.repo_type != "model":
                continue
            for revision in repo.revisions:
                if revision.snapshot_path.resolve(strict=False) != snapshot_resolved:
                    continue
                return any(
                    fnmatchcase(cached_file.file_name, pattern)
                    for cached_file in revision.files
                    for pattern in artifact_patterns
                )
        return False

    @staticmethod
    def _model_artifact_patterns() -> tuple[str, ...]:
        """Return known model artifact names using HF Hub's public constants."""
        from huggingface_hub.constants import (
            FLAX_WEIGHTS_NAME,
            PYTORCH_WEIGHTS_FILE_PATTERN,
            PYTORCH_WEIGHTS_NAME,
            SAFETENSORS_SINGLE_FILE,
            SAFETENSORS_WEIGHTS_FILE_PATTERN,
            TF2_WEIGHTS_FILE_PATTERN,
            TF2_WEIGHTS_NAME,
            TF_WEIGHTS_NAME,
        )

        return (
            PYTORCH_WEIGHTS_NAME,
            PYTORCH_WEIGHTS_FILE_PATTERN.format(suffix="*"),
            SAFETENSORS_SINGLE_FILE,
            SAFETENSORS_WEIGHTS_FILE_PATTERN.format(suffix="*"),
            TF2_WEIGHTS_NAME,
            TF2_WEIGHTS_FILE_PATTERN.format(suffix="*"),
            TF_WEIGHTS_NAME,
            FLAX_WEIGHTS_NAME,
            "*.gguf",
            "consolidated*.pth",
        )
        if not ModelRef._snapshot_has_model_artifacts(snapshot_path, cache_root):
            return None
        return snapshot_path

    @classmethod
    def _snapshot_has_model_artifacts(cls, snapshot_path: Path, cache_root: Path) -> bool:
        """Return whether HF Hub's cache index reports weight artifacts in ``snapshot_path``."""
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.errors import CacheNotFound

        artifact_patterns = cls._model_artifact_patterns()
        snapshot_resolved = snapshot_path.resolve(strict=False)

        try:
            repos = scan_cache_dir(cache_root).repos
        except (CacheNotFound, ValueError):
            return False

        for repo in repos:
            if repo.repo_type != "model":
                continue
            for revision in repo.revisions:
                if revision.snapshot_path.resolve(strict=False) != snapshot_resolved:
                    continue
                return any(
                    fnmatchcase(cached_file.file_name, pattern)
                    for cached_file in revision.files
                    for pattern in artifact_patterns
                )
        return False

    @staticmethod
    def _model_artifact_patterns() -> tuple[str, ...]:
        """Return known model artifact names using HF Hub's public constants."""
        from huggingface_hub.constants import (
            FLAX_WEIGHTS_NAME,
            PYTORCH_WEIGHTS_FILE_PATTERN,
            PYTORCH_WEIGHTS_NAME,
            SAFETENSORS_SINGLE_FILE,
            SAFETENSORS_WEIGHTS_FILE_PATTERN,
            TF2_WEIGHTS_FILE_PATTERN,
            TF2_WEIGHTS_NAME,
            TF_WEIGHTS_NAME,
        )

        return (
            PYTORCH_WEIGHTS_NAME,
            PYTORCH_WEIGHTS_FILE_PATTERN.format(suffix="*"),
            SAFETENSORS_SINGLE_FILE,
            SAFETENSORS_WEIGHTS_FILE_PATTERN.format(suffix="*"),
            TF2_WEIGHTS_NAME,
            TF2_WEIGHTS_FILE_PATTERN.format(suffix="*"),
            TF_WEIGHTS_NAME,
            FLAX_WEIGHTS_NAME,
            "*.gguf",
            "consolidated*.pth",
        )

    @classmethod
    def is_trusted_org(cls, org: str) -> bool:
        """Return whether an organization is allowed to load remote code."""
        return org.casefold() in cls.trusted_orgs

    @property
    def trust_remote_code(self) -> bool:
        """Whether loaders should pass ``trust_remote_code=True`` for this model."""
        if not self.repo_id or "/" not in self.repo_id:
            return False
        org, _ = self.repo_id.split("/", 1)
        return self.is_trusted_org(org)

    def target(self) -> str:
        """Return the local snapshot path when available, otherwise the original input."""
        return str(self.local_path or self.original)


def trust_remote_code_for_model(model_name: str | Path, *, cache_root: str | Path | None = None) -> bool:
    """Determine whether to trust remote code when loading a model.

    Returns ``True`` for model identifiers owned by trusted organizations,
    including configured Hugging Face cache snapshots for those organizations.
    Returns ``True`` for model identifiers owned by trusted organizations,
    including configured Hugging Face cache snapshots for those organizations.

    Args:
        model_name: HuggingFace model identifier or local path.
        cache_root: Hugging Face Hub cache root. Defaults to the configured hub cache.
        cache_root: Hugging Face Hub cache root. Defaults to the configured hub cache.

    Returns:
        Whether to set ``trust_remote_code=True`` when loading the model.
    """
    return ModelRef.parse(model_name, cache_root=cache_root).trust_remote_code
    return ModelRef.parse(model_name, cache_root=cache_root).trust_remote_code


def cleanup_memory() -> None:
    """Run garbage collection and empty the CUDA cache."""
    import torch

    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()


def gpu_stats() -> None:
    """Log current GPU memory reservation and total capacity.

    Queries CUDA device 0 and logs the peak reserved memory and total
    available memory in GiB.
    """
    import torch

    def round_gb(value: float) -> float:
        return round(value / 1024 / 1024 / 1024, 3)

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round_gb(torch.cuda.max_memory_reserved())
    max_memory = round_gb(gpu_stats.total_memory)
    logger.info(f"{start_gpu_memory} GB of memory reserved.")
    logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")


def get_max_vram(max_vram_fraction: float | None = None) -> dict[int, float]:
    """Calculate maximum memory allocation for each available GPU.

    Reserves a 2 GiB safety buffer on each device, then applies
    ``max_vram_fraction`` to the remaining free memory.

    Args:
        max_vram_fraction: Fraction of total GPU memory to allocate.
            Defaults to ``0.8`` (80 %).

    Returns:
        Mapping of CUDA device index to the usable memory fraction.
    """
    import torch

    if max_vram_fraction is None:
        max_vram_fraction = 0.8
    max_memory = {}

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            free, total = torch.cuda.mem_get_info(device=i)
            safe_free = free - (2 * 1024**3)
            gpu_memory_utilization = min(max_vram_fraction, safe_free / total)
            memory_gib = gpu_memory_utilization * total / (1024**3)
            max_memory[i] = gpu_memory_utilization
            logger.info(
                f"GPU {i}: Will allocate {memory_gib:.2f}GiB ({max_vram_fraction * 100}% of {total / (1024**3):.2f}GiB)"
            )

    return max_memory


def add_bos_eos_tokens_to_tokenizer(tokenizer: PreTrainedTokenizer) -> PreTrainedTokenizer:
    """Enable BOS/EOS token injection and set a pad token if missing.

    Mutates ``tokenizer`` in-place to set ``add_bos_token`` and
    ``add_eos_token`` to ``True``.  If no pad token is configured,
    ``pad_token_id`` is set to ``eos_token_id``.

    Args:
        tokenizer: The tokenizer to configure.

    Returns:
        The same tokenizer instance, modified in-place.
    """
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = True
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def get_param_from_config(
    param: str,
    default_value: Any | None = None,
    model_name: str | None = None,
    trust_remote_code: bool | None = None,
    config: AutoConfig | None = None,
) -> str | None:
    """Read a single attribute from a HuggingFace ``AutoConfig``.

    Either an existing ``config`` object or a ``model_name`` (used to
    load one on the fly) must be provided.

    Args:
        param: Name of the config attribute to retrieve.
        default_value: Fallback value when the attribute is absent.
        model_name: HuggingFace model identifier.  Required when
            ``config`` is not supplied.
        trust_remote_code: Passed through to
            ``AutoConfig.from_pretrained`` when loading a config.
        config: Pre-loaded ``AutoConfig``.  Takes precedence over
            ``model_name``.

    Returns:
        The attribute value, or ``default_value`` if the attribute does
        not exist on the config.

    Raises:
        ValueError: If neither ``model_name`` nor ``config`` is provided.
    """
    from transformers import AutoConfig

    if config is None:
        if model_name is None:
            raise ValueError("model_name is required if config is not provided")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    return getattr(config, param, default_value)


def _get_auto_tokenizer(
    model_name: Path | str,
    max_position_embeddings: int,
) -> PreTrainedTokenizer:
    """Load a tokenizer and configure it with BOS/EOS tokens.

    Args:
        model_name: HuggingFace model identifier or local path.
        max_position_embeddings: Maximum sequence length to set on the
            tokenizer via ``model_max_length``.

    Returns:
        Configured ``PreTrainedTokenizer`` with BOS/EOS tokens enabled.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        model_max_length=max_position_embeddings,
    )
    tokenizer = add_bos_eos_tokens_to_tokenizer(tokenizer)
    return tokenizer


def get_device_map(
    model_target: str,
    autoconfig: AutoConfig | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
    local_files_only: bool = False,
    force_single_device: int | None = None,
) -> str | dict[str, int | str]:
    """Infer the device map for a model and optionally pin all layers to one device.

    Uses ``accelerate.infer_auto_device_map`` on an empty-weight model
    skeleton to determine layer-to-device assignments.

    Args:
        model_target: HuggingFace model identifier or local path.
        autoconfig: Pre-loaded ``AutoConfig``.  If ``None``, one is
            loaded from ``model_target``.
        revision: Model revision (branch, tag, or commit hash).
        trust_remote_code: Whether to trust remote code when loading.
        local_files_only: Restrict loading to local files only.
        force_single_device: When set, every layer is assigned to this
            CUDA device index.

    Returns:
        Ordered dictionary mapping layer names to device identifiers.
    """
    from accelerate import infer_auto_device_map, init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = autoconfig or AutoConfig.from_pretrained(
        model_target,
        revision=revision,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    # Create an empty model with the configuration
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=trust_remote_code)
    device_map = infer_auto_device_map(model)
    if force_single_device is not None:
        for key in device_map:
            device_map[key] = force_single_device
    return device_map


def count_trainable_params(model: PeftModel) -> tuple[int, int]:
    """Count trainable and total parameters in a PEFT model.

    Args:
        model: A ``PeftModel`` (or any ``torch.nn.Module``) to inspect.

    Returns:
        A tuple of ``(trainable_params, all_params)``.
    """
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    return trainable_params, all_params


def get_quantization_config(quantization_bits: Literal[4, 8]) -> BitsAndBytesConfig:
    """Build a ``BitsAndBytesConfig`` for 4-bit or 8-bit quantization.

    Both configurations use NormalFloat quantization with double
    quantization enabled and ``bfloat16`` as the compute dtype.

    Args:
        quantization_bits: Number of bits — must be ``4`` or ``8``.

    Returns:
        A ``BitsAndBytesConfig`` ready to pass to model loading.

    Raises:
        ValueError: If ``quantization_bits`` is not 4 or 8.
    """
    import torch
    from transformers import BitsAndBytesConfig

    if quantization_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif quantization_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    else:
        raise ValueError(f"Invalid quantization bits: {quantization_bits}")
