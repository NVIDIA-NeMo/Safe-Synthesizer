# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU memory management, quantization, device mapping, and tokenizer helpers for LLM loading.

Optional LLM dependencies are imported inside the helpers that need them so
lightweight utilities such as ``trust_remote_code_for_model`` remain usable
without installing the full training or inference stack.
"""

from __future__ import annotations

import gc
import json
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, TypeAlias, cast

from typing_extensions import TypeIs

from ..observability import get_logger

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import AutoConfig, PreTrainedTokenizer
    from transformers.utils.quantization_config import QuantizationConfigMixin

    from ..config.training import QuantizationScheme

logger = get_logger(__name__)

AutoMapValue: TypeAlias = str | list[object]
WeightMap: TypeAlias = dict[str, str]


def _is_weight_map(value: object) -> TypeIs[WeightMap]:
    return isinstance(value, dict) and all(
        isinstance(key, str) and isinstance(shard_name, str) for key, shard_name in value.items()
    )


@dataclass(frozen=True, slots=True)
class ModelRef:
    """Resolved model reference for local cache and trust policy decisions.

    Intended public API:
    - ``parse()`` normalizes a user-supplied model string or path without
      contacting Hugging Face.
    - ``target()`` returns the value that should be passed to
      ``from_pretrained``-style loaders: a local snapshot path when available,
      otherwise the original model reference.
    - ``trust_remote_code`` reports whether the reference belongs to a trusted
      organization after accounting for resolved local HF cache paths.
    - ``partial_cached_snapshot()`` returns HF's local snapshot path for the
      repo/revision, even when the snapshot is incomplete.
    - ``missing_required_components()`` reports whether a local model directory
      has the components this project expects before an offline load.
    - ``missing_remote_code_components()`` reports trusted remote-code files
      referenced by Transformers ``auto_map`` metadata but absent locally.

    Deliberate Hugging Face coupling:
    repo-id validation, cache-root resolution, cache scanning, snapshot layout,
    artifact names, tokenizer filenames, and sharded weight index parsing mirror
    current Hugging Face Hub and Transformers behavior. This is intentional so
    NSS decisions match the libraries that load the model. If model loading or
    cache preflight behavior changes after an upstream HF release, inspect this
    class first.

    Internal helpers are not a generic model-layout abstraction. They should
    stay close to HF's implementation rather than grow compatibility shims for
    unrelated storage formats.
    """

    original: str | Path
    repo_id: str | None = None
    revision: str = "main"
    local_path: Path | None = None
    cache_root: Path | None = None

    trusted_orgs: ClassVar[frozenset[str]] = frozenset({"nvidia"})
    tokenizer_artifact_names: ClassVar[frozenset[str]] = frozenset(
        {
            "tokenizer.json",
            "tokenizer.model",
            "sentencepiece.bpe.model",
            "spiece.model",
            "vocab.json",
            "vocab.txt",
            "merges.txt",
        }
    )

    @classmethod
    def parse(
        cls,
        model_name: str | Path,
        *,
        revision: str = "main",
        cache_root: str | Path | None = None,
    ) -> Self:
        """Parse a model identifier or path without contacting Hugging Face.

        This is safe to call in preflight and loader setup because it uses
        Hugging Face's local cache APIs only. Cached-model hits may still cost a
        few milliseconds because HF cache scanning walks cache metadata to
        confirm model artifacts exist.
        """
        cache_root_path = Path(cache_root) if cache_root is not None else cls._default_hf_cache_root()
        model_ref = str(model_name)
        if not model_ref:
            return cls(original=model_name, revision=revision, cache_root=cache_root_path)

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
        """Return the HF repo id for a path inside the configured Hub cache.

        This relies on ``huggingface_hub.scan_cache_dir`` and the current
        ``models--org--repo/snapshots/<commit>`` cache model. It is deliberately
        not a generic path parser.
        """
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
    def _local_snapshot_for_repo(repo_id: str, revision: str, cache_root: Path) -> Path | None:
        """Return HF's local snapshot path without validating completeness.

        Delegates to ``snapshot_download(local_files_only=True)`` so behavior
        stays aligned with Hugging Face cache resolution instead of duplicating
        ref-file lookup rules.
        """
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
        return snapshot_path

    @classmethod
    def _cached_snapshot_for_repo(cls, repo_id: str, revision: str, cache_root: Path) -> Path | None:
        snapshot_path = cls._local_snapshot_for_repo(repo_id, revision, cache_root)
        if snapshot_path is None:
            return None
        if not cls._snapshot_has_model_artifacts(snapshot_path, cache_root):
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
        """Return known model artifact names using HF Hub's public constants.

        Keep this close to Hugging Face's weight naming conventions. New HF
        artifact names or index formats should be reflected here.
        """
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
    def _required_component_status(cls, model_dir: Path) -> dict[str, bool]:
        """Return required local model component presence for a Transformers load.

        The checks are intentionally shaped around ``from_pretrained`` layouts:
        root ``config.json``, recognized tokenizer files, and HF-style weight
        files or shard indexes. Revisit this if Transformers changes accepted
        directory layouts.
        """
        files = [path for path in model_dir.rglob("*") if path.is_file()]
        return {
            "config": (model_dir / "config.json").is_file(),
            "tokenizer": any(path.name in cls.tokenizer_artifact_names for path in files),
            "model weights": cls._has_complete_model_artifacts(model_dir, files),
        }

    @classmethod
    def missing_required_components(cls, model_dir: Path) -> list[str]:
        """Return local model components missing from ``model_dir``."""
        return [name for name, present in cls._required_component_status(model_dir).items() if not present]

    @classmethod
    def missing_remote_code_components(cls, model_dir: Path) -> list[str]:
        """Return trusted remote-code components referenced by config but absent locally."""
        required = cls._remote_code_components(model_dir)
        missing: list[str] = []
        for component, local_path in required:
            if local_path is None or not (model_dir / local_path).is_file():
                missing.append(component)
        return sorted(missing)

    @classmethod
    def _remote_code_components(cls, model_dir: Path) -> list[tuple[str, Path | None]]:
        config_path = model_dir / "config.json"
        try:
            data = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []

        match data.get("auto_map"):
            case dict() as auto_map:
                components: list[tuple[str, Path | None]] = []
                for value in auto_map.values():
                    match value:
                        case (str() | list()) as auto_map_value:
                            for class_ref in cls._auto_map_class_refs(auto_map_value):
                                component = cls._remote_code_component(class_ref)
                                if component is not None:
                                    components.append(component)
                return components
            case _:
                return []

    @staticmethod
    def _auto_map_class_refs(value: AutoMapValue) -> list[str]:
        match value:
            case str() as class_ref:
                return [class_ref]
            case list() as class_refs:
                return [item for item in class_refs if isinstance(item, str)]

    @staticmethod
    def _remote_code_component(class_ref: str) -> tuple[str, Path | None] | None:
        repo_id: str | None = None
        module_ref = class_ref
        if "--" in class_ref:
            repo_id, module_ref = class_ref.split("--", 1)
        if "." not in module_ref:
            return None

        module_name, _ = module_ref.rsplit(".", 1)
        module_path = Path(*module_name.split(".")).with_suffix(".py")
        if repo_id is not None:
            return f"remote code from {repo_id} ({module_path.as_posix()})", None
        return module_path.as_posix(), module_path

    @classmethod
    def _has_complete_model_artifacts(cls, model_dir: Path, files: list[Path]) -> bool:
        weight_indexes = [path for path in files if path.name.endswith(".index.json")]
        if weight_indexes:
            return any(cls._index_references_existing_shards(model_dir, index_path) for index_path in weight_indexes)

        return any(fnmatchcase(path.name, pattern) for path in files for pattern in cls._model_artifact_patterns())

    @staticmethod
    def _index_references_existing_shards(model_dir: Path, index_path: Path) -> bool:
        """Return whether an HF weight index references shards present on disk."""
        try:
            data = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False

        match data.get("weight_map"):
            case weight_map if _is_weight_map(weight_map) and weight_map:
                return all((model_dir / shard_name).is_file() for shard_name in set(weight_map.values()))
            case _:
                return False

    def partial_cached_snapshot(self) -> Path | None:
        """Return the local HF snapshot for this repo/revision, even if it is partial."""
        if self.repo_id is None or self.cache_root is None:
            return None
        return self._local_snapshot_for_repo(self.repo_id, self.revision, self.cache_root)

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

    Args:
        model_name: HuggingFace model identifier or local path.
        cache_root: Hugging Face Hub cache root. Defaults to the configured hub cache.

    Returns:
        Whether to set ``trust_remote_code=True`` when loading the model.
    """
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


@dataclass(frozen=True, slots=True)
class _VRAMAllocation:
    """Shared GPU memory calculation for runtime loaders."""

    utilization: float
    memory_bytes: int


def _get_vram_allocations(max_vram_fraction: float | None = None) -> dict[int, _VRAMAllocation]:
    """Calculate maximum memory allocation for each available GPU.

    Reserves a 2 GiB safety buffer on each device, then applies
    ``max_vram_fraction`` to the remaining free memory.

    Args:
        max_vram_fraction: Fraction of total GPU memory to allocate.
            Defaults to ``0.8`` (80 %).

    Returns:
        Mapping of CUDA device index to utilization and byte limit.
    """
    import torch

    if max_vram_fraction is None:
        max_vram_fraction = 0.8
    allocations = {}

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            free, total = torch.cuda.mem_get_info(device=i)
            safe_free = max(free - (2 * 1024**3), 0)
            gpu_memory_utilization = min(max_vram_fraction, safe_free / total) if total > 0 else 0.0
            memory_bytes = int(gpu_memory_utilization * total)
            memory_gib = memory_bytes / (1024**3)
            allocations[i] = _VRAMAllocation(utilization=gpu_memory_utilization, memory_bytes=memory_bytes)
            logger.info(
                f"GPU {i}: Will allocate {memory_gib:.2f}GiB "
                f"({gpu_memory_utilization * 100:.1f}% of {total / (1024**3):.2f}GiB)"
            )

    return allocations


def get_max_vram(max_vram_fraction: float | None = None) -> dict[int, float]:
    """Return vLLM-style GPU utilization fractions for each available GPU."""
    return {device: allocation.utilization for device, allocation in _get_vram_allocations(max_vram_fraction).items()}


def get_max_memory_map(max_vram_fraction: float | None = None) -> dict[int, int]:
    """Return Hugging Face ``max_memory`` byte limits for each available GPU."""
    return {device: allocation.memory_bytes for device, allocation in _get_vram_allocations(max_vram_fraction).items()}


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
    tokenizer = load_fast_tokenizer(
        model_name,
        model_max_length=max_position_embeddings,
    )
    tokenizer = add_bos_eos_tokens_to_tokenizer(tokenizer)
    return tokenizer


def load_fast_tokenizer(model_name_or_path: Path | str, **kwargs: Any) -> PreTrainedTokenizer:
    """Load a tokenizer, preferring the Rust ``tokenizers`` backend.

    Centralizes our tokenizer loads so we consistently request the fast
    (Rust) backend that transformers v5 auto-selects, and log when the
    selected backend falls back to the slow Python implementation.

    Why this matters under v5: transformers v5 consolidated the previously
    split ``tokenization_*.py`` / ``tokenization_*_fast.py`` modules into
    a single file per model with automatic backend selection. ``use_fast``
    defaults to ``True``, but a small set of models with no Rust port
    (older SentencePiece-only checkpoints) still resolve to the slow
    backend. Surfacing that fallback gives operators a clear signal when
    tokenization is on the slow path — meaningful in our data-prep
    pipeline where assembling training examples is tokenizer-bound.

    Args:
        model_name_or_path: HuggingFace model id or local path.
        **kwargs: Forwarded to ``AutoTokenizer.from_pretrained`` (e.g.
            ``model_max_length``, ``trust_remote_code``). ``use_fast`` is
            forced to ``True``.

    Returns:
        Loaded ``PreTrainedTokenizer`` (Rust-backed when available).
    """
    from transformers import AutoTokenizer, PreTrainedTokenizer

    kwargs["use_fast"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
    if not getattr(tokenizer, "is_fast", False):
        logger.warning(
            "Loaded slow (Python) tokenizer for %r — no Rust backend available. "
            "Data-prep tokenization will be ~5-10x slower than the fast path.",
            str(model_name_or_path),
        )
    return cast(PreTrainedTokenizer, tokenizer)


def get_device_name() -> str:
    """Get the name of the current device (first index). Returns 'undefined' if the device is not available."""
    # torch may be absent (CPU-only install); CUDA/driver problems surface as
    # RuntimeError/AssertionError from get_device_properties. Anything else is
    # unexpected and should propagate rather than masquerade as 'undefined'.
    try:
        import torch

        return torch.cuda.get_device_properties(0).name
    except (ImportError, RuntimeError, AssertionError):
        logger.debug("Could not resolve CUDA device name; reporting 'undefined'.", exc_info=True)
        return "undefined"


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


def get_quantization_config(scheme: QuantizationScheme | str | Literal[4, 8]) -> QuantizationConfigMixin:
    """Compatibility wrapper for building a transformers v5 quantization config.

    Accepts a :class:`QuantizationScheme` (or its string value) for new
    callers, or an integer ``4`` / ``8`` for backward compatibility with the
    legacy ``quantization_bits`` field (4 → ``bnb-4bit``, 8 → ``bnb-8bit``).
    New code should prefer
    :meth:`nemo_safe_synthesizer.config.training.QuantizationScheme.to_transformers_config`.

    Args:
        scheme: A ``QuantizationScheme`` value, its string equivalent
            (e.g. ``"nvfp4"``), or the legacy bit-count alias.

    Returns:
        A transformers ``QuantizationConfigMixin`` subclass instance
        (``BitsAndBytesConfig``, ``FineGrainedFP8Config``, ``TorchAoConfig``,
        or ``Mxfp4Config``) ready to pass to ``from_pretrained()`` via the
        ``quantization_config=`` kwarg.

    Raises:
        ValueError: If ``scheme`` is not a recognized scheme name or bit count.
        ImportError: If the underlying quantization backend is not installed
            (e.g. torchao for NVFP4 / MXFP4).
    """
    from ..config.training import QuantizationScheme

    return QuantizationScheme.from_alias(scheme).to_transformers_config()
