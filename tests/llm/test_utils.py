# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for llm.utils helpers."""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from nemo_safe_synthesizer.llm.utils import (
    ModelRef,
    _reclaimable_available_bytes,
    get_max_memory_map,
    get_max_vram,
    get_quantization_config,
    load_fast_tokenizer,
    trust_remote_code_for_model,
)


def test_load_fast_tokenizer_forces_fast_backend() -> None:
    tokenizer = MagicMock()
    tokenizer.is_fast = True

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as from_pretrained:
        result = load_fast_tokenizer("test-model", use_fast=False, trust_remote_code=True)

    assert result is tokenizer
    from_pretrained.assert_called_once_with("test-model", use_fast=True, trust_remote_code=True)


def test_get_quantization_config_rejects_invalid_legacy_bit_alias() -> None:
    with pytest.raises(ValueError, match="Expected 4 or 8"):
        get_quantization_config(5)  # ty: ignore[invalid-argument-type] -- intentionally invalid alias


def test_get_quantization_config_8bit_uses_valid_bitsandbytes_kwargs() -> None:
    config = MagicMock()

    with patch("transformers.BitsAndBytesConfig", return_value=config) as bitsandbytes_config:
        result = get_quantization_config(8)

    assert result is config
    bitsandbytes_config.assert_called_once_with(load_in_8bit=True)


@pytest.mark.parametrize(
    "model_name, expected",
    [
        ("nvidia/Nemotron-Mini-4B-Instruct", True),
        ("nvidia/some-model", True),
        ("gretel/tabulargemma-2b", False),
        ("meta-llama/Llama-3.2-1B-Instruct", False),
        ("/models/my-local-model", False),
        ("/tmp/models--nvidia--not-a-cache-entry", False),
        ("/home/user/.cache/huggingface/models--nvidia--missing-hub", False),
        ("", False),
        ("nvidia", False),
        (Path("/home/user/models/nvidia/Nemotron-Mini-4B-Instruct"), False),
    ],
)
def test_trust_remote_code_for_model(model_name: str | Path, expected: bool) -> None:
    assert trust_remote_code_for_model(model_name) is expected


def test_trust_remote_code_for_model_requires_configured_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "hf-cache"
    spoofed_snapshot = tmp_path / "other" / "huggingface" / "hub" / "models--nvidia--fake" / "snapshots" / "abc123"
    spoofed_snapshot.mkdir(parents=True)

    assert trust_remote_code_for_model(spoofed_snapshot, cache_root=cache_root) is False


def test_vram_helpers_return_fraction_and_hf_memory_map() -> None:
    gib = 1024**3
    discrete_gpu = MagicMock(is_integrated=False)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.mem_get_info", return_value=(10 * gib, 16 * gib)),
        patch("torch.cuda.get_device_properties", return_value=discrete_gpu),
    ):
        assert get_max_vram(max_vram_fraction=0.8) == {0: 0.5}
        assert get_max_memory_map(max_vram_fraction=0.8) == {0: 8 * gib}


def test_vram_helpers_use_reclaimable_memory_on_integrated_gpus() -> None:
    gib = 1024**3
    integrated_gpu = MagicMock(is_integrated=True)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        # Unified memory: mem_get_info under-reports free (2 GiB) ...
        patch("torch.cuda.mem_get_info", return_value=(2 * gib, 120 * gib)),
        patch("torch.cuda.get_device_properties", return_value=integrated_gpu),
        # ... but 60 GiB is reclaimable/available per the kernel.
        patch("nemo_safe_synthesizer.llm.utils._reclaimable_available_bytes", return_value=60 * gib),
    ):
        # free -> 60 GiB, safe_free -> 58 GiB, utilization -> 58/120.
        assert get_max_vram(max_vram_fraction=0.8) == {0: pytest.approx(58 / 120)}
        assert get_max_memory_map(max_vram_fraction=0.8) == {0: 58 * gib}


def test_vram_helpers_cap_reclaimable_memory_at_device_total() -> None:
    gib = 1024**3
    integrated_gpu = MagicMock(is_integrated=True)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.mem_get_info", return_value=(2 * gib, 120 * gib)),
        patch("torch.cuda.get_device_properties", return_value=integrated_gpu),
        # MemAvailable exceeds device total; must be capped so the 2 GiB buffer survives.
        patch("nemo_safe_synthesizer.llm.utils._reclaimable_available_bytes", return_value=200 * gib),
    ):
        # free -> 120 GiB (capped), safe_free -> 118 GiB, utilization -> 118/120.
        assert get_max_vram(max_vram_fraction=1.0) == {0: pytest.approx(118 / 120)}
        assert get_max_memory_map(max_vram_fraction=1.0) == {0: 118 * gib}


def test_vram_helpers_fall_back_to_cuda_free_when_meminfo_unreadable() -> None:
    gib = 1024**3
    integrated_gpu = MagicMock(is_integrated=True)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=1),
        patch("torch.cuda.mem_get_info", return_value=(10 * gib, 16 * gib)),
        patch("torch.cuda.get_device_properties", return_value=integrated_gpu),
        patch("nemo_safe_synthesizer.llm.utils._reclaimable_available_bytes", return_value=None),
    ):
        assert get_max_vram(max_vram_fraction=0.8) == {0: 0.5}


def test_reclaimable_available_bytes_parses_meminfo() -> None:
    meminfo = "MemTotal:       127603160 kB\nMemFree:             204 kB\nMemAvailable:   68157440 kB\n"
    with patch("builtins.open", mock_open(read_data=meminfo)):
        assert _reclaimable_available_bytes() == 68157440 * 1024


def test_reclaimable_available_bytes_returns_none_when_absent() -> None:
    with patch("builtins.open", side_effect=FileNotFoundError):
        assert _reclaimable_available_bytes() is None


def test_model_ref_trusts_snapshot_under_configured_cache_root(tmp_path: Path, hf_cached_snapshot_factory) -> None:
    """HF cache path recognition intentionally tracks Hub snapshot metadata."""
    cache_root = tmp_path / "custom-cache"
    _, snapshot = hf_cached_snapshot_factory("nvidia/Nemotron-Mini-4B-Instruct", cache_root=cache_root)

    ref = ModelRef.parse(snapshot, cache_root=cache_root)

    assert ref.repo_id == "nvidia/Nemotron-Mini-4B-Instruct"
    assert ref.trust_remote_code is True
    assert ref.target() == str(snapshot)


def test_model_ref_prefers_cached_snapshot_for_hub_id(tmp_path: Path, hf_cached_snapshot_factory) -> None:
    """HF cache selection intentionally tracks Hub snapshot resolution."""
    cache_root = tmp_path / "custom-cache"
    _, snapshot = hf_cached_snapshot_factory("nvidia/Nemotron-Mini-4B-Instruct", cache_root=cache_root)

    ref = ModelRef.parse("nvidia/Nemotron-Mini-4B-Instruct", cache_root=cache_root)

    assert ref.trust_remote_code is True
    assert ref.target() == str(snapshot)


@pytest.mark.parametrize("files", [("config.json",), ("model.safetensors.index.json",)])
def test_model_ref_ignores_cached_snapshot_without_model_artifacts(
    tmp_path: Path, files: tuple[str, ...], hf_cached_snapshot_factory
) -> None:
    """HF artifact filtering intentionally tracks Hub weight-file conventions."""
    cache_root = tmp_path / "custom-cache"
    hf_cached_snapshot_factory("nvidia/Nemotron-Mini-4B-Instruct", cache_root=cache_root, files=files)

    ref = ModelRef.parse("nvidia/Nemotron-Mini-4B-Instruct", cache_root=cache_root)

    assert ref.repo_id == "nvidia/Nemotron-Mini-4B-Instruct"
    assert ref.trust_remote_code is True
    assert ref.target() == "nvidia/Nemotron-Mini-4B-Instruct"


def test_model_ref_prefers_cached_snapshot_for_single_component_hub_id(
    tmp_path: Path, hf_cached_snapshot_factory
) -> None:
    """HF cache lookup intentionally handles Hub's single-component repo ids."""
    cache_root = tmp_path / "custom-cache"
    _, snapshot = hf_cached_snapshot_factory("gpt2", cache_root=cache_root)

    ref = ModelRef.parse("gpt2", cache_root=cache_root)

    assert ref.repo_id == "gpt2"
    assert ref.trust_remote_code is False
    assert ref.target() == str(snapshot)


def test_model_ref_resolves_single_component_hub_id_revision(tmp_path: Path, hf_cached_snapshot_factory) -> None:
    """HF revision lookup intentionally tracks Hub refs-to-snapshots layout."""
    cache_root = tmp_path / "custom-cache"
    _, snapshot = hf_cached_snapshot_factory(
        "bert-base-uncased",
        cache_root=cache_root,
        revision="v1",
        commit="def456",
    )

    ref = ModelRef.parse("bert-base-uncased", revision="v1", cache_root=cache_root)

    assert ref.repo_id == "bert-base-uncased"
    assert ref.target() == str(snapshot)


def test_model_ref_existing_local_path_takes_precedence_over_single_component_hub_id(
    tmp_path: Path, hf_cached_snapshot_factory
) -> None:
    """HF-style cache presence must not override an explicit existing local path."""
    cache_root = tmp_path / "custom-cache"
    hf_cached_snapshot_factory("gpt2", cache_root=cache_root)
    local_model = tmp_path / "gpt2"
    local_model.mkdir()

    ref = ModelRef.parse(local_model, cache_root=cache_root)

    assert ref.repo_id is None
    assert ref.target() == str(local_model)


def test_model_ref_falls_back_to_original_when_cache_missing(tmp_path: Path) -> None:
    ref = ModelRef.parse("meta-llama/Llama-3.2-1B-Instruct", cache_root=tmp_path / "empty-cache")

    assert ref.trust_remote_code is False
    assert ref.target() == "meta-llama/Llama-3.2-1B-Instruct"


def test_model_ref_preserves_empty_model_name(tmp_path: Path) -> None:
    ref = ModelRef.parse("", cache_root=tmp_path / "empty-cache")

    assert ref.trust_remote_code is False
    assert ref.target() == ""


def test_model_ref_reports_missing_required_components(tmp_path: Path, model_files_factory) -> None:
    """Local component checks intentionally mirror Transformers load requirements."""
    model_dir = model_files_factory(tmp_path / "model", files=("config.json",))

    assert ModelRef.missing_required_components(model_dir) == ["tokenizer", "model weights"]


def test_model_ref_reports_missing_root_config(tmp_path: Path, model_files_factory) -> None:
    """Root config handling intentionally mirrors Transformers directory loading."""
    model_dir = tmp_path / "model"
    model_files_factory(model_dir, files=("nested/config.json", "tokenizer.json", "model.safetensors"))

    assert ModelRef.missing_required_components(model_dir) == ["config"]


def test_model_ref_reports_incomplete_sharded_weights(
    tmp_path: Path, model_files_factory, hf_weight_index_factory
) -> None:
    """Shard completeness intentionally tracks HF weight-index semantics."""
    model_dir = model_files_factory(
        tmp_path / "model",
        files=("config.json", "tokenizer.json", "model-00001-of-00002.safetensors"),
    )
    hf_weight_index_factory(
        model_dir,
        shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
    )

    assert ModelRef.missing_required_components(model_dir) == ["model weights"]


def test_model_ref_accepts_complete_sharded_weights(
    tmp_path: Path, model_files_factory, hf_weight_index_factory
) -> None:
    """Shard acceptance intentionally tracks HF weight-index semantics."""
    model_dir = model_files_factory(
        tmp_path / "model",
        files=(
            "config.json",
            "tokenizer.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
    )
    hf_weight_index_factory(
        model_dir,
        shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
    )

    assert ModelRef.missing_required_components(model_dir) == []


def test_model_ref_partial_cached_snapshot_returns_partial_snapshot(tmp_path: Path, hf_cached_snapshot_factory) -> None:
    """Partial snapshot discovery intentionally delegates to HF local cache rules."""
    cache_root = tmp_path / "custom-cache"
    _, snapshot = hf_cached_snapshot_factory(
        "nvidia/Nemotron-Mini-4B-Instruct", cache_root=cache_root, files=("config.json",)
    )

    ref = ModelRef.parse("nvidia/Nemotron-Mini-4B-Instruct", cache_root=cache_root)

    assert ref.target() == "nvidia/Nemotron-Mini-4B-Instruct"
    assert ref.partial_cached_snapshot() == snapshot
