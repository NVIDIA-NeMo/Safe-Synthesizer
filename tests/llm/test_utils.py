# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for llm.utils helpers."""

from pathlib import Path

import pytest

from nemo_safe_synthesizer.llm.utils import ModelRef, trust_remote_code_for_model


def _write_cached_snapshot(
    cache_root: Path,
    repo_id: str,
    *,
    revision: str = "main",
    commit: str = "abc123",
) -> Path:
    repo_cache = cache_root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_cache / "snapshots" / commit
    snapshot.mkdir(parents=True)
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / revision).write_text(commit)
    return snapshot


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


def test_model_ref_trusts_snapshot_under_configured_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "custom-cache"
    snapshot = _write_cached_snapshot(cache_root, "nvidia/Nemotron-Mini-4B-Instruct")

    ref = ModelRef.parse(snapshot, cache_root=cache_root)

    assert ref.repo_id == "nvidia/Nemotron-Mini-4B-Instruct"
    assert ref.trust_remote_code is True
    assert ref.target() == str(snapshot)


def test_model_ref_prefers_cached_snapshot_for_hub_id(tmp_path: Path) -> None:
    cache_root = tmp_path / "custom-cache"
    snapshot = _write_cached_snapshot(cache_root, "nvidia/Nemotron-Mini-4B-Instruct")

    ref = ModelRef.parse("nvidia/Nemotron-Mini-4B-Instruct", cache_root=cache_root)

    assert ref.trust_remote_code is True
    assert ref.target() == str(snapshot)


def test_model_ref_falls_back_to_original_when_cache_missing(tmp_path: Path) -> None:
    ref = ModelRef.parse("meta-llama/Llama-3.2-1B-Instruct", cache_root=tmp_path / "empty-cache")

    assert ref.trust_remote_code is False
    assert ref.target() == "meta-llama/Llama-3.2-1B-Instruct"
