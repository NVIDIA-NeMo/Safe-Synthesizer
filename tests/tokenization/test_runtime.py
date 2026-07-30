# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2 offline runtime construction and immutable provenance tests."""

from __future__ import annotations

import shutil

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.llm.utils import ModelRef
from nemo_safe_synthesizer.tokenization import WorkloadKind, create_runtime_nss_tokenizer, resolve_native_provenance


def test_local_provenance_is_absolute_and_deterministic(tmp_path) -> None:
    local = tmp_path / "tokenizer"
    local.mkdir()

    first = resolve_native_provenance(local)
    second = resolve_native_provenance(str(local))

    assert first == second
    assert first[0] == str(local.resolve())
    assert first[1].startswith("local-path-")
    assert len(first[1]) == len("local-path-") + 64


def test_cached_snapshot_path_uses_immutable_snapshot_commit(tmp_path) -> None:
    commit = "a" * 40
    snapshot = tmp_path / "models--org--model" / "snapshots" / commit
    snapshot.mkdir(parents=True)

    source, revision, _trust_remote_code = resolve_native_provenance(snapshot)

    assert source == str(snapshot.resolve())
    assert revision == commit


def test_explicit_immutable_remote_commit_is_admitted() -> None:
    commit = "b" * 40

    source, revision, _trust_remote_code = resolve_native_provenance(
        "example-org/example-model",
        revision=commit,
    )

    assert source == "example-org/example-model"
    assert revision == commit


@pytest.mark.parametrize("revision", [None, "main", "v1.2.3", "branch-name", "ABCDEF"])
def test_mutable_or_unresolved_remote_provenance_fails_closed(revision) -> None:
    with pytest.raises(ParameterError, match="no resolved immutable commit"):
        resolve_native_provenance("example-org/not-present-model", revision=revision)


def test_runtime_factory_binds_local_native_to_declared_source(fixture_tokenizer, fixture_smollm3_tokenizer) -> None:
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=fixture_smollm3_tokenizer)
    assert metadata is not None

    tokenizer = create_runtime_nss_tokenizer(
        fixture_tokenizer,
        metadata,
        workload_kind=WorkloadKind.TABULAR,
    )

    assert tokenizer.spec.native_source == str(fixture_smollm3_tokenizer)


def test_runtime_factory_rejects_native_and_declared_local_source_mismatch(
    fixture_tokenizer,
    fixture_smollm3_tokenizer,
    tmp_path,
) -> None:
    different_source = tmp_path / "different-tokenizer"
    different_source.mkdir()
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=fixture_smollm3_tokenizer)
    assert metadata is not None
    metadata.model_name_or_path = str(different_source)

    with pytest.raises(ParameterError, match="does not match"):
        create_runtime_nss_tokenizer(
            fixture_tokenizer,
            metadata,
            workload_kind=WorkloadKind.TABULAR,
        )


def test_runtime_factory_prefers_native_cached_snapshot_provenance(
    fixture_smollm3_tokenizer,
    tmp_path,
) -> None:
    commit = "c" * 40
    snapshot = tmp_path / "models--HuggingFaceTB--SmolLM3-3B" / "snapshots" / commit
    shutil.copytree(fixture_smollm3_tokenizer, snapshot)
    native = AutoTokenizer.from_pretrained(snapshot)
    assert isinstance(native, PreTrainedTokenizerBase)
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=snapshot, tokenizer=native)

    tokenizer = create_runtime_nss_tokenizer(
        native,
        metadata,
        workload_kind=WorkloadKind.TABULAR,
    )

    assert tokenizer.spec.native_source == str(snapshot.resolve())
    assert tokenizer.spec.native_revision == commit


def test_runtime_factory_correlates_declared_remote_with_native_cached_snapshot(
    fixture_smollm3_tokenizer,
    hf_cached_snapshot_factory,
    monkeypatch,
) -> None:
    repo_id = "HuggingFaceTB/SmolLM3-3B"
    commit = "e" * 40
    cache_root, snapshot = hf_cached_snapshot_factory(repo_id, commit=commit)
    shutil.copytree(fixture_smollm3_tokenizer, snapshot, dirs_exist_ok=True)
    monkeypatch.setattr(ModelRef, "_default_hf_cache_root", staticmethod(lambda: cache_root))
    native = AutoTokenizer.from_pretrained(snapshot)
    assert isinstance(native, PreTrainedTokenizerBase)
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=repo_id, tokenizer=native)

    tokenizer = create_runtime_nss_tokenizer(
        native,
        metadata,
        workload_kind=WorkloadKind.TABULAR,
    )

    assert tokenizer.spec.native_source == repo_id
    assert tokenizer.spec.native_revision == commit


def test_runtime_factory_derives_nonmutating_eos_padding_for_raw_mistral(tokenizers_dir) -> None:
    local_path = tokenizers_dir / "mistral7b"
    native = AutoTokenizer.from_pretrained(local_path)
    assert isinstance(native, PreTrainedTokenizerBase)
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=local_path, tokenizer=native)
    assert native.pad_token is None
    assert native.pad_token_id is None

    tokenizer = create_runtime_nss_tokenizer(
        native,
        metadata,
        workload_kind=WorkloadKind.TABULAR,
    )

    payload = tokenizer.spec.implementation_payload
    assert '"pad_token":"</s>"' in payload
    assert '"pad_token_id":2' in payload
    assert native.pad_token is None
    assert native.pad_token_id is None
