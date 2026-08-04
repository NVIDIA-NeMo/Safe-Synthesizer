# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""New/legacy artifact selection and saved-tokenizer authority tests."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.llm.metadata import ModelMetadata, TokenizerRepresentation
from nemo_safe_synthesizer.tokenization import WorkloadKind, bind_tokenizer
from nemo_safe_synthesizer.tokenization.persistence import (
    TOKENIZER_ASSET_DIRECTORY,
    load_tokenizer_assets,
    save_tokenizer_assets,
    tokenizer_asset_digest,
)


def _workdir(path: Path) -> Workdir:
    workdir = Workdir(
        base_path=path,
        dataset_name="dataset",
        config_name="config",
        run_name="run",
        _current_phase="train",
    )
    workdir.ensure_directories()
    return workdir


def _new_metadata(tokenizers_dir: Path, workdir: Workdir, name: str = "smollm3b") -> ModelMetadata:
    source = tokenizers_dir / name
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source))
    metadata = ModelMetadata.from_str_or_path(source, tokenizer=native, workdir=workdir)
    metadata.tokenization = bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)
    metadata.tokenizer = native
    return metadata


def test_saved_assets_are_authoritative_and_relocatable(tokenizers_dir: Path, tmp_path: Path) -> None:
    original = _workdir(tmp_path / "original")
    metadata = _new_metadata(tokenizers_dir, original)
    assert metadata.tokenizer is not None
    expected_ids = metadata.tokenizer.encode("probe", add_special_tokens=False)
    metadata.save_metadata()
    source_adapter = original.train.adapter.path
    relocated = tmp_path / "relocated-adapter"
    shutil.copytree(source_adapter, relocated)

    loaded = ModelMetadata.from_metadata_json(relocated / "metadata_v2.json")

    assert loaded.tokenizer_representation is not None
    assert loaded.tokenization is not None
    assert loaded.tokenizer is loaded.tokenization.native
    assert loaded.tokenizer.encode("probe", add_special_tokens=False) == expected_ids
    assert Path(loaded.tokenizer.name_or_path).resolve() == (relocated / TOKENIZER_ASSET_DIRECTORY).resolve()


def test_mistral_saved_assets_persist_derived_pad_id(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir, "mistral7b")
    assert metadata.tokenizer is not None
    assert metadata.tokenizer.pad_token_id == 2

    metadata.save_metadata()
    loaded = ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)

    assert loaded.tokenizer is not None
    assert loaded.tokenizer.pad_token_id == 2
    assert loaded.tokenizer.eos_token_id == 2


def test_metadata_representation_is_minimal(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)

    metadata.save_metadata()
    payload = json.loads(workdir.train.adapter.metadata.read_text())

    assert set(payload["tokenizer_representation"]) == {
        "artifact_version",
        "workload",
        "record_format_version",
        "tokenizer_asset_digest",
    }
    assert payload["tokenizer_representation"]["artifact_version"] == 2
    assert payload["tokenizer_representation"]["record_format_version"] == 1
    assert "registry" not in json.dumps(payload)
    assert "source" not in payload["tokenizer_representation"]


def test_claimed_representation_cannot_be_resaved_without_native_binding(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    metadata.tokenization = None
    metadata.tokenizer_representation = TokenizerRepresentation(
        workload=WorkloadKind.TABULAR,
        tokenizer_asset_digest="a" * 64,
    )

    with pytest.raises(ParameterError, match="without its bound native tokenizer"):
        metadata.save_metadata()


def test_absent_representation_selects_explicit_legacy_route(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    payload = metadata.model_dump(mode="json")
    payload.pop("tokenizer_representation", None)
    workdir.train.adapter.metadata.write_text(json.dumps(payload))

    loaded = ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)

    assert loaded.tokenizer_representation is None
    assert loaded.tokenization is None
    assert loaded.tokenizer is None


def test_claimed_new_artifact_missing_asset_directory_fails(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    metadata.tokenizer_representation = TokenizerRepresentation(
        workload=WorkloadKind.TABULAR,
        tokenizer_asset_digest="0" * 64,
    )
    workdir.train.adapter.metadata.write_text(metadata.model_dump_json())

    with pytest.raises(ParameterError, match="directory is missing or invalid"):
        ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)


def test_claimed_new_artifact_missing_config_fails(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    metadata.save_metadata()
    (workdir.train.adapter.path / TOKENIZER_ASSET_DIRECTORY / "tokenizer_config.json").unlink()

    with pytest.raises(ParameterError, match="assets are missing"):
        ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)


def test_claimed_new_artifact_corrupt_asset_fails_before_load(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    metadata.save_metadata()
    config = workdir.train.adapter.path / TOKENIZER_ASSET_DIRECTORY / "tokenizer_config.json"
    config.write_bytes(config.read_bytes() + b"\n")

    with pytest.raises(ParameterError, match="digest does not match"):
        ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)


def test_symlinked_asset_is_rejected(tokenizers_dir: Path, tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizers_dir / "smollm3b"))
    digest = save_tokenizer_assets(native, adapter)
    config = adapter / TOKENIZER_ASSET_DIRECTORY / "tokenizer_config.json"
    target = tmp_path / "outside.json"
    target.write_bytes(config.read_bytes())
    config.unlink()
    config.symlink_to(target)

    with pytest.raises(ParameterError, match="regular entries"):
        load_tokenizer_assets(adapter, expected_digest=digest, trust_remote_code=False)


def test_asset_digest_identifies_saved_bytes_not_source(tokenizers_dir: Path, tmp_path: Path) -> None:
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizers_dir / "smollm3b"))
    first = tmp_path / "first"
    second = tmp_path / "second"

    first_digest = save_tokenizer_assets(native, first)
    second_digest = save_tokenizer_assets(native, second)

    assert first_digest == second_digest
    assert first_digest == tokenizer_asset_digest(first / TOKENIZER_ASSET_DIRECTORY)


def test_asset_digest_changes_with_saved_special_token_binding(tokenizers_dir: Path, tmp_path: Path) -> None:
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizers_dir / "mistral7b"))
    assert native.pad_token_id is None
    first_digest = save_tokenizer_assets(native, tmp_path / "first")
    native.pad_token = native.eos_token
    second_digest = save_tokenizer_assets(native, tmp_path / "second")

    assert first_digest != second_digest


def test_invalid_representation_digest_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        TokenizerRepresentation(workload=WorkloadKind.TABULAR, tokenizer_asset_digest="not-a-digest")


def test_unknown_representation_version_is_rejected(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    metadata.save_metadata()
    payload = json.loads(workdir.train.adapter.metadata.read_text())
    payload["tokenizer_representation"]["artifact_version"] = 3
    workdir.train.adapter.metadata.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="artifact_version"):
        ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)


def test_new_artifact_remote_code_requires_existing_modelref_admission(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    metadata.save_metadata()
    payload = json.loads(workdir.train.adapter.metadata.read_text())
    payload["model_name_or_path"] = "nvidia/custom-tokenizer"
    workdir.train.adapter.metadata.write_text(json.dumps(payload))

    with pytest.raises(ParameterError, match="remote-code admission"):
        ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)


def test_reader_does_not_fall_back_when_new_assets_are_bad(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    metadata.save_metadata()
    assets = workdir.train.adapter.path / TOKENIZER_ASSET_DIRECTORY
    shutil.rmtree(assets)

    with pytest.raises(ParameterError):
        ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)


def test_saved_native_is_one_genuine_transformers_tokenizer(tokenizers_dir: Path, tmp_path: Path) -> None:
    workdir = _workdir(tmp_path)
    metadata = _new_metadata(tokenizers_dir, workdir)
    metadata.save_metadata()

    loaded = ModelMetadata.from_metadata_json(workdir.train.adapter.metadata)

    assert loaded.tokenization is not None
    assert loaded.tokenizer is loaded.tokenization.native
    assert not hasattr(loaded.tokenization, "for_hf")
    assert not hasattr(loaded.tokenization, "native_snapshot")
