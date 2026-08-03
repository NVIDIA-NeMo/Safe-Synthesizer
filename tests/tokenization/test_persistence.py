# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native asset and NSS manifest persistence tests."""

from __future__ import annotations

import json
import shutil
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.tokenization import WorkloadKind, create_runtime_nss_tokenizer
from nemo_safe_synthesizer.tokenization.base import native_snapshot
from nemo_safe_synthesizer.tokenization.persistence import load_nss_tokenizer, save_nss_tokenizer


def _runtime(fixture_tokenizer, fixture_smollm3_tokenizer):
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_smollm3_tokenizer,
        tokenizer=fixture_tokenizer,
    )
    return create_runtime_nss_tokenizer(fixture_tokenizer, metadata, workload_kind=WorkloadKind.TABULAR)


def test_native_assets_and_canonical_manifest_round_trip(
    fixture_tokenizer, fixture_smollm3_tokenizer, tmp_path
) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    before = native_snapshot(fixture_tokenizer)

    save_nss_tokenizer(tokenizer, tmp_path)
    restored = load_nss_tokenizer(tmp_path)

    assert (tmp_path / "nss_tokenizer.json").read_bytes() == tokenizer.spec.canonical_bytes()
    assert (tmp_path / "tokenizer_config.json").is_file()
    assert restored is not None
    assert restored.spec == tokenizer.spec
    assert native_snapshot(fixture_tokenizer) == before


def test_present_corrupt_manifest_never_falls_back_to_legacy(
    fixture_tokenizer, fixture_smollm3_tokenizer, tmp_path
) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    save_nss_tokenizer(tokenizer, tmp_path)
    (tmp_path / "nss_tokenizer.json").write_bytes(b"{not-json")

    with pytest.raises(ParameterError, match="nss_tokenizer.json"):
        load_nss_tokenizer(tmp_path, allow_legacy=True)


def test_missing_manifest_has_explicit_legacy_admission(tmp_path) -> None:
    assert load_nss_tokenizer(tmp_path, allow_legacy=True) is None
    with pytest.raises(ParameterError, match="missing"):
        load_nss_tokenizer(tmp_path)


def test_malformed_manifest_filesystem_entry_never_falls_back_to_legacy(tmp_path) -> None:
    (tmp_path / "nss_tokenizer.json").mkdir()

    with pytest.raises(ParameterError, match="regular file"):
        load_nss_tokenizer(tmp_path, allow_legacy=True)


def test_native_assets_without_manifest_are_not_admitted_as_legacy(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").write_text("{}")

    with pytest.raises(ParameterError, match="missing"):
        load_nss_tokenizer(tmp_path, allow_legacy=True)


def test_dangling_native_asset_without_manifest_is_not_admitted_as_legacy(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").symlink_to("missing-tokenizer-config.json")

    with pytest.raises(ParameterError, match="missing"):
        load_nss_tokenizer(tmp_path, allow_legacy=True)


@pytest.mark.parametrize("filename", ["tokenizer_config.json", "tokenizer.json", "merges.txt"])
def test_symlinked_native_asset_is_rejected_before_transformers_loading(
    fixture_tokenizer,
    fixture_smollm3_tokenizer,
    tmp_path,
    filename,
) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    save_nss_tokenizer(tokenizer, tmp_path)
    native_asset = tmp_path / filename
    saved_asset = tmp_path / f"saved-{filename}"
    if native_asset.exists():
        native_asset.rename(saved_asset)
    else:
        saved_asset.write_text("# tokenizer asset\n")
    native_asset.symlink_to(saved_asset.name)

    with patch("nemo_safe_synthesizer.tokenization.persistence.AutoTokenizer.from_pretrained") as load_native:
        with pytest.raises(ParameterError, match="symlink"):
            load_nss_tokenizer(tmp_path)

    load_native.assert_not_called()


def test_present_manifest_requires_matching_native_assets(
    fixture_tokenizer, fixture_smollm3_tokenizer, tmp_path
) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    (tmp_path / "nss_tokenizer.json").write_bytes(tokenizer.spec.canonical_bytes())

    with pytest.raises(ParameterError, match="native tokenizer assets"):
        load_nss_tokenizer(tmp_path)


def test_noncanonical_manifest_fails_closed(fixture_tokenizer, fixture_smollm3_tokenizer, tmp_path) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    save_nss_tokenizer(tokenizer, tmp_path)
    value = json.loads(tokenizer.spec.canonical_bytes())
    (tmp_path / "nss_tokenizer.json").write_text(json.dumps(value, indent=2))

    with pytest.raises(ParameterError, match="canonical"):
        load_nss_tokenizer(tmp_path)


def test_model_metadata_round_trip_owns_manifest_without_embedding_live_tokenizer(
    fixture_tokenizer,
    fixture_smollm3_tokenizer,
    tmp_path,
) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    workdir = Workdir(tmp_path, "config", "dataset", "run")
    workdir.ensure_directories()
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_smollm3_tokenizer,
        tokenizer=fixture_tokenizer,
        workdir=workdir,
    )
    metadata.tokenizer = tokenizer.for_hf()
    metadata.nss_tokenizer = tokenizer

    metadata.save_metadata()
    saved_metadata = json.loads(workdir.train.adapter.metadata.read_bytes())
    restored = ModelMetadata.from_metadata_json(workdir.train.adapter.metadata, workdir=workdir)

    assert "tokenizer" not in saved_metadata
    assert "nss_tokenizer" not in saved_metadata
    assert workdir.train.adapter.nss_tokenizer.read_bytes() == tokenizer.spec.canonical_bytes()
    assert restored.nss_tokenizer is not None
    assert restored.nss_tokenizer.spec == tokenizer.spec
    assert restored.tokenizer is restored.nss_tokenizer.for_hf()


def test_model_metadata_rejects_prompt_policy_drift_when_manifest_is_present(
    fixture_tokenizer,
    fixture_smollm3_tokenizer,
    tmp_path,
) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    workdir = Workdir(tmp_path, "config", "dataset", "run")
    workdir.ensure_directories()
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_smollm3_tokenizer,
        tokenizer=fixture_tokenizer,
        workdir=workdir,
        nss_tokenizer=tokenizer,
    )
    metadata.save_metadata()
    saved_metadata = json.loads(workdir.train.adapter.metadata.read_bytes())
    saved_metadata["prompt_config"]["template"] = "drifted {instruction} {schema} {prefill}"
    workdir.train.adapter.metadata.write_text(json.dumps(saved_metadata))

    with pytest.raises(ParameterError, match="drift"):
        ModelMetadata.from_metadata_json(workdir.train.adapter.metadata, workdir=workdir)


def test_model_metadata_rejects_source_drift_before_transformers_load(
    fixture_tokenizer,
    fixture_smollm3_tokenizer,
    tmp_path,
) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    workdir = Workdir(tmp_path, "config", "dataset", "run")
    workdir.ensure_directories()
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_smollm3_tokenizer,
        tokenizer=fixture_tokenizer,
        workdir=workdir,
        nss_tokenizer=tokenizer,
    )
    metadata.save_metadata()
    saved_metadata = json.loads(workdir.train.adapter.metadata.read_bytes())
    saved_metadata["model_name_or_path"] = "nvidia/drifted-remote-code-model"
    workdir.train.adapter.metadata.write_text(json.dumps(saved_metadata))

    with patch("nemo_safe_synthesizer.llm.metadata.AutoConfig.from_pretrained") as native_loader:
        with pytest.raises(ParameterError, match="source drift"):
            ModelMetadata.from_metadata_json(workdir.train.adapter.metadata, workdir=workdir)

    assert all(call.args[0] != "nvidia/drifted-remote-code-model" for call in native_loader.call_args_list)


def test_manifest_claim_prevents_deletion_downgrade(
    fixture_tokenizer,
    fixture_smollm3_tokenizer,
    tmp_path,
) -> None:
    tokenizer = _runtime(fixture_tokenizer, fixture_smollm3_tokenizer)
    workdir = Workdir(tmp_path, "config", "dataset", "run")
    workdir.ensure_directories()
    metadata = ModelMetadata.from_str_or_path(
        model_name_or_path=fixture_smollm3_tokenizer,
        tokenizer=fixture_tokenizer,
        workdir=workdir,
        nss_tokenizer=tokenizer,
    )
    metadata.save_metadata()
    workdir.train.adapter.nss_tokenizer.unlink()

    with pytest.raises(ParameterError, match="missing"):
        ModelMetadata.from_metadata_json(workdir.train.adapter.metadata, workdir=workdir)


def test_local_source_manifest_reconstructs_after_artifact_relocation(
    fixture_smollm3_tokenizer,
    tmp_path,
) -> None:
    source = tmp_path / "source-smollm3"
    unavailable = tmp_path / "source-moved-away"
    artifact = tmp_path / "portable-artifact"
    shutil.copytree(fixture_smollm3_tokenizer, source)
    native = AutoTokenizer.from_pretrained(source, local_files_only=True)
    assert isinstance(native, PreTrainedTokenizerBase)
    metadata = ModelMetadata.from_str_or_path(model_name_or_path=source, tokenizer=native)
    tokenizer = create_runtime_nss_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)
    save_nss_tokenizer(tokenizer, artifact)
    source.rename(unavailable)

    restored = load_nss_tokenizer(artifact)

    assert restored is not None
    assert restored.spec == tokenizer.spec
