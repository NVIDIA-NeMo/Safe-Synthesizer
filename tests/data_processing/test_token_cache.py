# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Datasets-owned semantic record-token cache contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from datasets import Dataset, Features, List, Value
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.data_processing.assembler import TabularDataExampleAssembler
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.tokenization import WorkloadKind, bind_tokenizer
from nemo_safe_synthesizer.tokenization.cache import (
    RECORD_FORMAT_VERSION,
    TOKENIZATION_TRANSFORM_VERSION,
    TokenCacheKey,
    dataset_fingerprint,
    token_cache_features,
    token_cache_file,
    validate_token_cache,
)
from nemo_safe_synthesizer.tokenization.core import _BoundTokenization


def _key(**overrides: Any) -> TokenCacheKey:
    values: dict[str, Any] = {
        "dataset_fingerprint": "source-123",
        "tokenizer_digest": "a" * 64,
        "serialized_columns": ("a", "b"),
        "excluded_columns": ("internal",),
        "retained_columns": ("group",),
    }
    values.update(overrides)
    return TokenCacheKey(**values)


def _assembler(
    tokenizers_dir: Path,
    dataset: Dataset,
    cache_root: Path,
) -> tuple[TabularDataExampleAssembler, _BoundTokenization, ModelMetadata]:
    source = tokenizers_dir / "smollm3b"
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source))
    metadata = ModelMetadata.from_str_or_path(source, tokenizer=native)
    tokenization = bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)
    assembler = TabularDataExampleAssembler(
        dataset=dataset,
        tokenization=tokenization,
        metadata=metadata,
        cache_file_path=cache_root,
        seed=1,
    )
    return assembler, tokenization, metadata


def test_key_contains_only_semantic_transform_inputs() -> None:
    key = _key()

    assert set(key.__dataclass_fields__) == {
        "dataset_fingerprint",
        "tokenizer_digest",
        "serialized_columns",
        "excluded_columns",
        "retained_columns",
        "record_format_version",
        "transform_version",
    }
    assert key.record_format_version == RECORD_FORMAT_VERSION
    assert key.transform_version == TOKENIZATION_TRANSFORM_VERSION


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_fingerprint", "different-source"),
        ("tokenizer_digest", "b" * 64),
        ("serialized_columns", ("b", "a")),
        ("excluded_columns", ("other",)),
        ("retained_columns", ("order",)),
    ],
)
def test_every_semantic_input_invalidates_digest(field: str, value: object) -> None:
    assert _key().digest != _key(**{field: value}).digest


def test_prompt_capacity_producer_and_split_do_not_pollute_key() -> None:
    fields = set(_key().__dataclass_fields__)

    assert "prompt" not in " ".join(fields)
    assert "capacity" not in " ".join(fields)
    assert "producer" not in " ".join(fields)
    assert "partition" not in fields
    assert "registry" not in " ".join(fields)


def test_key_digest_is_stable() -> None:
    assert _key().digest == "975fb6970bcc3d8c50b1d8f927a3b5fe0b94c6e600ff640a67632035df6a4fc8"


def test_invalid_dataset_fingerprint_fails() -> None:
    with pytest.raises(ParameterError, match="fingerprint"):
        _key(dataset_fingerprint="spaces are invalid")


def test_invalid_tokenizer_digest_fails() -> None:
    with pytest.raises(ParameterError, match="SHA-256"):
        _key(tokenizer_digest="short")


def test_old_record_format_version_cannot_be_reused() -> None:
    with pytest.raises(ParameterError, match="record format version"):
        _key(record_format_version=0)


def test_old_transform_version_cannot_be_reused() -> None:
    with pytest.raises(ParameterError, match="transform version"):
        _key(transform_version=1)


def test_new_cache_namespace_has_one_arrow_file(tmp_path: Path) -> None:
    path = token_cache_file(tmp_path, _key())

    assert path.parent.parent.name == "nss-record-tokens"
    assert path.parent.name == "v2"
    assert path.name == f"{_key().digest}.arrow"
    assert "manifest" not in str(path)
    assert "lock" not in str(path)


def test_dataset_fingerprint_is_authoritative() -> None:
    dataset = Dataset.from_dict({"a": [1, 2]})

    assert dataset_fingerprint(dataset) == dataset._fingerprint


def test_missing_dataset_fingerprint_fails() -> None:
    dataset = Dataset.from_dict({"a": [1]})
    dataset._fingerprint = "invalid fingerprint"

    with pytest.raises(ParameterError, match="fingerprint"):
        dataset_fingerprint(dataset)


def test_explicit_features_preserve_retained_types_and_token_types() -> None:
    source = Dataset.from_dict({"group": [1], "value": [2.0]})

    features = token_cache_features(source, ("group",))

    assert features == Features(
        {
            "group": source.features["group"],
            "text": Value("string"),
            "input_ids": List(Value("int32")),
            "attention_mask": List(Value("int8")),
        }
    )


def test_unknown_retained_column_fails() -> None:
    source = Dataset.from_dict({"value": [1]})

    with pytest.raises(ParameterError, match="Retained"):
        token_cache_features(source, ("missing",))


def test_duplicate_retained_column_fails() -> None:
    source = Dataset.from_dict({"value": [1]})

    with pytest.raises(ParameterError, match="Retained"):
        token_cache_features(source, ("value", "value"))


def test_cache_validation_checks_columns_features_and_rows() -> None:
    dataset = Dataset.from_dict(
        {"text": ['{"x":1}\n'], "input_ids": [[1]], "attention_mask": [[1]]},
        features=Features(
            {
                "text": Value("string"),
                "input_ids": List(Value("int32")),
                "attention_mask": List(Value("int8")),
            }
        ),
    )

    validate_token_cache(
        dataset,
        expected_features=dataset.features,
        expected_columns=("text", "input_ids", "attention_mask"),
        expected_row_count=1,
    )


@pytest.mark.parametrize("mismatch", ["columns", "features", "rows"])
def test_cache_validation_rejects_mismatch(mismatch: str) -> None:
    dataset = Dataset.from_dict({"text": ["x"], "input_ids": [[1]], "attention_mask": [[1]]})
    expected_features = dataset.features
    expected_columns = tuple(dataset.column_names)
    expected_rows = 1
    if mismatch == "columns":
        expected_columns = ("input_ids", "text", "attention_mask")
    elif mismatch == "features":
        expected_features = Features({"text": Value("string")})
    else:
        expected_rows = 2

    with pytest.raises(GenerationError, match="Arrow schema or row count"):
        validate_token_cache(
            dataset,
            expected_features=expected_features,
            expected_columns=expected_columns,
            expected_row_count=expected_rows,
        )


def test_datasets_cache_hit_avoids_record_reencoding(
    tokenizers_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = Dataset.from_dict({"x": [1, 2, 3]})
    source = tokenizers_dir / "smollm3b"
    native = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(source))
    metadata = ModelMetadata.from_str_or_path(source, tokenizer=native)
    tokenization = bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)
    calls = 0
    original = _BoundTokenization.encode_records

    def recording_encode(self, records, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, records, **kwargs)

    monkeypatch.setattr(_BoundTokenization, "encode_records", recording_encode)
    first = TabularDataExampleAssembler(
        dataset=dataset,
        tokenization=tokenization,
        metadata=metadata,
        cache_file_path=tmp_path,
    )
    first_calls = calls
    second = TabularDataExampleAssembler(
        dataset=dataset,
        tokenization=tokenization,
        metadata=metadata,
        cache_file_path=tmp_path,
    )

    assert first_calls == 1
    assert calls == first_calls
    assert second.tokenized_records["input_ids"] == first.tokenized_records["input_ids"]


def test_cache_hit_replays_statistics(tokenizers_dir: Path, tmp_path: Path) -> None:
    dataset = Dataset.from_dict({"x": [1, 20, 300]})
    first, tokenization, metadata = _assembler(tokenizers_dir, dataset, tmp_path)

    second = TabularDataExampleAssembler(
        dataset=dataset,
        tokenization=tokenization,
        metadata=metadata,
        cache_file_path=tmp_path,
    )

    assert second.stats["tokens_per_record"].count == len(dataset)
    assert second.stats["tokens_per_record"].mean == first.stats["tokens_per_record"].mean


def test_cache_hit_replays_current_capacity_without_reencoding(
    tokenizers_dir: Path,
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = Dataset.from_dict({"x": ["a moderately long record"]})
    first, tokenization, metadata = _assembler(tokenizers_dir, dataset, tmp_path)
    record_length = len(first.tokenized_records[0]["input_ids"])
    metadata.base_max_seq_length = len(first.prompt_encoding.input_ids) + 2 + record_length - 1

    def forbidden(*args, **kwargs):
        raise AssertionError("cache hit must not encode records")

    monkeypatch.setattr(_BoundTokenization, "encode_records", forbidden)

    with pytest.raises(GenerationError, match="At least one record"):
        TabularDataExampleAssembler(
            dataset=dataset,
            tokenization=tokenization,
            metadata=metadata,
            cache_file_path=tmp_path,
        )


def test_changed_source_dataset_uses_a_distinct_cache_file(tokenizers_dir: Path, tmp_path: Path) -> None:
    first_data = Dataset.from_dict({"x": [1, 2]})
    second_data = Dataset.from_dict({"x": [1, 3]})
    _assembler(tokenizers_dir, first_data, tmp_path)
    _assembler(tokenizers_dir, second_data, tmp_path)

    assert len(list((tmp_path / "nss-record-tokens" / "v2").glob("*.arrow"))) == 2


def test_changed_column_order_uses_a_distinct_cache_file(tokenizers_dir: Path, tmp_path: Path) -> None:
    first_data = Dataset.from_dict({"a": [1], "b": [2]})
    second_data = first_data.select_columns(["b", "a"])
    _assembler(tokenizers_dir, first_data, tmp_path)
    _assembler(tokenizers_dir, second_data, tmp_path)

    assert len(list((tmp_path / "nss-record-tokens" / "v2").glob("*.arrow"))) == 2


def test_cache_directory_contains_no_nss_manifest_or_lock(tokenizers_dir: Path, tmp_path: Path) -> None:
    _assembler(tokenizers_dir, Dataset.from_dict({"x": [1]}), tmp_path)
    names = {path.name for path in tmp_path.rglob("*")}

    assert "manifest.json" not in names
    assert not any(name.endswith(".lock") for name in names)
