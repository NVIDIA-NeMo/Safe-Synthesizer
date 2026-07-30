# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T2 complete token-cache identity, integrity, and publication tests."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from datasets import Dataset

from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.tokenization.cache import (
    ARROW_SCHEMA_ABI,
    RECORD_ENCODING_ABI,
    TokenCacheKey,
    TokenCacheLock,
    TokenCacheManifest,
    TokenCachePartition,
    expected_token_cache_feature_types,
    load_valid_token_cache,
    publish_token_cache_manifest,
    token_cache_paths,
)
from nemo_safe_synthesizer.tokenization.types import canonical_json_bytes


def _key() -> TokenCacheKey:
    return TokenCacheKey(
        producer_kind="training-example-assembler",
        tokenizer_fragment="a" * 64,
        dataset_fingerprint="dataset-fingerprint",
        input_columns=("b", "hidden", "a"),
        effective_exclusions=("hidden",),
        serialized_columns=("b", "a"),
        retained_columns=("hidden",),
        schema_prompt_ids_digest="b" * 64,
        max_seq_length=2048,
        partition=TokenCachePartition.TRAIN,
    )


def test_legacy_uuid_cache_behavior_is_characterized_before_replacement(tmp_path):
    legacy_base = tmp_path / "safe-synthesizer-dataset-cache_abcde"

    assert legacy_base.with_suffix(".tokens.arrow").name == "safe-synthesizer-dataset-cache_abcde.tokens.arrow"
    assert legacy_base.with_suffix(".val.tokens.arrow").name == "safe-synthesizer-dataset-cache_abcde.val.tokens.arrow"

    legacy_base.with_suffix(".tokens.arrow").write_bytes(b"poisoned legacy cache")
    paths, _mapped = _write_valid_cache(tmp_path)

    assert paths.arrow != legacy_base.with_suffix(".tokens.arrow")
    assert _load(paths, _key()) is not None


def test_complete_cache_key_is_deterministic_and_root_independent(tmp_path) -> None:
    key = _key()

    first = token_cache_paths(tmp_path / "one", key)
    second = token_cache_paths(tmp_path / "two", key)

    assert key.record_encoding_abi == RECORD_ENCODING_ABI
    assert key.arrow_schema_abi == ARROW_SCHEMA_ABI
    assert first.digest == second.digest == key.digest
    assert first.directory.relative_to(tmp_path / "one").parts == ("nss-token-cache", "v1", key.digest)
    assert second.directory.relative_to(tmp_path / "two").parts == ("nss-token-cache", "v1", key.digest)
    assert first.arrow.name == "records.tokens.arrow"
    assert first.manifest.name == "manifest.json"
    assert all(
        raw not in str(first.directory) for raw in ("training-example-assembler", "hidden", "dataset-fingerprint")
    )


def test_every_complete_identity_component_invalidates_cache() -> None:
    base = _key()
    variants = (
        replace(base, producer_kind="other"),
        replace(base, record_encoding_abi="record-v2"),
        replace(base, arrow_schema_abi="arrow-v2"),
        replace(base, tokenizer_fragment="c" * 64),
        replace(base, dataset_fingerprint="other-dataset"),
        replace(base, input_columns=("a", "hidden", "b")),
        replace(base, effective_exclusions=()),
        replace(base, serialized_columns=("a", "b")),
        replace(base, retained_columns=("a",)),
        replace(base, record_framing_operation="record-frame-v2"),
        replace(base, native_batch_operation="native-batch-v2"),
        replace(base, schema_prompt_ids_digest="d" * 64),
        replace(base, max_seq_length=4096),
        replace(base, partition=TokenCachePartition.VALIDATION),
    )

    assert len({base.digest, *(variant.digest for variant in variants)}) == len(variants) + 1


def _write_valid_cache(tmp_path, key: TokenCacheKey | None = None):
    key = key or _key()
    paths = token_cache_paths(tmp_path, key)
    paths.directory.mkdir(parents=True, exist_ok=True)
    source = Dataset.from_dict({"a": [1, 2]})
    mapped = source.map(
        lambda batch: {
            "text": [f'{{"a":{value}}}\n' for value in batch["a"]],
            "input_ids": [[value] for value in batch["a"]],
            "attention_mask": [[1] for _value in batch["a"]],
        },
        batched=True,
        remove_columns=["a"],
        cache_file_name=str(paths.arrow),
        load_from_cache_file=False,
        new_fingerprint=key.digest,
    )
    expected_types = expected_token_cache_feature_types(source, ())
    publish_token_cache_manifest(paths, key, mapped, expected_feature_types=expected_types)
    return paths, mapped


def _load(paths, key: TokenCacheKey):
    return load_valid_token_cache(
        paths,
        key,
        expected_columns=("text", "input_ids", "attention_mask"),
        expected_feature_types=(
            ("text", "Value('string')"),
            ("input_ids", "List(Value('int32'))"),
            ("attention_mask", "List(Value('int8'))"),
        ),
        expected_row_count=2,
    )


def _process_partial_writer(cache_root, key, partial_ready, finish_write, result_queue) -> None:
    paths = token_cache_paths(cache_root, key)
    with TokenCacheLock(paths.lock):
        paths.directory.mkdir(parents=True, exist_ok=True)
        paths.arrow.write_bytes(b"partial")
        partial_ready.set()
        if not finish_write.wait(10):
            result_queue.put("writer-timeout")
            return
        source = Dataset.from_dict({"a": [1, 2]})
        mapped = source.map(
            lambda batch: {
                "text": [f'{{"a":{value}}}\n' for value in batch["a"]],
                "input_ids": [[value] for value in batch["a"]],
                "attention_mask": [[1] for _value in batch["a"]],
            },
            batched=True,
            remove_columns=["a"],
            cache_file_name=str(paths.arrow),
            load_from_cache_file=False,
            new_fingerprint=key.digest,
        )
        publish_token_cache_manifest(
            paths,
            key,
            mapped,
            expected_feature_types=expected_token_cache_feature_types(source, ()),
        )
    result_queue.put("writer-published")


def _process_cache_reader(cache_root, key, result_queue) -> None:
    paths = token_cache_paths(cache_root, key)
    with TokenCacheLock(paths.lock):
        cached = _load(paths, key)
    result_queue.put(None if cached is None else cached.to_dict())


def test_valid_manifest_hit_loads_without_mapping(tmp_path) -> None:
    key = _key()
    paths, mapped = _write_valid_cache(tmp_path, key)

    cached = _load(paths, key)

    assert cached is not None
    assert cached.to_dict() == mapped.to_dict()


def test_missing_malformed_and_partial_manifest_states_are_misses(tmp_path) -> None:
    key = _key()
    paths, _mapped = _write_valid_cache(tmp_path / "missing", key)
    paths.manifest.unlink()
    assert _load(paths, key) is None

    paths, _mapped = _write_valid_cache(tmp_path / "malformed", key)
    paths.manifest.write_bytes(b"{")
    assert _load(paths, key) is None

    paths, _mapped = _write_valid_cache(tmp_path / "partial", key)
    paths.arrow.unlink()
    assert _load(paths, key) is None


def test_truncated_wrong_hash_schema_and_row_count_are_misses(tmp_path) -> None:
    key = _key()
    paths, _mapped = _write_valid_cache(tmp_path / "truncated", key)
    paths.arrow.write_bytes(paths.arrow.read_bytes()[:32])
    assert _load(paths, key) is None

    paths, _mapped = _write_valid_cache(tmp_path / "hash", key)
    manifest = json.loads(paths.manifest.read_bytes())
    manifest["arrow_sha256"] = "0" * 64
    paths.manifest.write_bytes(canonical_json_bytes(manifest))
    assert _load(paths, key) is None

    paths, _mapped = _write_valid_cache(tmp_path / "schema", key)
    manifest = json.loads(paths.manifest.read_bytes())
    manifest["output_columns"] = ["wrong"]
    paths.manifest.write_bytes(canonical_json_bytes(manifest))
    assert _load(paths, key) is None

    paths, _mapped = _write_valid_cache(tmp_path / "rows", key)
    manifest = json.loads(paths.manifest.read_bytes())
    manifest["row_count"] = 3
    paths.manifest.write_bytes(canonical_json_bytes(manifest))
    assert _load(paths, key) is None


def test_internally_consistent_wrong_arrow_types_are_a_miss(tmp_path) -> None:
    key = _key()
    paths = token_cache_paths(tmp_path, key)
    paths.directory.mkdir(parents=True)
    wrong = Dataset.from_dict(
        {
            "text": ['{"a":1}\n', '{"a":2}\n'],
            "input_ids": [["1"], ["2"]],
            "attention_mask": [[1], [1]],
        }
    )
    wrong = wrong.map(
        lambda batch: batch,
        batched=True,
        cache_file_name=str(paths.arrow),
        load_from_cache_file=False,
        new_fingerprint=key.digest,
    )

    wrong_types = tuple((name, repr(wrong.features[name])) for name in wrong.column_names)
    manifest = TokenCacheManifest(
        key=key,
        key_digest=key.digest,
        output_columns=tuple(wrong.column_names),
        feature_types=wrong_types,
        row_count=len(wrong),
        arrow_sha256=hashlib.sha256(paths.arrow.read_bytes()).hexdigest(),
    )
    paths.manifest.write_bytes(canonical_json_bytes(manifest.to_dict()))

    assert _load(paths, key) is None


def test_requested_partition_mismatch_is_a_miss(tmp_path) -> None:
    key = _key()
    paths, _mapped = _write_valid_cache(tmp_path, key)
    validation_key = replace(key, partition=TokenCachePartition.VALIDATION)

    assert _load(paths, validation_key) is None


def test_concurrent_same_key_publishers_produce_one_valid_cache(tmp_path) -> None:
    key = _key()
    paths = token_cache_paths(tmp_path, key)
    map_calls = 0

    def get_or_create():
        nonlocal map_calls
        with TokenCacheLock(paths.lock):
            if cached := _load(paths, key):
                return cached
            map_calls += 1
            _paths, mapped = _write_valid_cache(tmp_path, key)
            return mapped

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: get_or_create(), range(2)))

    assert map_calls == 1
    assert results[0].to_dict() == results[1].to_dict()
    assert _load(paths, key) is not None


def test_active_old_lock_is_not_stolen_and_times_out(tmp_path) -> None:
    path = tmp_path / "key.lock"
    owner = TokenCacheLock(path, timeout_seconds=0.05)
    with owner:
        with pytest.raises(GenerationError, match="Timed out"):
            with TokenCacheLock(path, timeout_seconds=0.01):
                pass
        assert path.is_file()


def test_lock_release_preserves_the_stable_lock_inode(tmp_path) -> None:
    path = tmp_path / "key.lock"
    with TokenCacheLock(path):
        inode = path.stat().st_ino

    assert path.stat().st_ino == inode
    with TokenCacheLock(path):
        assert path.stat().st_ino == inode


def test_malformed_persistent_lock_file_does_not_require_recovery(tmp_path) -> None:
    malformed = tmp_path / "malformed.lock"
    malformed.write_bytes(b"{")
    with TokenCacheLock(malformed, timeout_seconds=0.1):
        assert malformed.is_file()
    assert malformed.read_bytes() == b"{"


def test_two_contenders_for_unowned_lock_remain_mutually_exclusive(tmp_path) -> None:
    path = tmp_path / "persistent.lock"
    path.touch()
    barrier = threading.Barrier(2)
    active = 0
    maximum_active = 0

    def contend() -> None:
        nonlocal active, maximum_active
        barrier.wait()
        with TokenCacheLock(path):
            active += 1
            maximum_active = max(maximum_active, active)
            time.sleep(0.05)
            active -= 1

    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(executor.map(lambda _index: contend(), range(2)))

    assert maximum_active == 1


def test_process_reader_never_accepts_partial_same_key_publication(tmp_path) -> None:
    context = multiprocessing.get_context("fork")
    key = _key()
    partial_ready = context.Event()
    finish_write = context.Event()
    result_queue = context.Queue()
    writer = context.Process(
        target=_process_partial_writer,
        args=(tmp_path, key, partial_ready, finish_write, result_queue),
    )
    reader = context.Process(target=_process_cache_reader, args=(tmp_path, key, result_queue))

    writer.start()
    assert partial_ready.wait(5)
    reader.start()
    time.sleep(0.1)
    assert reader.is_alive()
    finish_write.set()
    writer.join(10)
    reader.join(10)

    assert writer.exitcode == 0
    assert reader.exitcode == 0
    first = result_queue.get(timeout=1)
    second = result_queue.get(timeout=1)
    assert "writer-published" in (first, second)
    cached = second if first == "writer-published" else first
    assert cached is not None
