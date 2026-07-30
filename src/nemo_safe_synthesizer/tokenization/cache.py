# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete semantic identity and filesystem layout for T2 token caches."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, cast

from datasets import Dataset, List, Value

from ..errors import GenerationError, ParameterError
from .types import JsonObject, JsonValue, canonical_json_bytes

CACHE_KEY_VERSION = 1
RECORD_ENCODING_ABI = "nss-ordered-jsonl-lf-v1"
ARROW_SCHEMA_ABI = "nss-record-arrow-v1"
RECORD_FRAMING_OPERATION = "compact-utf8-json-top-level-exclusions-terminal-lf-v1"
NATIVE_BATCH_OPERATION = "hf-call-list-no-special-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DATASET_FINGERPRINT = re.compile(r"[A-Za-z0-9._-]{1,256}")
_LOCK_TIMEOUT_SECONDS = 30.0
_MANIFEST_VERSION = 1


class TokenCachePartition(StrEnum):
    """Partition roles that may never share a token cache."""

    TRAIN = "train"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class TokenCacheKey:
    """Immutable complete v1 compatibility key for one mapped record dataset."""

    producer_kind: str
    tokenizer_fragment: str
    dataset_fingerprint: str
    input_columns: tuple[str, ...]
    effective_exclusions: tuple[str, ...]
    serialized_columns: tuple[str, ...]
    retained_columns: tuple[str, ...]
    schema_prompt_ids_digest: str
    max_seq_length: int
    partition: TokenCachePartition
    cache_key_version: int = CACHE_KEY_VERSION
    record_encoding_abi: str = RECORD_ENCODING_ABI
    arrow_schema_abi: str = ARROW_SCHEMA_ABI
    record_framing_operation: str = RECORD_FRAMING_OPERATION
    native_batch_operation: str = NATIVE_BATCH_OPERATION

    def __post_init__(self) -> None:
        if self.cache_key_version != CACHE_KEY_VERSION:
            raise ParameterError(f"Unsupported token cache key version: {self.cache_key_version}.")
        scalar_strings = (
            self.producer_kind,
            self.record_encoding_abi,
            self.arrow_schema_abi,
            self.record_framing_operation,
            self.native_batch_operation,
        )
        if not all(isinstance(value, str) and value for value in scalar_strings):
            raise ParameterError("Token cache operation and ABI identities must be nonempty strings.")
        if _SHA256.fullmatch(self.tokenizer_fragment) is None:
            raise ParameterError("Tokenizer cache identity fragment must be a SHA-256 digest.")
        if _SHA256.fullmatch(self.schema_prompt_ids_digest) is None:
            raise ParameterError("Schema prompt IDs digest must be a SHA-256 digest.")
        if _DATASET_FINGERPRINT.fullmatch(self.dataset_fingerprint) is None:
            raise ParameterError("Dataset fingerprint is missing or invalid for reusable token caching.")
        column_groups = (
            self.input_columns,
            self.effective_exclusions,
            self.serialized_columns,
            self.retained_columns,
        )
        if not all(
            isinstance(group, tuple) and all(isinstance(column, str) for column in group) for group in column_groups
        ):
            raise ParameterError("Token cache column identities must be immutable tuples of strings.")
        if len(set(self.input_columns)) != len(self.input_columns):
            raise ParameterError("Token cache input columns must be unique and ordered.")
        if not isinstance(self.max_seq_length, int) or isinstance(self.max_seq_length, bool) or self.max_seq_length < 0:
            raise ParameterError("Token cache max sequence length must be a non-negative integer.")
        if not isinstance(self.partition, TokenCachePartition):
            raise ParameterError("Token cache partition must be train or validation.")

    def to_dict(self) -> JsonObject:
        """Return the canonical v1 payload."""
        return {
            "cache_key_version": self.cache_key_version,
            "producer_kind": self.producer_kind,
            "record_encoding_abi": self.record_encoding_abi,
            "arrow_schema_abi": self.arrow_schema_abi,
            "tokenizer_fragment": self.tokenizer_fragment,
            "dataset_fingerprint": self.dataset_fingerprint,
            "input_columns": list(self.input_columns),
            "effective_exclusions": list(self.effective_exclusions),
            "serialized_columns": list(self.serialized_columns),
            "retained_columns": list(self.retained_columns),
            "record_framing_operation": self.record_framing_operation,
            "native_batch_operation": self.native_batch_operation,
            "capacity_guard": {
                "schema_prompt_ids_sha256": self.schema_prompt_ids_digest,
                "max_seq_length": self.max_seq_length,
            },
            "partition": self.partition.value,
        }

    @property
    def canonical_bytes(self) -> bytes:
        """Return canonical compact UTF-8 key bytes."""
        return canonical_json_bytes(cast(JsonValue, self.to_dict()))

    @property
    def digest(self) -> str:
        """Return the semantic SHA-256 namespace digest."""
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class TokenCachePaths:
    """Filesystem paths for one semantic cache namespace."""

    digest: str
    directory: Path
    arrow: Path
    manifest: Path
    lock: Path


def token_cache_paths(cache_root: str | Path, key: TokenCacheKey) -> TokenCachePaths:
    """Select the versioned filesystem-safe namespace for ``key``."""
    digest = key.digest
    directory = Path(cache_root) / "nss-token-cache" / f"v{key.cache_key_version}" / digest
    return TokenCachePaths(
        digest=digest,
        directory=directory,
        arrow=directory / "records.tokens.arrow",
        manifest=directory / "manifest.json",
        lock=directory.parent / f"{digest}.lock",
    )


def schema_prompt_ids_digest(input_ids: list[int]) -> str:
    """Hash exact schema prompt IDs for the unchanged capacity guard."""
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in input_ids):
        raise ParameterError("Schema prompt IDs must be integers.")
    return hashlib.sha256(canonical_json_bytes(cast(JsonValue, input_ids))).hexdigest()


def dataset_fingerprint(dataset: Dataset) -> str:
    """Return the authoritative reusable pre-map Dataset fingerprint."""
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str) or _DATASET_FINGERPRINT.fullmatch(fingerprint) is None:
        raise ParameterError("Dataset fingerprint is missing or invalid for reusable token caching.")
    return fingerprint


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GenerationError(f"Failed to read token cache file {path}: {exc}") from exc
    return digest.hexdigest()


def _feature_types(dataset: Dataset) -> tuple[tuple[str, str], ...]:
    return tuple((name, repr(dataset.features[name])) for name in dataset.column_names)


def expected_token_cache_feature_types(
    source: Dataset,
    retained_columns: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    """Derive the v1 output feature contract independently from mapped output."""
    if not all(column in source.column_names for column in retained_columns):
        raise ParameterError("Retained token cache columns must exist in the source dataset.")
    return (
        *((column, repr(source.features[column])) for column in retained_columns),
        ("text", repr(Value("string"))),
        ("input_ids", repr(List(Value("int32")))),
        ("attention_mask", repr(List(Value("int8")))),
    )


@dataclass(frozen=True, slots=True)
class TokenCacheManifest:
    """Validated commit marker for one completed Arrow cache."""

    key: TokenCacheKey
    key_digest: str
    output_columns: tuple[str, ...]
    feature_types: tuple[tuple[str, str], ...]
    row_count: int
    arrow_sha256: str
    manifest_version: int = _MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.manifest_version != _MANIFEST_VERSION:
            raise ParameterError(f"Unsupported token cache manifest version: {self.manifest_version}.")
        if self.key_digest != self.key.digest or _SHA256.fullmatch(self.arrow_sha256) is None:
            raise ParameterError("Token cache manifest digests are invalid.")
        if not isinstance(self.output_columns, tuple) or not all(
            isinstance(column, str) for column in self.output_columns
        ):
            raise ParameterError("Token cache manifest output columns are invalid.")
        if not isinstance(self.feature_types, tuple) or not all(
            isinstance(item, tuple) and len(item) == 2 and all(isinstance(value, str) for value in item)
            for item in self.feature_types
        ):
            raise ParameterError("Token cache manifest feature types are invalid.")
        if not isinstance(self.row_count, int) or isinstance(self.row_count, bool) or self.row_count < 0:
            raise ParameterError("Token cache manifest row count is invalid.")

    def to_dict(self) -> JsonObject:
        """Return the canonical manifest payload."""
        return {
            "manifest_version": self.manifest_version,
            "key": self.key.to_dict(),
            "key_digest": self.key_digest,
            "arrow_schema_abi": self.key.arrow_schema_abi,
            "output_columns": list(self.output_columns),
            "feature_types": [[name, type_name] for name, type_name in self.feature_types],
            "row_count": self.row_count,
            "arrow_sha256": self.arrow_sha256,
            "partition": self.key.partition.value,
        }


def _key_from_dict(value: object) -> TokenCacheKey:
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        raise ParameterError("Token cache manifest key must be an object.")
    key_values = {cast(str, name): item for name, item in value.items()}
    expected = {
        "cache_key_version",
        "producer_kind",
        "record_encoding_abi",
        "arrow_schema_abi",
        "tokenizer_fragment",
        "dataset_fingerprint",
        "input_columns",
        "effective_exclusions",
        "serialized_columns",
        "retained_columns",
        "record_framing_operation",
        "native_batch_operation",
        "capacity_guard",
        "partition",
    }
    if set(key_values) != expected:
        raise ParameterError("Token cache manifest key schema is invalid.")
    guard_value = key_values["capacity_guard"]
    if not isinstance(guard_value, dict) or not all(isinstance(name, str) for name in guard_value):
        raise ParameterError("Token cache capacity guard schema is invalid.")
    guard = {cast(str, name): item for name, item in guard_value.items()}
    if set(guard) != {"schema_prompt_ids_sha256", "max_seq_length"}:
        raise ParameterError("Token cache capacity guard schema is invalid.")

    def _string(name: str) -> str:
        item = key_values[name]
        if not isinstance(item, str):
            raise ParameterError(f"Token cache key field {name!r} must be a string.")
        return item

    def _integer(name: str) -> int:
        item = key_values[name]
        if not isinstance(item, int) or isinstance(item, bool):
            raise ParameterError(f"Token cache key field {name!r} must be an integer.")
        return item

    def _strings(name: str) -> tuple[str, ...]:
        item = key_values[name]
        if not isinstance(item, list) or not all(isinstance(entry, str) for entry in item):
            raise ParameterError(f"Token cache key field {name!r} must be a string list.")
        return tuple(cast(str, entry) for entry in item)

    try:
        return TokenCacheKey(
            cache_key_version=_integer("cache_key_version"),
            producer_kind=_string("producer_kind"),
            record_encoding_abi=_string("record_encoding_abi"),
            arrow_schema_abi=_string("arrow_schema_abi"),
            tokenizer_fragment=_string("tokenizer_fragment"),
            dataset_fingerprint=_string("dataset_fingerprint"),
            input_columns=_strings("input_columns"),
            effective_exclusions=_strings("effective_exclusions"),
            serialized_columns=_strings("serialized_columns"),
            retained_columns=_strings("retained_columns"),
            record_framing_operation=_string("record_framing_operation"),
            native_batch_operation=_string("native_batch_operation"),
            schema_prompt_ids_digest=_guard_string(guard, "schema_prompt_ids_sha256"),
            max_seq_length=_guard_integer(guard, "max_seq_length"),
            partition=TokenCachePartition(_string("partition")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ParameterError("Token cache manifest key values are invalid.") from exc


def _guard_string(values: dict[str, object], name: str) -> str:
    item = values[name]
    if not isinstance(item, str):
        raise ParameterError(f"Token cache capacity guard field {name!r} must be a string.")
    return item


def _guard_integer(values: dict[str, object], name: str) -> int:
    item = values[name]
    if not isinstance(item, int) or isinstance(item, bool):
        raise ParameterError(f"Token cache capacity guard field {name!r} must be an integer.")
    return item


def _manifest_from_bytes(value: bytes) -> TokenCacheManifest:
    try:
        decoded = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParameterError("Token cache manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(decoded, dict):
        raise ParameterError("Token cache manifest must be an object.")
    expected = {
        "manifest_version",
        "key",
        "key_digest",
        "arrow_schema_abi",
        "output_columns",
        "feature_types",
        "row_count",
        "arrow_sha256",
        "partition",
    }
    if set(decoded) != expected:
        raise ParameterError("Token cache manifest schema is invalid.")
    key = _key_from_dict(decoded["key"])
    if decoded["arrow_schema_abi"] != key.arrow_schema_abi or decoded["partition"] != key.partition.value:
        raise ParameterError("Token cache manifest repeats inconsistent identity values.")
    try:
        manifest = TokenCacheManifest(
            manifest_version=decoded["manifest_version"],
            key=key,
            key_digest=decoded["key_digest"],
            output_columns=tuple(decoded["output_columns"]),
            feature_types=tuple(tuple(item) for item in decoded["feature_types"]),
            row_count=decoded["row_count"],
            arrow_sha256=decoded["arrow_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise ParameterError("Token cache manifest values are invalid.") from exc
    if canonical_json_bytes(cast(JsonValue, manifest.to_dict())) != value:
        raise ParameterError("Token cache manifest must use canonical JSON.")
    return manifest


def load_valid_token_cache(
    paths: TokenCachePaths,
    key: TokenCacheKey,
    *,
    expected_columns: tuple[str, ...],
    expected_feature_types: tuple[tuple[str, str], ...],
    expected_row_count: int,
) -> Dataset | None:
    """Return a strictly validated hit, or ``None`` for any on-disk miss."""
    if not paths.manifest.is_file() or not paths.arrow.is_file():
        return None
    try:
        manifest = _manifest_from_bytes(paths.manifest.read_bytes())
        if (
            manifest.key != key
            or manifest.key_digest != paths.digest
            or manifest.output_columns != expected_columns
            or manifest.feature_types != expected_feature_types
            or manifest.row_count != expected_row_count
            or manifest.arrow_sha256 != _file_sha256(paths.arrow)
        ):
            return None
        dataset = Dataset.from_file(str(paths.arrow))
        if (
            tuple(dataset.column_names) != manifest.output_columns
            or len(dataset) != manifest.row_count
            or _feature_types(dataset) != manifest.feature_types
        ):
            return None
        return dataset
    except (OSError, ValueError, ParameterError, GenerationError):
        return None


def publish_token_cache_manifest(
    paths: TokenCachePaths,
    key: TokenCacheKey,
    dataset: Dataset,
    *,
    expected_feature_types: tuple[tuple[str, str], ...],
) -> None:
    """Atomically publish the manifest commit marker for a completed Arrow file."""
    if not paths.arrow.is_file():
        raise GenerationError(f"Token cache Arrow output was not published at {paths.arrow}.")
    if _feature_types(dataset) != expected_feature_types:
        raise GenerationError("Token cache map produced unexpected Arrow feature types.")
    manifest = TokenCacheManifest(
        key=key,
        key_digest=paths.digest,
        output_columns=tuple(dataset.column_names),
        feature_types=_feature_types(dataset),
        row_count=len(dataset),
        arrow_sha256=_file_sha256(paths.arrow),
    )
    payload = canonical_json_bytes(cast(JsonValue, manifest.to_dict()))
    paths.directory.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=paths.directory, delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, paths.manifest)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise GenerationError(f"Failed to publish token cache manifest {paths.manifest}: {exc}") from exc


class TokenCacheLock:
    """Bounded kernel-backed lock on one persistent per-key inode."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_seconds: float = _LOCK_TIMEOUT_SECONDS,
    ) -> None:
        self._path = path
        self._timeout_seconds = timeout_seconds
        self._handle: BinaryIO | None = None

    def __enter__(self) -> TokenCacheLock:
        deadline = time.monotonic() + self._timeout_seconds
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = self._path.open("a+b")
        except OSError as exc:
            raise GenerationError(f"Failed to open token cache lock {self._path}: {exc}") from exc
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._handle = handle
                return self
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    handle.close()
                    raise GenerationError(
                        f"Timed out after {self._timeout_seconds:g} seconds waiting for token cache lock {self._path}."
                    )
                time.sleep(0.05)
            except OSError as exc:
                handle.close()
                raise GenerationError(f"Failed to acquire token cache lock {self._path}: {exc}") from exc

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError as exc:
            raise GenerationError(f"Failed to release token cache lock {self._path}: {exc}") from exc
        finally:
            handle.close()
            self._handle = None
