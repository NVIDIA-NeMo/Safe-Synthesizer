# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic identity and Arrow schema for Datasets-owned record-token caches."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from datasets import Dataset, Features, List, Value

from ..errors import GenerationError, ParameterError

RECORD_FORMAT_VERSION = 1
TOKENIZATION_TRANSFORM_VERSION = 2
_DATASET_FINGERPRINT = re.compile(r"[A-Za-z0-9._-]{1,256}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class TokenCacheKey:
    """Inputs that can change the record-token Arrow transform."""

    dataset_fingerprint: str
    tokenizer_digest: str
    serialized_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    retained_columns: tuple[str, ...]
    record_format_version: int = RECORD_FORMAT_VERSION
    transform_version: int = TOKENIZATION_TRANSFORM_VERSION

    def __post_init__(self) -> None:
        if _DATASET_FINGERPRINT.fullmatch(self.dataset_fingerprint) is None:
            raise ParameterError("Dataset fingerprint is missing or invalid for reusable token caching.")
        if _SHA256.fullmatch(self.tokenizer_digest) is None:
            raise ParameterError("Tokenizer cache digest must be a SHA-256 digest.")
        groups = (self.serialized_columns, self.excluded_columns, self.retained_columns)
        if not all(isinstance(group, tuple) and all(isinstance(name, str) for name in group) for group in groups):
            raise ParameterError("Token cache columns must be ordered tuples of strings.")
        if self.record_format_version != RECORD_FORMAT_VERSION:
            raise ParameterError(f"Unsupported record format version: {self.record_format_version}.")
        if self.transform_version != TOKENIZATION_TRANSFORM_VERSION:
            raise ParameterError(f"Unsupported tokenization transform version: {self.transform_version}.")

    @property
    def digest(self) -> str:
        """Return the deterministic transform fingerprint."""
        payload = {
            "dataset_fingerprint": self.dataset_fingerprint,
            "tokenizer_digest": self.tokenizer_digest,
            "serialized_columns": self.serialized_columns,
            "excluded_columns": self.excluded_columns,
            "retained_columns": self.retained_columns,
            "record_format_version": self.record_format_version,
            "transform_version": self.transform_version,
        }
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()


def dataset_fingerprint(dataset: Dataset) -> str:
    """Return the source Dataset fingerprint selected by Datasets."""
    fingerprint = getattr(dataset, "_fingerprint", None)
    if not isinstance(fingerprint, str) or _DATASET_FINGERPRINT.fullmatch(fingerprint) is None:
        raise ParameterError("Dataset fingerprint is missing or invalid for reusable token caching.")
    return fingerprint


def token_cache_file(cache_root: str | Path, key: TokenCacheKey) -> Path:
    """Return the stable cache filename in the redesigned namespace."""
    return Path(cache_root) / "nss-record-tokens" / "v2" / f"{key.digest}.arrow"


def token_cache_features(source: Dataset, retained_columns: tuple[str, ...]) -> Features:
    """Build the complete expected mapped schema before running the transform."""
    if len(set(retained_columns)) != len(retained_columns) or not all(
        column in source.column_names for column in retained_columns
    ):
        raise ParameterError("Retained token cache columns must exist and be unique.")
    return Features(
        {
            **{column: source.features[column] for column in retained_columns},
            "text": Value("string"),
            "input_ids": List(Value("int32")),
            "attention_mask": List(Value("int8")),
        }
    )


@dataclass(frozen=True, slots=True)
class TokenCacheExpectation:
    """Expected Arrow schema, column order, and row count for a cache publication."""

    features: Features
    columns: tuple[str, ...]
    row_count: int

    def validate(self, dataset: Dataset) -> None:
        """Reject an unexpected cache hit or transform publication."""
        if (
            tuple(dataset.column_names) != self.columns
            or dataset.features != self.features
            or len(dataset) != self.row_count
        ):
            raise GenerationError("Token cache produced an unexpected Arrow schema or row count.")


@dataclass(frozen=True, slots=True)
class TokenCacheSpec:
    """Semantic cache identity, storage path, and publication expectation."""

    key: TokenCacheKey
    path: Path
    expectation: TokenCacheExpectation

    @classmethod
    def from_dataset(
        cls,
        source: Dataset,
        *,
        cache_root: str | Path,
        tokenizer_digest: str,
        excluded_columns: Sequence[str],
        retained_columns: Sequence[str],
    ) -> Self:
        """Build the complete cache contract for one source Dataset."""
        if isinstance(excluded_columns, (str, bytes)) or not all(
            isinstance(column, str) for column in excluded_columns
        ):
            raise ParameterError("Excluded token cache columns must be a sequence of strings.")
        if isinstance(retained_columns, (str, bytes)) or not all(
            isinstance(column, str) for column in retained_columns
        ):
            raise ParameterError("Retained token cache columns must be a sequence of strings.")
        input_columns = tuple(source.column_names)
        exclusions = frozenset(excluded_columns)
        retained_names = frozenset(retained_columns)
        effective_exclusions = tuple(column for column in input_columns if column in exclusions)
        retained = tuple(column for column in input_columns if column in retained_names)
        key = TokenCacheKey(
            dataset_fingerprint=dataset_fingerprint(source),
            tokenizer_digest=tokenizer_digest,
            serialized_columns=tuple(column for column in input_columns if column not in exclusions),
            excluded_columns=effective_exclusions,
            retained_columns=retained,
        )
        expectation = TokenCacheExpectation(
            features=token_cache_features(source, retained),
            columns=(*retained, "text", "input_ids", "attention_mask"),
            row_count=len(source),
        )
        return cls(key, token_cache_file(cache_root, key), expectation)

    def validate(self, dataset: Dataset) -> None:
        """Validate a cache hit or transform publication against this specification."""
        self.expectation.validate(dataset)


def validate_token_cache(
    dataset: Dataset,
    *,
    expected_features: Features,
    expected_columns: tuple[str, ...],
    expected_row_count: int,
) -> None:
    """Reject an unexpected cache hit or transform publication."""
    TokenCacheExpectation(expected_features, expected_columns, expected_row_count).validate(dataset)
