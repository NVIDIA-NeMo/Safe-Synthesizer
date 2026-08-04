# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence for authoritative Hugging Face tokenizer assets."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from transformers import PreTrainedTokenizerBase

from ..errors import ArtifactError, ParameterError
from ..llm.utils import load_fast_tokenizer

TOKENIZER_ASSET_DIRECTORY = "tokenizer"


@dataclass(frozen=True, slots=True)
class _TokenizerAssetStore:
    """Filesystem boundary for one persisted native-tokenizer directory."""

    directory: Path

    @classmethod
    def from_adapter(cls, adapter_directory: Path) -> Self:
        """Locate the tokenizer store beneath an adapter directory."""
        return cls(Path(adapter_directory) / TOKENIZER_ASSET_DIRECTORY)

    @staticmethod
    @contextmanager
    def _translate_errors(message: str) -> Iterator[None]:
        """Translate expected filesystem and tokenizer failures once."""
        try:
            yield
        except ArtifactError:
            raise
        except (OSError, ValueError) as exc:
            raise ArtifactError(message) from exc

    def files(self) -> tuple[Path, ...]:
        """Return validated regular asset files in stable relative-path order."""
        message = f"Could not inspect persisted tokenizer assets beneath {self.directory}."
        with self._translate_errors(message):
            if not self.directory.is_dir() or self.directory.is_symlink():
                raise ArtifactError(f"Persisted tokenizer asset directory is missing or invalid: {self.directory}.")
            files: list[Path] = []
            for entry in self.directory.rglob("*"):
                if entry.is_symlink() or (not entry.is_file() and not entry.is_dir()):
                    raise ArtifactError(f"Persisted tokenizer assets must contain only regular entries: {entry}.")
                if entry.is_file():
                    files.append(entry)
            if not files or not (self.directory / "tokenizer_config.json").is_file():
                raise ArtifactError(f"Persisted tokenizer assets are missing beneath {self.directory}.")
        return tuple(sorted(files, key=lambda path: path.relative_to(self.directory).as_posix()))

    def digest(self) -> str:
        """Hash saved filenames and bytes for corruption detection."""
        digest = hashlib.sha256()
        for path in self.files():
            relative = path.relative_to(self.directory).as_posix().encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            with self._translate_errors(f"Could not read persisted tokenizer asset {path}."):
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        return digest.hexdigest()

    def save(self, native: PreTrainedTokenizerBase) -> str:
        """Save the finalized tokenizer and return its stable byte digest."""
        with self._translate_errors(f"Could not save tokenizer assets beneath {self.directory}."):
            self.directory.mkdir(parents=True, exist_ok=True)
            native.save_pretrained(os.fspath(self.directory))
        return self.digest()

    def load(self, *, expected_digest: str, trust_remote_code: bool) -> PreTrainedTokenizerBase:
        """Verify and load the local native tokenizer."""
        if self.digest() != expected_digest:
            raise ArtifactError("Persisted tokenizer asset digest does not match model metadata.")
        with self._translate_errors(f"Could not load persisted tokenizer assets beneath {self.directory}."):
            native = load_fast_tokenizer(
                self.directory,
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
        if not isinstance(native, PreTrainedTokenizerBase):
            raise ArtifactError("Persisted tokenizer assets resolved to an unsupported object.")
        return native


def tokenizer_asset_digest(directory: Path) -> str:
    """Hash saved filenames and bytes for corruption detection."""
    return _TokenizerAssetStore(Path(directory)).digest()


def save_tokenizer_assets(native: PreTrainedTokenizerBase, adapter_directory: Path) -> str:
    """Save the finalized native tokenizer and return its byte digest."""
    if not isinstance(native, PreTrainedTokenizerBase):
        raise ParameterError("Tokenizer persistence requires a native Hugging Face tokenizer.")
    return _TokenizerAssetStore.from_adapter(adapter_directory).save(native)


def load_tokenizer_assets(
    adapter_directory: Path,
    *,
    expected_digest: str,
    trust_remote_code: bool,
) -> PreTrainedTokenizerBase:
    """Validate and load a relocatable artifact's local tokenizer assets."""
    return _TokenizerAssetStore.from_adapter(adapter_directory).load(
        expected_digest=expected_digest,
        trust_remote_code=trust_remote_code,
    )
