# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistence for authoritative Hugging Face tokenizer assets."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from transformers import PreTrainedTokenizerBase

from ..errors import ParameterError
from ..llm.utils import load_fast_tokenizer

TOKENIZER_ASSET_DIRECTORY = "tokenizer"


def _asset_files(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir() or directory.is_symlink():
        raise ParameterError(f"Persisted tokenizer asset directory is missing or invalid: {directory}.")
    files: list[Path] = []
    try:
        for entry in directory.rglob("*"):
            if entry.is_symlink() or (not entry.is_file() and not entry.is_dir()):
                raise ParameterError(f"Persisted tokenizer assets must contain only regular entries: {entry}.")
            if entry.is_file():
                files.append(entry)
    except OSError as exc:
        raise ParameterError(f"Could not inspect persisted tokenizer assets beneath {directory}.") from exc
    if not files or not (directory / "tokenizer_config.json").is_file():
        raise ParameterError(f"Persisted tokenizer assets are missing beneath {directory}.")
    return tuple(sorted(files, key=lambda path: path.relative_to(directory).as_posix()))


def tokenizer_asset_digest(directory: Path) -> str:
    """Hash saved filenames and bytes for corruption detection."""
    digest = hashlib.sha256()
    for path in _asset_files(directory):
        relative = path.relative_to(directory).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise ParameterError(f"Could not read persisted tokenizer asset {path}.") from exc
    return digest.hexdigest()


def save_tokenizer_assets(native: PreTrainedTokenizerBase, adapter_directory: Path) -> str:
    """Save the finalized native tokenizer and return its byte digest."""
    if not isinstance(native, PreTrainedTokenizerBase):
        raise ParameterError("Tokenizer persistence requires a native Hugging Face tokenizer.")
    directory = Path(adapter_directory) / TOKENIZER_ASSET_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    try:
        native.save_pretrained(os.fspath(directory))
    except (OSError, ValueError) as exc:
        raise ParameterError(f"Could not save tokenizer assets beneath {directory}.") from exc
    return tokenizer_asset_digest(directory)


def load_tokenizer_assets(
    adapter_directory: Path,
    *,
    expected_digest: str,
    trust_remote_code: bool,
) -> PreTrainedTokenizerBase:
    """Validate and load a relocatable artifact's local tokenizer assets."""
    directory = Path(adapter_directory) / TOKENIZER_ASSET_DIRECTORY
    actual_digest = tokenizer_asset_digest(directory)
    if actual_digest != expected_digest:
        raise ParameterError("Persisted tokenizer asset digest does not match model metadata.")
    try:
        native = load_fast_tokenizer(
            directory,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
    except (OSError, ValueError) as exc:
        raise ParameterError(f"Could not load persisted tokenizer assets beneath {directory}.") from exc
    if not isinstance(native, PreTrainedTokenizerBase):
        raise ParameterError("Persisted tokenizer assets resolved to an unsupported object.")
    return native
