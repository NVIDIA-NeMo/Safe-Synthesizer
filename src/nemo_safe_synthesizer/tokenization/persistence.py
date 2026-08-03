# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native tokenizer asset and NSS manifest persistence boundaries."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ..errors import ParameterError
from .base import NssTokenizerCore
from .registry import NssTokenizerRegistry, builtin_registry
from .spec import NssTokenizerSpec

NSS_TOKENIZER_MANIFEST = "nss_tokenizer.json"
_NATIVE_ASSET_MARKERS = frozenset(
    {
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
        "special_tokens_map.json",
        "vocab.json",
        "vocab.txt",
    }
)


def _manifest_path(directory: Path) -> Path:
    """Return the fixed manifest path beneath an artifact directory."""
    return directory / NSS_TOKENIZER_MANIFEST


def _entry_exists(path: Path) -> bool:
    """Return whether a filesystem entry exists, including a dangling symlink."""
    return os.path.lexists(path)


def _require_regular_file(path: Path, *, description: str) -> None:
    """Require a present regular file without following a symlink."""
    if not _entry_exists(path):
        raise ParameterError(f"{description} is missing: {path}.")
    if path.is_symlink() or not path.is_file():
        raise ParameterError(f"{description} must be a regular file and not a symlink: {path}.")


def _validate_artifact_tree(directory: Path) -> None:
    """Reject symlinked or special entries before external tokenizer loading."""
    try:
        entries = tuple(directory.iterdir())
    except OSError as exc:
        raise ParameterError(f"Could not inspect persisted tokenizer artifact beneath {directory}.") from exc

    for entry in entries:
        if entry.is_symlink():
            raise ParameterError(f"Persisted tokenizer artifact entries must not be symlinks: {entry}.")
        if entry.is_dir():
            _validate_artifact_tree(entry)
        elif not entry.is_file():
            raise ParameterError(f"Persisted tokenizer artifact entry must be a regular file or directory: {entry}.")


def save_nss_tokenizer(tokenizer: NssTokenizerCore, directory: Path) -> None:
    """Persist native assets and the canonical NSS manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    # ``save_pretrained`` is a genuine Transformers boundary.  The NSS manifest
    # remains separate so metadata JSON never serializes a live tokenizer.
    tokenizer.for_hf().save_pretrained(str(directory))

    manifest = _manifest_path(directory)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=directory,
            prefix=f".{manifest.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(tokenizer.spec.canonical_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_nss_tokenizer(
    directory: Path,
    *,
    allow_legacy: bool = False,
    admit_remote_code: bool = False,
    registry: NssTokenizerRegistry | None = None,
) -> NssTokenizerCore | None:
    """Reconstruct a persisted NSS tokenizer or explicitly admit legacy absence."""
    directory = Path(directory)
    manifest = _manifest_path(directory)
    manifest_entry_exists = _entry_exists(manifest)
    if manifest_entry_exists and (manifest.is_symlink() or not manifest.is_file()):
        raise ParameterError(f"{NSS_TOKENIZER_MANIFEST} must be a regular file and not a symlink.")
    if not manifest_entry_exists:
        if any(_entry_exists(directory / name) for name in _NATIVE_ASSET_MARKERS):
            raise ParameterError(
                f"NSS tokenizer manifest is missing beside persisted native tokenizer assets: {manifest}."
            )
        if allow_legacy:
            return None
        raise ParameterError(f"NSS tokenizer manifest is missing: {manifest}.")

    try:
        raw_manifest = manifest.read_bytes()
        spec = NssTokenizerSpec.from_json_bytes(raw_manifest)
    except (OSError, ParameterError) as exc:
        raise ParameterError(f"Invalid {NSS_TOKENIZER_MANIFEST}: {exc}") from exc
    if raw_manifest != spec.canonical_bytes():
        raise ParameterError(f"{NSS_TOKENIZER_MANIFEST} must use canonical JSON serialization.")

    _validate_artifact_tree(directory)
    for name in _NATIVE_ASSET_MARKERS:
        native_asset = directory / name
        if _entry_exists(native_asset):
            _require_regular_file(native_asset, description="Persisted native tokenizer asset")
    tokenizer_config = directory / "tokenizer_config.json"
    if not _entry_exists(tokenizer_config):
        raise ParameterError(f"Persisted native tokenizer assets are missing beneath {directory}.")
    _require_regular_file(
        tokenizer_config,
        description="Persisted tokenizer configuration",
    )

    def load_local_native(
        source: str,
        revision: str,
        trust_remote_code: bool,
    ) -> PreTrainedTokenizerBase:
        del source, revision
        try:
            native = AutoTokenizer.from_pretrained(
                str(directory),
                local_files_only=True,
                trust_remote_code=trust_remote_code,
            )
        except (OSError, ValueError) as exc:
            raise ParameterError(f"Could not load persisted native tokenizer assets beneath {directory}.") from exc
        if not isinstance(native, PreTrainedTokenizerBase):
            raise ParameterError("Persisted native tokenizer assets resolved to an unsupported object.")
        return native

    active_registry = builtin_registry() if registry is None else registry
    return active_registry.reconstruct(
        spec,
        native_loader=load_local_native,
        admit_remote_code=admit_remote_code,
    )
