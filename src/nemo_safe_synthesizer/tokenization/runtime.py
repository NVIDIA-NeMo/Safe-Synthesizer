# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline runtime construction for registered built-in NSS tokenizers."""

from __future__ import annotations

import copy
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

from transformers import PreTrainedTokenizerBase

from ..errors import ParameterError
from ..llm.utils import ModelRef
from .base import NssTokenizer
from .registry import builtin_registry
from .tabular import TabularNssTokenizer
from .timeseries import TimeSeriesNssTokenizer
from .types import FramingPolicy, WorkloadKind

if TYPE_CHECKING:
    from ..llm.metadata import ModelMetadata

_IMMUTABLE_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def _snapshot_commit(path: Path) -> str | None:
    resolved = path.resolve()
    for parent in (resolved, *resolved.parents):
        if parent.parent.name == "snapshots" and _IMMUTABLE_COMMIT.fullmatch(parent.name):
            return parent.name
    return None


def resolve_native_provenance(
    source: str | Path,
    *,
    revision: str | None = None,
) -> tuple[str, str, bool]:
    """Resolve deterministic local or immutable cached/remote provenance offline."""
    if not isinstance(source, (str, Path)) or not str(source):
        raise ParameterError("Native tokenizer provenance requires a nonempty model source.")
    source_text = str(source)
    source_path = Path(source_text)
    if source_path.exists():
        resolved = source_path.resolve()
        model_ref = ModelRef.parse(resolved)
        commit = _snapshot_commit(resolved)
        if commit is not None:
            return model_ref.repo_id or str(resolved), commit, model_ref.trust_remote_code
        local_revision = "local-path-" + hashlib.sha256(str(resolved).encode()).hexdigest()
        return str(resolved), local_revision, model_ref.trust_remote_code

    model_ref = ModelRef.parse(source_text, revision=revision or "main")
    snapshot = model_ref.partial_cached_snapshot()
    if snapshot is not None and (commit := _snapshot_commit(snapshot)) is not None:
        return model_ref.repo_id or source_text, commit, model_ref.trust_remote_code
    if revision is not None and _IMMUTABLE_COMMIT.fullmatch(revision):
        return source_text, revision, model_ref.trust_remote_code
    raise ParameterError(
        f"Native tokenizer source {source_text!r} has no resolved immutable commit or admitted local path."
    )


def _framing_policy(native: PreTrainedTokenizerBase, metadata: ModelMetadata) -> FramingPolicy:
    bos_token = native.bos_token or metadata.prompt_config.bos_token
    bos_token_id = native.bos_token_id
    if bos_token_id is None:
        converted_bos = native.convert_tokens_to_ids(bos_token)
        bos_token_id = converted_bos if isinstance(converted_bos, int) else None
    if bos_token_id is None and "<|im_start|>" in native.get_added_vocab():
        bos_token = "<|im_start|>"
        converted_bos = native.convert_tokens_to_ids(bos_token)
        bos_token_id = converted_bos if isinstance(converted_bos, int) else None
    eos_token = native.eos_token
    eos_token_id = native.eos_token_id
    pad_token = native.pad_token or eos_token
    pad_token_id = native.pad_token_id if native.pad_token_id is not None else eos_token_id
    if not isinstance(bos_token, str) or not isinstance(eos_token, str) or not isinstance(pad_token, str):
        raise ParameterError("Runtime NSS construction requires native BOS, EOS, and pad token strings.")
    return FramingPolicy(
        prompt_template=metadata.prompt_config.template,
        add_bos_token_to_prompt=metadata.prompt_config.add_bos_token_to_prompt,
        add_eos_token_to_prompt=metadata.prompt_config.add_eos_token_to_prompt,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
    )


def _record_native_handle(native: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Return a genuine nonmutating native handle with deterministic padding."""
    if native.pad_token is not None and native.pad_token_id is not None:
        return native
    if native.eos_token is None or native.eos_token_id is None:
        raise ParameterError("Runtime NSS construction requires a native EOS token for record padding.")
    try:
        record_native = copy.deepcopy(native)
        record_native.pad_token = record_native.eos_token
    except Exception as exc:
        raise ParameterError("Runtime NSS construction could not derive a nonmutating record pad token.") from exc
    if record_native.pad_token is None or record_native.pad_token_id is None:
        raise ParameterError("Runtime NSS construction could not derive a nonmutating record pad token.")
    return record_native


def create_runtime_nss_tokenizer(
    native: PreTrainedTokenizerBase,
    metadata: ModelMetadata,
    *,
    workload_kind: WorkloadKind,
    native_revision: str | None = None,
) -> NssTokenizer:
    """Construct the registered built-in tokenizer from existing runtime values."""
    if not isinstance(native, PreTrainedTokenizerBase):
        raise ParameterError("Runtime NSS construction requires a native Hugging Face tokenizer.")
    declared_source = metadata.model_name_or_path
    native_source = getattr(native, "name_or_path", None)
    if not isinstance(native_source, str) or not native_source:
        raise ParameterError("Native tokenizer does not expose trustworthy load provenance.")
    native_path = Path(native_source)
    declared_path = Path(declared_source)
    if native_path.exists():
        if declared_path.exists():
            if native_path.resolve() != declared_path.resolve():
                raise ParameterError(
                    f"Native tokenizer source {native_source!r} does not match declared source {declared_source!r}."
                )
            source, revision, trust_remote_code = resolve_native_provenance(native_path)
        else:
            native_ref = ModelRef.parse(native_path)
            declared_ref = ModelRef.parse(declared_source)
            commit = _snapshot_commit(native_path)
            declared_snapshot = declared_ref.local_path
            if (
                commit is None
                or native_ref.repo_id is None
                or native_ref.repo_id != declared_ref.repo_id
                or declared_snapshot is None
                or declared_snapshot.resolve() != native_path.resolve()
                or (native_revision is not None and native_revision != commit)
            ):
                raise ParameterError(
                    f"Native tokenizer snapshot {native_source!r} does not match declared remote source "
                    f"{declared_source!r} at an immutable commit."
                )
            source = native_ref.repo_id
            revision = commit
            trust_remote_code = native_ref.trust_remote_code
    else:
        native_commit = getattr(native, "_commit_hash", None)
        if native_commit is None:
            init_kwargs = getattr(native, "init_kwargs", None)
            native_commit = init_kwargs.get("_commit_hash") if isinstance(init_kwargs, dict) else None
        if not isinstance(native_commit, str) or _IMMUTABLE_COMMIT.fullmatch(native_commit) is None:
            raise ParameterError("Remote native tokenizer provenance has no trustworthy immutable commit.")
        if native_source != declared_source:
            raise ParameterError(
                f"Native tokenizer source {native_source!r} does not match declared source {declared_source!r}."
            )
        if native_revision is not None and native_revision != native_commit:
            raise ParameterError("Native tokenizer immutable commit does not match the requested revision.")
        source, revision, trust_remote_code = resolve_native_provenance(
            declared_source,
            revision=native_commit,
        )
    implementation_id = {
        WorkloadKind.TABULAR: TabularNssTokenizer.IMPLEMENTATION_ID,
        WorkloadKind.TIME_SERIES: TimeSeriesNssTokenizer.IMPLEMENTATION_ID,
    }.get(workload_kind)
    if implementation_id is None:
        raise ParameterError(f"No built-in NSS tokenizer is registered for workload {workload_kind!r}.")
    record_native = _record_native_handle(native)
    tokenizer = builtin_registry().create(
        (NssTokenizer.API_VERSION, implementation_id, "1"),
        record_native,
        framing=_framing_policy(record_native, metadata),
        native_source=source,
        native_revision=revision,
        native_trust_remote_code=trust_remote_code,
    )
    if not tokenizer.capabilities.record_jsonl:
        raise ParameterError("Selected NSS tokenizer does not declare ordered record JSONL capability.")
    return cast(NssTokenizer, tokenizer)
