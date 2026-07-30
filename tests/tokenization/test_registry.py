# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1 registry, persistence, reconstruction, and parity contracts."""

from __future__ import annotations

import base64
import copy
import multiprocessing
from dataclasses import replace
from typing import cast, get_type_hints

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.tokenization import (
    FramingPolicy,
    NssTokenizer,
    NssTokenizerRegistry,
    NssTokenizerSpec,
    RegistryEntry,
    TabularContext,
    TabularNssTokenizer,
    TimeSeriesNssTokenizer,
    TokenizerCapabilities,
    WorkloadKind,
    builtin_registry,
)
from nemo_safe_synthesizer.tokenization.base import native_snapshot
from nemo_safe_synthesizer.tokenization.types import JsonObject, canonical_json_bytes


def _make_tokenizer(tokenizers_dir, implementation=TabularNssTokenizer):
    native = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizers_dir / "tinyllama", local_files_only=True),
    )
    policy = FramingPolicy(
        prompt_template="{instruction}|{schema}|{prefill}",
        add_bos_token_to_prompt=True,
        add_eos_token_to_prompt=True,
        bos_token_id=native.bos_token_id,
        eos_token_id=native.eos_token_id,
        pad_token_id=native.pad_token_id,
        bos_token=cast(str, native.bos_token),
        eos_token=cast(str, native.eos_token),
        pad_token=cast(str, native.pad_token),
    )
    registry = builtin_registry()
    tokenizer = registry.create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )
    return registry, tokenizer


def test_spec_round_trip_canonical_schema_and_cache_identity(tokenizers_dir) -> None:
    _, tokenizer = _make_tokenizer(tokenizers_dir)

    manifest = tokenizer.spec.canonical_bytes()
    restored = NssTokenizerSpec.from_json_bytes(manifest)

    assert restored == tokenizer.spec
    assert restored.canonical_bytes() == manifest
    assert canonical_json_bytes(restored.to_dict()) == manifest
    assert restored.cache_identity_fragment == tokenizer.spec.cache_identity_fragment


@pytest.mark.parametrize("implementation", [TabularNssTokenizer, TimeSeriesNssTokenizer])
def test_spawned_worker_reconstruction(tokenizers_dir, tmp_path, implementation) -> None:
    _, tokenizer = _make_tokenizer(tokenizers_dir, implementation)
    context = multiprocessing.get_context("spawn")
    result_path = tmp_path / "worker-result"
    encoded = base64.b64encode(tokenizer.spec.canonical_bytes()).decode()
    code = (
        "import base64;"
        "from nemo_safe_synthesizer.tokenization import NssTokenizerSpec,builtin_registry;"
        f"s=NssTokenizerSpec.from_json_bytes(base64.b64decode({encoded!r}));"
        "t=builtin_registry().reconstruct(s);"
        f"open({str(result_path)!r},'w').write(t.spec.cache_identity_fragment+'|'+type(t).__name__)"
    )
    process = context.Process(target=exec, args=(code,))
    process.start()
    process.join(timeout=30)

    assert process.exitcode == 0
    assert result_path.read_text() == f"{tokenizer.spec.cache_identity_fragment}|{implementation.__name__}"


class _ExternalA(TabularNssTokenizer):
    IMPLEMENTATION_ID = "example.org:a"
    IMPLEMENTATION_VERSION = "1"


class _ExternalZ(TabularNssTokenizer):
    IMPLEMENTATION_ID = "example.org:z"
    IMPLEMENTATION_VERSION = "2"


class _ConfigurableExternal(TabularNssTokenizer):
    IMPLEMENTATION_ID = "example.org:configurable"

    @classmethod
    def _default_workload_payload(cls) -> JsonObject:
        return {"payload_version": 1, "mode": "default"}

    @classmethod
    def _validate_workload_payload(cls, payload: object) -> JsonObject:
        payload_dict = cast(dict[str, object], payload) if isinstance(payload, dict) else {}
        if (
            not isinstance(payload, dict)
            or payload_dict.get("payload_version") != 1
            or payload_dict.get("mode") not in {"default", "alternate"}
            or set(payload_dict) != {"payload_version", "mode"}
        ):
            raise ParameterError("Invalid configurable workload payload.")
        return {"payload_version": 1, "mode": cast(str, payload_dict["mode"])}


class _AliasingExternal(_ConfigurableExternal):
    IMPLEMENTATION_ID = "example.org:aliasing"

    @classmethod
    def _validate_workload_payload(cls, payload: object) -> JsonObject:
        if not isinstance(payload, dict):
            raise ParameterError("Invalid aliasing workload payload.")
        return cast(JsonObject, payload)


class _DirectExternal(NssTokenizer[TabularContext]):
    IMPLEMENTATION_ID = "example.org:direct"
    IMPLEMENTATION_VERSION = "1"
    WORKLOAD_KIND = WorkloadKind.TABULAR

    @property
    def capabilities(self) -> TokenizerCapabilities:
        return TokenizerCapabilities(record_jsonl=True, prompt_encoding=True, rolling_prefill=False)

    def _prompt_parts(self, context: TabularContext) -> tuple[str, str, str]:
        return context.instruction, ",".join(context.ordered_columns), ""

    @classmethod
    def _default_workload_payload(cls) -> JsonObject:
        return {"payload_version": 1}

    @classmethod
    def _validate_workload_payload(cls, payload: object) -> JsonObject:
        if payload != {"payload_version": 1}:
            raise ParameterError("Invalid direct external workload payload.")
        return {"payload_version": 1}


def test_registry_digest_is_sorted_and_provider_normalized() -> None:
    a = RegistryEntry((1, "example.org:z", "2"), "Example_Pkg", "1.0", _ExternalZ)
    b = RegistryEntry((1, "example.org:a", "1"), "example.pkg", "1.0", _ExternalA)
    left = NssTokenizerRegistry().register(a, admit_external=True).register(b, admit_external=True)
    right = NssTokenizerRegistry().register(b, admit_external=True).register(a, admit_external=True)

    assert left.digest == right.digest
    assert [entry.key for entry in left.entries] == sorted((a.key, b.key))


def test_direct_registry_construction_rejects_duplicates_and_unsorted_entries() -> None:
    a = RegistryEntry((1, "example.org:a", "1"), "pkg", "1", _ExternalA)
    z = RegistryEntry((1, "example.org:z", "2"), "pkg", "1", _ExternalZ)

    with pytest.raises(ParameterError, match="Duplicate exact"):
        NssTokenizerRegistry((a, a))
    with pytest.raises(ParameterError, match="sorted"):
        NssTokenizerRegistry((z, a))


def test_registry_entry_rejects_non_nss_factory() -> None:
    class NotAnNssTokenizer:
        API_VERSION = 1
        IMPLEMENTATION_ID = "example.org:not-nss"
        IMPLEMENTATION_VERSION = "1"

    with pytest.raises(ParameterError, match="inherit NssTokenizer"):
        RegistryEntry(
            (1, NotAnNssTokenizer.IMPLEMENTATION_ID, "1"),
            "example-extension",
            "1",
            cast(type[NssTokenizer], NotAnNssTokenizer),
        )


def test_external_implementation_uses_typed_hooks_and_base_owned_invariants(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)
    external_registry = registry.register(
        RegistryEntry((1, _DirectExternal.IMPLEMENTATION_ID, "1"), "example-extension", "1", _DirectExternal),
        admit_external=True,
    )

    external = external_registry.create(
        (1, _DirectExternal.IMPLEMENTATION_ID, "1"),
        tokenizer.for_hf(),
        framing=tokenizer._framing,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )

    assert external.render_prompt(TabularContext(("a", "b"), "generate")).text == "generate|a,b|"
    assert get_type_hints(_DirectExternal._prompt_parts)["context"] is TabularContext
    for method_name in (
        "__init__",
        "for_hf",
        "encode_no_special",
        "batch_encode_no_special",
        "render_prompt",
        "encode_records",
        "frame_training",
        "compare_engine",
        "_validate_binding",
        "_serialized_payload",
        "policies_from_spec",
    ):
        assert getattr(getattr(NssTokenizer, method_name), "__final__", False) is True
    assert NssTokenizer.spec.fget is not None
    assert getattr(NssTokenizer.spec.fget, "__final__", False) is True


def test_duplicate_unknown_version_and_external_admission_fail_closed() -> None:
    registry = builtin_registry()
    duplicate = registry.entries[0]
    with pytest.raises(ParameterError, match="Duplicate"):
        registry.register(duplicate, admit_external=True)
    with pytest.raises(ParameterError, match="Unknown"):
        registry.resolve((1, "example.org:missing", "1"))
    with pytest.raises(ParameterError, match="Unsupported"):
        registry.resolve((1, TabularNssTokenizer.IMPLEMENTATION_ID, "99"))
    external = RegistryEntry((1, "example.org:a", "1"), "pkg", "1", _ExternalA)
    with pytest.raises(ParameterError, match="explicit admission"):
        registry.register(external)
    with pytest.raises(ParameterError, match="does not match factory key"):
        RegistryEntry((1, "example.org:wrong", "9"), "pkg", "1", _ExternalA)


def test_registry_drift_and_native_mismatch_fail_closed(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)
    drifted = replace(tokenizer.spec, registry_digest="0" * 64)
    mismatch = replace(tokenizer.spec, native_fingerprint="0" * 64)

    def loader(source: str, _revision: str, _trust_remote_code: bool):
        return cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(source, local_files_only=True),
        )

    with pytest.raises(ParameterError, match="registry drift"):
        registry.reconstruct(drifted, native_loader=loader)
    with pytest.raises(ParameterError, match="does not match"):
        registry.reconstruct(mismatch, native_loader=loader)


def test_unprobed_native_behavior_change_alters_fingerprint(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)
    native = tokenizer.for_hf()
    original_encode = native.encode

    def altered_encode(text, *args, **kwargs):
        if text == "outside persisted probes":
            return [12345]
        return original_encode(text, *args, **kwargs)

    setattr(native, "encode", altered_encode)

    def loader(_source: str, _revision: str, _trust_remote_code: bool):
        return native

    with pytest.raises(ParameterError, match="does not match"):
        registry.reconstruct(tokenizer.spec, native_loader=loader)


def test_real_transformers_special_token_splitting_change_fails_reconstruction(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)
    before = tokenizer.encode_no_special("x<s>y")

    def loader(source: str, _revision: str, _trust_remote_code: bool):
        native = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(source, local_files_only=True),
        )
        native.split_special_tokens = True
        assert tuple(native.encode("x<s>y", add_special_tokens=False)) != before
        return native

    with pytest.raises(ParameterError, match="does not match"):
        registry.reconstruct(tokenizer.spec, native_loader=loader)


def test_remote_code_manifest_requires_runtime_admission_before_loading(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)
    remote = replace(tokenizer.spec, native_trust_remote_code=True)
    called = False

    def loader(_source: str, _revision: str, _trust_remote_code: bool):
        nonlocal called
        called = True
        raise AssertionError("loader must not run")

    with pytest.raises(ParameterError, match="explicit reconstruction-time admission"):
        registry.reconstruct(remote, native_loader=loader)
    assert called is False


def test_workload_kind_mismatch_fails_before_native_loading(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)
    mismatched = replace(tokenizer.spec, workload_kind="time-series")
    called = False

    def loader(_source: str, _revision: str, _trust_remote_code: bool):
        nonlocal called
        called = True
        raise AssertionError("loader must not run")

    with pytest.raises(ParameterError, match="workload kind"):
        registry.reconstruct(mismatched, native_loader=loader)
    assert called is False


def test_manifest_schema_is_strict_and_canonical_json_rejects_nonfinite(tokenizers_dir) -> None:
    _, tokenizer = _make_tokenizer(tokenizers_dir)
    manifest = tokenizer.spec.to_dict()
    malformed = copy.deepcopy(manifest)
    malformed["spec_version"] = "1"
    with pytest.raises(ParameterError, match="schema"):
        NssTokenizerSpec.from_dict(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["unexpected"] = True
    with pytest.raises(ParameterError, match="schema"):
        NssTokenizerSpec.from_dict(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["native"]["unexpected"] = True
    with pytest.raises(ParameterError, match="schema"):
        NssTokenizerSpec.from_dict(malformed)
    malformed = copy.deepcopy(manifest)
    malformed["native"]["fingerprint"] = "z" * 64
    with pytest.raises(ParameterError, match="schema"):
        NssTokenizerSpec.from_dict(malformed)
    with pytest.raises(ParameterError, match="finite JSON"):
        canonical_json_bytes({"value": float("nan")})


def test_direct_construction_rejects_mutable_remote_revision(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)

    with pytest.raises(ParameterError, match="immutable"):
        registry.create(
            tokenizer.spec.key,
            tokenizer.for_hf(),
            framing=tokenizer._framing,
            native_source="nvidia/example-tokenizer",
            native_revision="main",
        )


@pytest.mark.parametrize("revision", ["main", "latest", "refs/heads/main"])
def test_manifest_rejects_mutable_remote_revision(tokenizers_dir, revision: str) -> None:
    _, tokenizer = _make_tokenizer(tokenizers_dir)
    manifest = tokenizer.spec.to_dict()
    manifest["native"]["source"] = "nvidia/example-tokenizer"
    manifest["native"]["revision"] = revision

    with pytest.raises(ParameterError, match="schema"):
        NssTokenizerSpec.from_dict(manifest)


def test_reconstruction_rejects_mutable_remote_revision_before_loader(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)
    remote = replace(
        tokenizer.spec,
        native_source="nvidia/example-tokenizer",
        native_revision="a" * 40,
    )
    object.__setattr__(remote, "native_revision", "main")
    called = False

    def loader(_source: str, _revision: str, _trust_remote_code: bool):
        nonlocal called
        called = True
        raise AssertionError("loader must not run")

    with pytest.raises(ParameterError, match="immutable"):
        registry.reconstruct(remote, native_loader=loader)
    assert called is False


def test_local_source_accepts_opaque_fixture_revision(tokenizers_dir) -> None:
    _, tokenizer = _make_tokenizer(tokenizers_dir)

    assert tokenizer.spec.native_revision == "fixture-v1"
    assert NssTokenizerSpec.from_json_bytes(tokenizer.spec.canonical_bytes()) == tokenizer.spec


def test_remote_source_accepts_resolved_immutable_revision(tokenizers_dir) -> None:
    registry, tokenizer = _make_tokenizer(tokenizers_dir)
    revision = "0123456789abcdef0123456789abcdef01234567"  # pragma: allowlist secret

    remote = registry.create(
        tokenizer.spec.key,
        tokenizer.for_hf(),
        framing=tokenizer._framing,
        native_source="nvidia/example-tokenizer",
        native_revision=revision,
    )

    assert remote.spec.native_revision == revision


def test_external_workload_payload_round_trips_through_factory_hook(tokenizers_dir) -> None:
    native = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizers_dir / "tinyllama", local_files_only=True),
    )
    policy = FramingPolicy(
        prompt_template="{instruction}|{schema}|{prefill}",
        add_bos_token_to_prompt=True,
        add_eos_token_to_prompt=True,
        bos_token_id=native.bos_token_id,
        eos_token_id=native.eos_token_id,
        pad_token_id=native.pad_token_id,
        bos_token=cast(str, native.bos_token),
        eos_token=cast(str, native.eos_token),
        pad_token=cast(str, native.pad_token),
    )
    registry = builtin_registry().register(
        RegistryEntry(
            (1, _ConfigurableExternal.IMPLEMENTATION_ID, "1"),
            "example-extension",
            "1.0",
            _ConfigurableExternal,
        ),
        admit_external=True,
    )
    tokenizer = registry.create(
        (1, _ConfigurableExternal.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
        workload_payload={"payload_version": 1, "mode": "alternate"},
    )

    def loader(source: str, _revision: str, _trust_remote_code: bool):
        return cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(source, local_files_only=True),
        )

    reconstructed = registry.reconstruct(tokenizer.spec, native_loader=loader)

    assert reconstructed.spec == tokenizer.spec


def test_external_workload_payload_is_deeply_detached_from_caller_alias(tokenizers_dir) -> None:
    native = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizers_dir / "tinyllama", local_files_only=True),
    )
    policy = FramingPolicy(
        prompt_template="{instruction}|{schema}|{prefill}",
        add_bos_token_to_prompt=True,
        add_eos_token_to_prompt=True,
        bos_token_id=native.bos_token_id,
        eos_token_id=native.eos_token_id,
        pad_token_id=native.pad_token_id,
        bos_token=cast(str, native.bos_token),
        eos_token=cast(str, native.eos_token),
        pad_token=cast(str, native.pad_token),
    )
    registry = builtin_registry().register(
        RegistryEntry((1, _AliasingExternal.IMPLEMENTATION_ID, "1"), "aliasing", "1", _AliasingExternal),
        admit_external=True,
    )
    payload: JsonObject = {"payload_version": 1, "nested": {"mode": "before"}}
    tokenizer = registry.create(
        (1, _AliasingExternal.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
        workload_payload=payload,
    )
    before = tokenizer.spec

    nested = cast(dict[str, JsonObject | str | int], payload["nested"])
    nested["mode"] = "after"

    assert tokenizer.spec == before
    assert b"before" in tokenizer.spec.canonical_bytes()
    assert b"after" not in tokenizer.spec.canonical_bytes()
    assert b"before" in canonical_json_bytes(tokenizer._serialized_payload())
    assert b"after" not in canonical_json_bytes(tokenizer._serialized_payload())


class _SnapshotEngine:
    def __init__(
        self,
        native,
        *,
        vocab_mismatch: bool = False,
        special_mismatch: bool = False,
        probe_mismatch: bool = False,
        decode_mismatch: bool = False,
    ) -> None:
        snapshot = native_snapshot(native)
        self.class_name = snapshot.class_name
        self.vocab_size = snapshot.vocab_size + int(vocab_mismatch)
        self.total_size = snapshot.total_size
        self.added_vocabulary = snapshot.added_vocabulary
        self.special_token_ids = (
            (*snapshot.special_token_ids[:-1], ("additional_special_tokens_ids", (999,)))
            if special_mismatch
            else snapshot.special_token_ids
        )
        self._native = native
        self._probe_mismatch = probe_mismatch
        self._decode_mismatch = decode_mismatch

    def encode_no_special(self, text: str):
        ids = self._native.encode(text, add_special_tokens=False)
        return [*ids, 0] if self._probe_mismatch and text == "NSS tokenizer probe" else ids

    def decode(self, input_ids):
        decoded = self._native.decode(input_ids)
        return f"{decoded}!" if self._decode_mismatch else decoded


def test_pure_engine_parity_matching_and_mismatching_doubles(tokenizers_dir) -> None:
    _, tokenizer = _make_tokenizer(tokenizers_dir)
    native = tokenizer.for_hf()

    matching = tokenizer.compare_engine(_SnapshotEngine(native))
    vocab_mismatch = tokenizer.compare_engine(_SnapshotEngine(native, vocab_mismatch=True))
    special_mismatch = tokenizer.compare_engine(_SnapshotEngine(native, special_mismatch=True))
    probe_mismatch = tokenizer.compare_engine(_SnapshotEngine(native, probe_mismatch=True))
    decode_mismatch = tokenizer.compare_engine(_SnapshotEngine(native, decode_mismatch=True))

    assert matching.matches is True
    assert matching.mismatches == ()
    assert vocab_mismatch.mismatches == ("vocab_size",)
    assert special_mismatch.mismatches == ("special_token_ids",)
    assert probe_mismatch.mismatches == ("probe[1].input_ids",)
    assert decode_mismatch.mismatches == tuple(
        f"probe[{index}].decoded" for index, _ in enumerate(native_snapshot(native).probes)
    )


def test_engine_parity_ignores_adapter_wrapper_class(tokenizers_dir) -> None:
    _, tokenizer = _make_tokenizer(tokenizers_dir)
    engine = _SnapshotEngine(tokenizer.for_hf())
    engine.class_name = "engine.adapters.DifferentTokenizerWrapper"

    parity = tokenizer.compare_engine(engine)

    assert parity.matches is True
    assert parity.mismatches == ()
