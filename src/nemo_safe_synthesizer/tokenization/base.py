# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NSS tokenizer abstract base and invariant-owning template methods."""

from __future__ import annotations

import hashlib
import inspect
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, cast, final

from transformers import PreTrainedTokenizerBase

from ..errors import ParameterError
from .spec import NssTokenizerSpec, PolicyEpochs
from .types import (
    FramingPolicy,
    JsonObject,
    JsonValue,
    NativeTokenizerSnapshot,
    PaddedTokenBatch,
    PromptEncoding,
    RecordBatch,
    RecordEncoding,
    TokenBatch,
    TokenizerCapabilities,
    TokenizerProbe,
    TrainingEncoding,
    WorkloadContext,
    WorkloadKind,
    canonical_json_bytes,
)

_BASE_PROBE_TEXTS = ("", "NSS tokenizer probe", '{"line":"a\\nb","unicode":"\u0085\u2028\u2029"}\n')
_IGNORE_LABEL = -100
ContextT = TypeVar("ContextT", bound=WorkloadContext)


class EngineTokenizerProbe(Protocol):
    """Small adapter protocol used for pure engine-tokenizer parity checks."""

    @property
    def vocab_size(self) -> int: ...

    @property
    def total_size(self) -> int: ...

    @property
    def added_vocabulary(self) -> Sequence[tuple[str, int]]: ...

    @property
    def special_token_ids(self) -> Sequence[tuple[str, int | tuple[int, ...] | None]]: ...

    def encode_no_special(self, text: str) -> Sequence[int]: ...

    def decode(self, input_ids: Sequence[int]) -> str: ...


@dataclass(frozen=True, slots=True)
class EngineParity:
    """Result of a pure native-versus-engine tokenizer comparison."""

    matches: bool
    mismatches: tuple[str, ...]


def _token_id_value(value: object) -> int | tuple[int, ...] | None:
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, list) and all(isinstance(item, int) for item in value):
        return tuple(cast(list[int], value))
    raise ParameterError("Native tokenizer exposed an invalid special token ID.")


def native_snapshot(native: PreTrainedTokenizerBase) -> NativeTokenizerSnapshot:
    """Capture deterministic native class, vocabulary, specials, and probes."""
    class_name = f"{type(native).__module__}.{type(native).__qualname__}"
    vocab_size = int(native.vocab_size)
    total_size = len(native)
    vocabulary = tuple(sorted((str(token), int(token_id)) for token, token_id in native.get_vocab().items()))
    added_vocabulary = tuple(
        sorted((str(token), int(token_id)) for token, token_id in native.get_added_vocab().items())
    )
    special_token_ids = tuple(
        (name, _token_id_value(getattr(native, name, None)))
        for name in (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "unk_token_id",
            "sep_token_id",
            "cls_token_id",
            "mask_token_id",
            "additional_special_tokens_ids",
        )
    )
    probe_texts = (*_BASE_PROBE_TEXTS, *(f"x{token}y" for token in dict.fromkeys(native.all_special_tokens)))
    probes = tuple(
        TokenizerProbe(
            text=text,
            input_ids=tuple(native.encode(text, add_special_tokens=False)),
            decoded=cast(str, native.decode(native.encode(text, add_special_tokens=False))),
        )
        for text in probe_texts
    )
    vocabulary_digest = hashlib.sha256(
        canonical_json_bytes(cast(JsonValue, [[token, token_id] for token, token_id in vocabulary]))
    ).hexdigest()
    pipeline_digest = _native_pipeline_digest(native)
    fingerprint_input: JsonObject = {
        "class": class_name,
        "vocab_size": vocab_size,
        "total_size": total_size,
        "vocabulary_digest": vocabulary_digest,
        "pipeline_digest": pipeline_digest,
        "added_vocabulary": [[token, token_id] for token, token_id in added_vocabulary],
        "special_token_ids": [
            [name, list(value) if isinstance(value, tuple) else value] for name, value in special_token_ids
        ],
        "probes": [
            {"text": probe.text, "input_ids": list(probe.input_ids), "decoded": probe.decoded} for probe in probes
        ],
    }
    fingerprint = hashlib.sha256(canonical_json_bytes(cast(JsonValue, fingerprint_input))).hexdigest()
    return NativeTokenizerSnapshot(
        class_name=class_name,
        vocab_size=vocab_size,
        total_size=total_size,
        added_vocabulary=added_vocabulary,
        special_token_ids=special_token_ids,
        probes=probes,
        fingerprint=fingerprint,
    )


def _native_pipeline_digest(native: PreTrainedTokenizerBase) -> str:
    """Hash available native tokenizer model and pipeline state."""
    pipeline_parts: list[bytes] = []
    backend = getattr(native, "backend_tokenizer", None)
    if backend is not None and callable(getattr(backend, "to_str", None)):
        pipeline_parts.append(backend.to_str().encode())
    sentencepiece = getattr(native, "sp_model", None)
    if sentencepiece is not None and callable(getattr(sentencepiece, "serialized_model_proto", None)):
        pipeline_parts.append(sentencepiece.serialized_model_proto())
    for method_name in ("encode", "decode", "convert_tokens_to_ids"):
        method = getattr(native, method_name)
        function = getattr(method, "__func__", method)
        code = getattr(function, "__code__", None)
        identity = (
            f"{getattr(function, '__module__', '')}.{getattr(function, '__qualname__', '')}"
            if code is None
            else f"{code.co_code.hex()}|{code.co_names!r}|{code.co_consts!r}"
        )
        pipeline_parts.append(identity.encode())
    try:
        pipeline_parts.append(inspect.getsource(type(native)).encode())
    except (OSError, TypeError):
        pipeline_parts.append(f"{type(native).__module__}.{type(native).__qualname__}".encode())
    policy: JsonObject = {
        "padding_side": native.padding_side,
        "truncation_side": native.truncation_side,
        "clean_up_tokenization_spaces": native.clean_up_tokenization_spaces,
        "split_special_tokens": native.split_special_tokens,
        "add_prefix_space": cast(JsonValue, getattr(native, "add_prefix_space", None)),
        "do_lower_case": cast(JsonValue, getattr(native, "do_lower_case", None)),
        "legacy": cast(JsonValue, getattr(native, "legacy", None)),
    }
    pipeline_parts.append(canonical_json_bytes(cast(JsonValue, policy)))
    return hashlib.sha256(b"\0".join(pipeline_parts)).hexdigest()


class NssTokenizer(ABC, Generic[ContextT]):
    """Compose one native tokenizer while owning immutable NSS framing policy."""

    API_VERSION = 1
    SPEC_VERSION = 1
    IMPLEMENTATION_ID: str
    IMPLEMENTATION_VERSION = "1"
    WORKLOAD_KIND: WorkloadKind

    @final
    def __init__(
        self,
        native: PreTrainedTokenizerBase,
        *,
        framing: FramingPolicy,
        native_source: str,
        native_revision: str,
        native_trust_remote_code: bool,
        registry_digest: str,
        policy_epochs: PolicyEpochs | None = None,
        persisted_spec: NssTokenizerSpec | None = None,
        workload_payload: object | None = None,
    ) -> None:
        if not isinstance(native, PreTrainedTokenizerBase):
            raise ParameterError("NSS tokenizers require a PreTrainedTokenizerBase native handle.")
        self._native = native
        self._framing = framing
        raw_workload_payload = self._default_workload_payload() if workload_payload is None else workload_payload
        normalized_workload = self._validate_workload_payload(raw_workload_payload)
        self._workload_policy_json = canonical_json_bytes(cast(JsonValue, normalized_workload)).decode()
        self._validate_binding()
        snapshot = native_snapshot(native)
        payload = self._serialized_payload()
        candidate = NssTokenizerSpec(
            spec_version=self.SPEC_VERSION,
            api_version=self.API_VERSION,
            workload_kind=self.WORKLOAD_KIND,
            implementation_id=self.IMPLEMENTATION_ID,
            implementation_version=self.IMPLEMENTATION_VERSION,
            implementation_payload=canonical_json_bytes(cast(JsonValue, payload)).decode(),
            native_source=native_source,
            native_revision=native_revision,
            native_trust_remote_code=native_trust_remote_code,
            native_class=snapshot.class_name,
            native_fingerprint=snapshot.fingerprint,
            native_probes=snapshot.probes,
            registry_digest=registry_digest,
            policy_epochs=policy_epochs or PolicyEpochs(),
        )
        if persisted_spec is not None and candidate != persisted_spec:
            raise ParameterError("Native tokenizer or NSS policy does not match the persisted tokenizer specification.")
        self._spec = candidate

    @property
    @final
    def spec(self) -> NssTokenizerSpec:
        """Return the immutable persistent specification."""
        return self._spec

    @property
    @abstractmethod
    def capabilities(self) -> TokenizerCapabilities:
        """Declare workload behavior without concrete-class inspection."""

    @final
    def for_hf(self) -> PreTrainedTokenizerBase:
        """Return the genuine mutable native handle at a framework boundary."""
        return self._native

    @final
    def encode_no_special(self, text: str) -> tuple[int, ...]:
        """Encode text with native special-token insertion disabled."""
        return tuple(self._native.encode(text, add_special_tokens=False))

    @final
    def batch_encode_no_special(self, texts: Sequence[str], *, padding: bool = False) -> TokenBatch | PaddedTokenBatch:
        """Encode text explicitly and optionally right-pad with the NSS pad ID."""
        rows = tuple(self.encode_no_special(text) for text in texts)
        if not padding:
            return TokenBatch(rows, tuple(tuple(1 for _ in row) for row in rows))
        width = max((len(row) for row in rows), default=0)
        pad_id = cast(int, self._framing.pad_token_id)
        return PaddedTokenBatch(
            tuple(row + (pad_id,) * (width - len(row)) for row in rows),
            tuple(tuple(1 for _ in row) + (0,) * (width - len(row)) for row in rows),
        )

    @final
    def render_prompt(self, context: ContextT) -> PromptEncoding:
        """Validate a context and render its exact prompt text and IDs."""
        instruction, schema, prefill = self._prompt_parts(context)
        text = self._framing.prompt_template.format(
            instruction=instruction,
            schema=schema,
            prefill=prefill,
        )
        input_ids = list(self.encode_no_special(text))
        if self._framing.add_bos_token_to_prompt:
            input_ids.insert(0, cast(int, self._framing.bos_token_id))
        if self._framing.add_eos_token_to_prompt:
            input_ids.append(cast(int, self._framing.eos_token_id))
        ids = tuple(input_ids)
        return PromptEncoding(text=text, input_ids=ids, attention_mask=tuple(1 for _ in ids))

    @final
    def encode_records(self, records: Sequence[Mapping[str, JsonValue]]) -> RecordBatch:
        """Serialize ordered mappings as JSONL and encode each record separately."""
        encoded: list[RecordEncoding] = []
        for record in records:
            payload = self._record_payload(record)
            try:
                text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            except (TypeError, ValueError) as exc:
                raise ParameterError("Records must contain finite JSON values.") from exc
            utf8 = text.encode()
            ids = self.encode_no_special(text)
            encoded.append(RecordEncoding(utf8=utf8, input_ids=ids, attention_mask=tuple(1 for _ in ids)))
        return RecordBatch(tuple(encoded))

    @final
    def frame_training(
        self,
        prompt: PromptEncoding,
        sequences: Sequence[Sequence[int]],
        *,
        add_sequence_delimiters: bool = True,
    ) -> TrainingEncoding:
        """Frame prompt and sequence IDs with stable masks and labels."""
        input_ids = list(prompt.input_ids)
        labels = [_IGNORE_LABEL] * len(input_ids)
        for sequence in sequences:
            framed = list(sequence)
            if add_sequence_delimiters:
                if self._framing.bos_token_id is None:
                    raise ParameterError("A BOS token ID is required for sequence delimiters.")
                framed = [self._framing.bos_token_id, *framed, cast(int, self._framing.eos_token_id)]
            input_ids.extend(framed)
            labels.extend(framed)
        return TrainingEncoding(
            input_ids=tuple(input_ids),
            attention_mask=tuple(1 for _ in input_ids),
            labels=tuple(labels),
        )

    @final
    def compare_engine(self, engine: EngineTokenizerProbe) -> EngineParity:
        """Compare a minimal engine probe without importing or mutating vLLM."""
        snapshot = native_snapshot(self._native)
        mismatches: list[str] = []
        for name, native_value, engine_value in (
            ("vocab_size", snapshot.vocab_size, engine.vocab_size),
            ("total_size", snapshot.total_size, engine.total_size),
            ("added_vocabulary", snapshot.added_vocabulary, tuple(engine.added_vocabulary)),
            ("special_token_ids", snapshot.special_token_ids, tuple(engine.special_token_ids)),
        ):
            if native_value != engine_value:
                mismatches.append(name)
        for index, probe in enumerate(snapshot.probes):
            engine_ids = tuple(engine.encode_no_special(probe.text))
            if engine_ids != probe.input_ids:
                mismatches.append(f"probe[{index}].input_ids")
            elif engine.decode(engine_ids) != probe.decoded:
                mismatches.append(f"probe[{index}].decoded")
        return EngineParity(matches=not mismatches, mismatches=tuple(mismatches))

    @final
    def _validate_binding(self) -> None:
        total_size = len(self._native)
        for name, policy_token, policy_id, native_token, native_id in (
            (
                "BOS",
                self._framing.bos_token,
                self._framing.bos_token_id,
                self._native.bos_token,
                self._native.bos_token_id,
            ),
            (
                "EOS",
                self._framing.eos_token,
                self._framing.eos_token_id,
                self._native.eos_token,
                self._native.eos_token_id,
            ),
            (
                "pad",
                self._framing.pad_token,
                self._framing.pad_token_id,
                self._native.pad_token,
                self._native.pad_token_id,
            ),
        ):
            if name != "BOS" and native_id is None:
                raise ParameterError(f"The native tokenizer must declare a {name} token ID.")
            if policy_id is not None and not 0 <= policy_id < total_size:
                raise ParameterError(f"{name} token ID {policy_id} is outside the native vocabulary.")
            if native_id is not None and policy_id != native_id:
                raise ParameterError(f"{name} token ID does not match the native tokenizer.")
            if native_token is not None and policy_token != str(native_token):
                raise ParameterError(f"{name} token string does not match the native tokenizer.")
            converted = self._native.convert_tokens_to_ids(policy_token)
            if not isinstance(converted, int) or converted != policy_id:
                raise ParameterError(f"{name} token string does not resolve to the configured token ID.")

    @final
    def _serialized_payload(self) -> JsonObject:
        framing: JsonObject = {
            "prompt_template": self._framing.prompt_template,
            "add_bos_token_to_prompt": self._framing.add_bos_token_to_prompt,
            "add_eos_token_to_prompt": self._framing.add_eos_token_to_prompt,
            "bos_token_id": self._framing.bos_token_id,
            "eos_token_id": self._framing.eos_token_id,
            "pad_token_id": self._framing.pad_token_id,
            "bos_token": self._framing.bos_token,
            "eos_token": self._framing.eos_token,
            "pad_token": self._framing.pad_token,
        }
        return {
            "payload_version": 1,
            "framing": framing,
            "workload": cast(JsonValue, json.loads(self._workload_policy_json)),
        }

    @classmethod
    @final
    def policies_from_spec(cls, spec: NssTokenizerSpec) -> tuple[FramingPolicy, JsonObject]:
        """Validate and recover shared and workload policy from a spec."""
        try:
            payload = json.loads(spec.implementation_payload)
            if payload["payload_version"] != 1 or not isinstance(payload["framing"], dict):
                raise TypeError
            framing = FramingPolicy(**payload["framing"])
            workload = cls._validate_workload_payload(payload["workload"])
            return framing, workload
        except (KeyError, TypeError, ValueError) as exc:
            raise ParameterError("Invalid tokenizer implementation payload schema.") from exc

    @abstractmethod
    def _prompt_parts(self, context: ContextT) -> tuple[str, str, str]:
        """Return validated instruction, schema fragment, and prefill."""

    @classmethod
    @abstractmethod
    def _default_workload_payload(cls) -> JsonObject:
        """Return the built-in default workload policy."""

    @classmethod
    @abstractmethod
    def _validate_workload_payload(cls, payload: object) -> JsonObject:
        """Validate and normalize a versioned workload policy payload."""

    def _record_payload(self, record: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        """Customize payload representation within the base JSONL frame."""
        return record
