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

from ..errors import GenerationError, ParameterError
from .records import validate_json_value
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
    TrainingCapacity,
    TrainingEncoding,
    WorkloadContext,
    WorkloadKind,
    canonical_json_bytes,
)

_BASE_PROBE_TEXTS = ("", "NSS tokenizer probe", '{"line":"a\\nb","unicode":"\u0085\u2028\u2029"}\n')
_IGNORE_LABEL = -100
ContextT = TypeVar("ContextT", bound=WorkloadContext)


def _max_tokens_action(rope_scaling_factor: float | None) -> str:
    """Return the stable user guidance for context-capacity failures."""
    factor = rope_scaling_factor if rope_scaling_factor is not None else 1
    if factor <= 5:
        return (
            "Training this model will require modifying your dataset and/or the model "
            "configuration. Consider increasing the rope_scaling_factor parameter "
            f"(currently set to {factor}, you could start by increasing "
            f"to {factor + 1} (must be an integer value between 1 and 6)), "
            "reducing the number of columns in your dataset, shortening the "
            "column names, filtering out rows with long text values, and/or "
            "reducing the number of rows per sequence if you are using the "
            "group_training_examples_by parameter."
        )
    return (
        "Training this model will require modifying your dataset. "
        "The rope_scaling_factor is currently set to 6, which cannot be increased further. "
        "Consider reducing the number of columns in your dataset, shortening the "
        "column names, filtering out rows with long text values, and/or "
        "reducing the number of rows per sequence if you are using the "
        "group_training_examples_by parameter."
    )


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
    POLICY_EPOCHS = PolicyEpochs()

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
            policy_epochs=policy_epochs if policy_epochs is not None else self.POLICY_EPOCHS,
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
        return self.encode_prompt_text(text)

    @final
    def encode_prompt_text(self, text: str) -> PromptEncoding:
        """Encode exact workload-rendered prompt text with NSS framing policy."""
        if not isinstance(text, str):
            raise ParameterError("Prompt text must be a string.")
        input_ids = list(self.encode_no_special(text))
        if self._framing.add_bos_token_to_prompt:
            input_ids.insert(0, cast(int, self._framing.bos_token_id))
        if self._framing.add_eos_token_to_prompt:
            input_ids.append(cast(int, self._framing.eos_token_id))
        ids = tuple(input_ids)
        return PromptEncoding(text=text, input_ids=ids, attention_mask=tuple(1 for _ in ids))

    @final
    def render_training_prompt(
        self,
        ordered_columns: Sequence[str],
        instruction: str,
        *,
        current_prefill: str = "",
    ) -> PromptEncoding:
        """Render the workload's exact training prompt from immutable inputs."""
        if (
            not isinstance(ordered_columns, Sequence)
            or isinstance(ordered_columns, (str, bytes))
            or not all(isinstance(column, str) for column in ordered_columns)
        ):
            raise ParameterError("Training prompt columns must be an ordered sequence of strings.")
        if not isinstance(instruction, str) or not isinstance(current_prefill, str):
            raise ParameterError("Training prompt instruction and prefill must be strings.")
        if not self.capabilities.training_prompt:
            raise ParameterError("The selected NSS tokenizer does not declare training prompt capability.")
        context = self._training_context(tuple(ordered_columns), instruction, current_prefill)
        return self.render_prompt(context)

    def _validate_prompt_encoding(self, prompt: PromptEncoding) -> None:
        if not isinstance(prompt, PromptEncoding):
            raise ParameterError("Training framing requires a PromptEncoding.")
        expected = self.encode_prompt_text(prompt.text)
        if (prompt.text, prompt.input_ids, prompt.attention_mask) != (
            expected.text,
            expected.input_ids,
            expected.attention_mask,
        ):
            raise ParameterError("PromptEncoding does not match this NSS tokenizer and framing policy.")

    @staticmethod
    def _delimiter_flags(
        sequence_count: int,
        add_sequence_delimiters: bool | Sequence[bool],
    ) -> tuple[bool, ...]:
        if isinstance(add_sequence_delimiters, bool):
            return (add_sequence_delimiters,) * sequence_count
        if (
            isinstance(add_sequence_delimiters, Sequence)
            and not isinstance(add_sequence_delimiters, (str, bytes))
            and len(add_sequence_delimiters) == sequence_count
            and all(isinstance(value, bool) for value in add_sequence_delimiters)
        ):
            return tuple(add_sequence_delimiters)
        raise ParameterError("Sequence delimiter flags must be one boolean or one boolean per sequence.")

    @final
    def capacity_for(
        self,
        prompt: PromptEncoding,
        *,
        context_limit: int,
        sequence_count: int,
        maximum_sequence_count: int | None = None,
        add_sequence_delimiters: bool | Sequence[bool] = True,
        rope_scaling_factor: float | None = None,
    ) -> TrainingCapacity:
        """Return exact record capacity after prompt and sequence delimiters."""
        self._validate_prompt_encoding(prompt)
        if not isinstance(context_limit, int) or isinstance(context_limit, bool) or context_limit <= 0:
            raise ParameterError("Training context limit must be a positive integer.")
        if not isinstance(sequence_count, int) or isinstance(sequence_count, bool) or sequence_count < 0:
            raise ParameterError("Training sequence count must be a non-negative integer.")
        if maximum_sequence_count is not None:
            if (
                not isinstance(maximum_sequence_count, int)
                or isinstance(maximum_sequence_count, bool)
                or maximum_sequence_count <= 0
            ):
                raise ParameterError("Training maximum sequence count must be a positive integer or None.")
            if sequence_count > maximum_sequence_count:
                raise ParameterError(
                    f"Training sequence count {sequence_count} exceeds maximum sequence count {maximum_sequence_count}."
                )
        if len(prompt.input_ids) > context_limit:
            action = _max_tokens_action(rope_scaling_factor)
            raise GenerationError(
                "The dataset schema requires more tokens than the max length of the model. "
                "This likely means that the table is too wide to be used with this model. "
                f"{action}"
            )
        delimiter_flags = self._delimiter_flags(sequence_count, add_sequence_delimiters)
        delimiter_tokens = 2 * sum(delimiter_flags)
        record_capacity = context_limit - len(prompt.input_ids) - delimiter_tokens
        return TrainingCapacity(
            context_limit=context_limit,
            prompt_tokens=len(prompt.input_ids),
            sequence_count=sequence_count,
            delimiter_tokens_per_sequence=2,
            delimiter_tokens=delimiter_tokens,
            record_token_capacity=record_capacity,
        )

    @final
    def validate_prompt_capacity(
        self,
        prompt: PromptEncoding,
        *,
        context_limit: int,
        rope_scaling_factor: float | None,
    ) -> None:
        """Raise the stable schema-overflow error when a prompt cannot fit."""
        self.capacity_for(
            prompt,
            context_limit=context_limit,
            sequence_count=0,
            rope_scaling_factor=rope_scaling_factor,
        )

    @final
    def validate_record_capacity(
        self,
        prompt: PromptEncoding,
        *,
        record_token_count: int,
        context_limit: int,
        rope_scaling_factor: float | None,
    ) -> None:
        """Raise the stable record-overflow error for one framed sequence."""
        if not isinstance(record_token_count, int) or isinstance(record_token_count, bool) or record_token_count < 0:
            raise ParameterError("Record token count must be a non-negative integer.")
        capacity = self.capacity_for(
            prompt,
            context_limit=context_limit,
            sequence_count=1,
            rope_scaling_factor=rope_scaling_factor,
        )
        if record_token_count <= capacity.record_token_capacity:
            return
        action = _max_tokens_action(rope_scaling_factor)
        raise GenerationError(
            f"At least one record requires more tokens than fit in the available context length. {action}"
        )

    @final
    def can_append_sequence(
        self,
        prompt: PromptEncoding,
        sequences: Sequence[Sequence[int]],
        candidate: Sequence[int],
        *,
        context_limit: int,
        maximum_sequence_count: int | None = None,
    ) -> bool:
        """Return whether the exact future delimiter-framed sequences fit."""
        sequence_count = len(sequences) + 1
        capacity = self.capacity_for(
            prompt,
            context_limit=context_limit,
            sequence_count=sequence_count,
            maximum_sequence_count=maximum_sequence_count,
        )
        record_tokens = sum(len(sequence) for sequence in sequences) + len(candidate)
        return record_tokens <= capacity.record_token_capacity

    @final
    def encode_records(
        self,
        records: Sequence[Mapping[str, JsonValue]],
        *,
        exclude_columns: Sequence[str] = (),
    ) -> RecordBatch:
        """Serialize and batch-encode ordered terminal-LF JSONL records."""
        if (
            not isinstance(exclude_columns, Sequence)
            or isinstance(exclude_columns, (str, bytes))
            or not all(isinstance(column, str) for column in exclude_columns)
        ):
            raise ParameterError("Record exclusions must be a sequence of column-name strings.")
        if not records:
            return RecordBatch(())
        excluded = frozenset(exclude_columns)
        texts: list[str] = []
        payloads: list[bytes] = []
        for record in records:
            if not isinstance(record, Mapping) or not all(isinstance(key, str) for key in record):
                raise ParameterError("Records must be mappings with string column names.")
            filtered = {key: validate_json_value(value) for key, value in record.items() if key not in excluded}
            payload = self._record_payload(filtered)
            try:
                text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            except (TypeError, ValueError) as exc:
                raise ParameterError("Records must contain finite JSON values.") from exc
            texts.append(text)
            payloads.append(text.encode())
        try:
            native_batch = self._native(texts, add_special_tokens=False)
        except Exception as exc:
            raise ParameterError("The native tokenizer batch operation failed for ordered records.") from exc
        rows = self._validated_native_record_ids(native_batch, len(texts))
        return RecordBatch(
            tuple(
                RecordEncoding(
                    utf8=utf8,
                    input_ids=input_ids,
                    attention_mask=tuple(1 for _ in input_ids),
                )
                for utf8, input_ids in zip(payloads, rows, strict=True)
            )
        )

    @staticmethod
    def _validated_native_record_ids(value: object, row_count: int) -> tuple[tuple[int, ...], ...]:
        if not isinstance(value, Mapping):
            raise ParameterError("Malformed native tokenizer batch: expected a Mapping result.")
        input_ids = next((item for key, item in value.items() if key == "input_ids"), None)
        if not isinstance(input_ids, Sequence) or isinstance(input_ids, (str, bytes, bytearray)):
            raise ParameterError("Malformed native tokenizer batch: input_ids must be a nested sequence.")
        if len(input_ids) != row_count:
            raise ParameterError("Malformed native tokenizer batch: row count does not match input records.")
        rows: list[tuple[int, ...]] = []
        for row in input_ids:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
                raise ParameterError("Malformed native tokenizer batch: every input_ids row must be a sequence.")
            if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in row):
                raise ParameterError("Malformed native tokenizer batch: token IDs must be integers.")
            rows.append(tuple(cast(int, token_id) for token_id in row))
        return tuple(rows)

    @final
    def frame_training(
        self,
        prompt: PromptEncoding,
        sequences: Sequence[Sequence[int]],
        *,
        add_sequence_delimiters: bool | Sequence[bool] = True,
        sequence_attention_masks: Sequence[Sequence[int]] | None = None,
        context_limit: int | None = None,
        maximum_sequence_count: int | None = None,
        rope_scaling_factor: float | None = None,
    ) -> TrainingEncoding:
        """Frame prompt and sequence IDs with stable masks and labels."""
        self._validate_prompt_encoding(prompt)
        delimiters = self._delimiter_flags(len(sequences), add_sequence_delimiters)
        if sequence_attention_masks is None:
            masks = tuple(tuple(1 for _ in sequence) for sequence in sequences)
        elif (
            isinstance(sequence_attention_masks, Sequence)
            and not isinstance(sequence_attention_masks, (str, bytes))
            and len(sequence_attention_masks) == len(sequences)
        ):
            masks = tuple(tuple(mask) for mask in sequence_attention_masks)
        else:
            raise ParameterError("Sequence attention masks must contain one mask per sequence.")
        if maximum_sequence_count is not None and len(sequences) > maximum_sequence_count:
            raise ParameterError(
                f"Training sequence count {len(sequences)} exceeds maximum sequence count {maximum_sequence_count}."
            )
        input_ids = list(prompt.input_ids)
        attention_mask = list(prompt.attention_mask)
        labels = [_IGNORE_LABEL] * len(input_ids)
        for sequence, mask, add_delimiters in zip(sequences, masks, delimiters, strict=True):
            if len(mask) != len(sequence) or any(value not in (0, 1) for value in mask):
                raise ParameterError("Each sequence attention mask must match its IDs and contain only zero or one.")
            framed = list(sequence)
            framed_mask = list(mask)
            if add_delimiters:
                if self._framing.bos_token_id is None:
                    raise ParameterError("A BOS token ID is required for sequence delimiters.")
                framed = [self._framing.bos_token_id, *framed, cast(int, self._framing.eos_token_id)]
                framed_mask = [1, *framed_mask, 1]
            input_ids.extend(framed)
            attention_mask.extend(framed_mask)
            labels.extend(framed)
        if context_limit is not None and len(input_ids) > context_limit:
            action = _max_tokens_action(rope_scaling_factor)
            raise GenerationError(f"The number of tokens in an example exceeds the available context length. {action}")
        return TrainingEncoding(
            input_ids=tuple(input_ids),
            attention_mask=tuple(attention_mask),
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

    def _training_context(
        self,
        ordered_columns: tuple[str, ...],
        instruction: str,
        current_prefill: str,
    ) -> ContextT:
        """Create the workload's immutable training context."""
        raise ParameterError(
            f"Tokenizer implementation {self.IMPLEMENTATION_ID!r} does not declare training-prompt construction."
        )

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
