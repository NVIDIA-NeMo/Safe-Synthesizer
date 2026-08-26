# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-tokenizer prompt, record, capacity, and training framing helpers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Self, cast, final

from transformers import PreTrainedTokenizerBase

from ..data_processing import record_utils
from ..errors import GenerationError, ParameterError

_IGNORE_LABEL = -100


class WorkloadKind(StrEnum):
    """Prompt shapes supported by Safe Synthesizer."""

    TABULAR = "tabular"
    TIME_SERIES = "time-series"


class _PromptConfig(Protocol):
    @property
    def template(self) -> str: ...

    @property
    def add_bos_token_to_prompt(self) -> bool: ...

    @property
    def add_eos_token_to_prompt(self) -> bool: ...

    @property
    def bos_token(self) -> str: ...

    @property
    def bos_token_id(self) -> int: ...

    @property
    def eos_token(self) -> str: ...

    @property
    def eos_token_id(self) -> int: ...


class _Metadata(Protocol):
    @property
    def prompt_config(self) -> _PromptConfig: ...


@dataclass(frozen=True, slots=True)
class PromptEncoding:
    """Exact rendered prompt text and the IDs dispatched to the model."""

    text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


_EMPTY_PROMPT = PromptEncoding("", (), ())


@dataclass(frozen=True, slots=True)
class RecordEncoding:
    """One ordered, terminal-LF JSON record and its native tokenization."""

    utf8: bytes
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class RecordBatch:
    """Variable-length record encodings in input order."""

    records: tuple[RecordEncoding, ...]

    @property
    def input_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return record IDs in input order."""
        return tuple(record.input_ids for record in self.records)

    @property
    def attention_mask(self) -> tuple[tuple[int, ...], ...]:
        """Return record masks in input order."""
        return tuple(record.attention_mask for record in self.records)


@dataclass(frozen=True, slots=True)
class TrainingCapacity:
    """Exact record-token capacity after prompt and delimiters."""

    context_limit: int
    prompt_tokens: int
    sequence_count: int
    delimiter_tokens_per_sequence: int
    delimiter_tokens: int
    record_token_capacity: int

    @classmethod
    def from_prompt(
        cls,
        prompt: PromptEncoding,
        *,
        context_limit: int,
        sequence_count: int,
        maximum_sequence_count: int | None = None,
        add_sequence_delimiters: bool | Sequence[bool] = True,
        rope_scaling_factor: float | None = None,
    ) -> Self:
        """Calculate capacity from an already encoded prompt."""
        if not isinstance(prompt, PromptEncoding):
            raise ParameterError("Training framing requires a PromptEncoding.")
        if not isinstance(context_limit, int) or isinstance(context_limit, bool) or context_limit <= 0:
            raise ParameterError("Training context limit must be a positive integer.")
        if not isinstance(sequence_count, int) or isinstance(sequence_count, bool) or sequence_count < 0:
            raise ParameterError("Training sequence count must be a non-negative integer.")
        match maximum_sequence_count:
            case None:
                pass
            case int() as maximum if not isinstance(maximum, bool) and maximum > 0:
                if sequence_count > maximum:
                    raise ParameterError(
                        f"Training sequence count {sequence_count} exceeds maximum sequence count {maximum}."
                    )
            case _:
                raise ParameterError("Training maximum sequence count must be a positive integer or None.")
        if len(prompt.input_ids) > context_limit:
            raise GenerationError(
                "The dataset schema requires more tokens than the max length of the model. "
                "This likely means that the table is too wide to be used with this model. "
                f"{_max_tokens_action(rope_scaling_factor)}"
            )
        delimiter_policy = _DelimiterPolicy.parse(sequence_count, add_sequence_delimiters)
        delimiter_tokens_per_sequence = 2
        delimiter_tokens = delimiter_tokens_per_sequence * sum(delimiter_policy.flags)
        return cls(
            context_limit,
            len(prompt.input_ids),
            sequence_count,
            delimiter_tokens_per_sequence,
            delimiter_tokens,
            context_limit - len(prompt.input_ids) - delimiter_tokens,
        )


@dataclass(frozen=True, slots=True)
class TrainingEncoding:
    """One fully framed training example."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    labels: tuple[int, ...]


def _max_tokens_action(rope_scaling_factor: float | None) -> str:
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


def _ordered_record_texts(
    records: Sequence[Mapping[str, object]],
    exclude_columns: Sequence[str],
) -> list[str]:
    """Serialize records through the shared JSONL byte-dialect authority."""
    if isinstance(exclude_columns, (str, bytes)) or not all(isinstance(column, str) for column in exclude_columns):
        raise ParameterError("Record exclusions must be a sequence of column-name strings.")
    excluded = frozenset(exclude_columns)
    filtered: list[record_utils.RawRecordMapping] = []
    for record in records:
        if not isinstance(record, Mapping) or not all(isinstance(column, str) for column in record):
            raise ParameterError("Records must be mappings with string column names.")
        filtered.append({column: value for column, value in record.items() if column not in excluded})
    try:
        jsonl = record_utils.records_to_jsonl(filtered)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParameterError("Records must contain values supported by pandas JSONL serialization.") from exc
    texts = jsonl.splitlines(keepends=True)
    if len(texts) != len(records):
        raise ParameterError("Pandas JSONL serialization did not preserve the record count.")
    return texts


@dataclass(frozen=True, slots=True)
class _NativeBatch:
    """Trusted token ID rows returned by the native tokenizer boundary."""

    input_ids: tuple[tuple[int, ...], ...]

    @classmethod
    def parse(cls, value: object, row_count: int) -> Self:
        """Validate the supported Hugging Face batch result shape."""
        match value:
            case Mapping() as batch:
                input_ids = batch.get("input_ids")
            case _:
                raise ParameterError("Malformed native tokenizer batch: expected a Mapping result.")
        match input_ids:
            case Sequence() as rows if not isinstance(rows, (str, bytes, bytearray)):
                pass
            case _:
                raise ParameterError("Malformed native tokenizer batch: input_ids must be a nested sequence.")
        if len(rows) != row_count:
            raise ParameterError("Malformed native tokenizer batch: row count does not match input records.")
        return cls(tuple(cls._parse_row(row) for row in rows))

    @staticmethod
    def _parse_row(value: object) -> tuple[int, ...]:
        match value:
            case Sequence() as row if not isinstance(row, (str, bytes, bytearray)):
                pass
            case _:
                raise ParameterError("Malformed native tokenizer batch: every input_ids row must be a sequence.")
        if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in row):
            raise ParameterError("Malformed native tokenizer batch: token IDs must be integers.")
        return tuple(cast(int, token_id) for token_id in row)


@dataclass(frozen=True, slots=True)
class _DelimiterPolicy:
    """Resolved per-sequence delimiter decisions."""

    flags: tuple[bool, ...]

    @classmethod
    def parse(cls, sequence_count: int, value: bool | Sequence[bool]) -> Self:
        """Resolve one shared flag or one flag per sequence."""
        match value:
            case bool() as shared:
                return cls((shared,) * sequence_count)
            case Sequence() as flags if not isinstance(flags, (str, bytes)):
                if len(flags) == sequence_count and all(isinstance(flag, bool) for flag in flags):
                    return cls(tuple(flags))
            case _:
                pass
        raise ParameterError("Sequence delimiter flags must be one boolean or one boolean per sequence.")


@dataclass(frozen=True, slots=True)
class _TrainingSequence:
    """One sequence with an aligned, validated attention mask and framing policy."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    add_delimiters: bool

    @staticmethod
    def resolve_masks(
        sequences: Sequence[Sequence[int]],
        attention_masks: Sequence[Sequence[int]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        """Resolve optional masks while preserving sequence alignment."""
        match attention_masks:
            case None:
                return tuple(tuple(1 for _ in sequence) for sequence in sequences)
            case Sequence() as supplied if len(supplied) == len(sequences):
                return tuple(tuple(mask) for mask in supplied)
            case _:
                raise ParameterError("Sequence attention masks must contain one mask per sequence.")

    @classmethod
    def parse_many(
        cls,
        sequences: Sequence[Sequence[int]],
        attention_masks: Sequence[tuple[int, ...]],
        delimiter_policy: _DelimiterPolicy,
    ) -> tuple[Self, ...]:
        """Validate aligned sequences, masks, and delimiter flags."""
        return tuple(
            cls._parse(sequence, mask, add_delimiters)
            for sequence, mask, add_delimiters in zip(
                sequences,
                attention_masks,
                delimiter_policy.flags,
                strict=True,
            )
        )

    @classmethod
    def _parse(cls, input_ids: Sequence[int], attention_mask: tuple[int, ...], add_delimiters: bool) -> Self:
        if len(attention_mask) != len(input_ids):
            raise ParameterError("Each sequence attention mask must match its IDs and contain only zero or one.")
        if any(value not in (0, 1) for value in attention_mask):
            raise ParameterError("Each sequence attention mask must match its IDs and contain only zero or one.")
        return cls(tuple(input_ids), attention_mask, add_delimiters)

    def frame(self, bos_token_id: int, eos_token_id: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Apply this sequence's explicit BOS/EOS delimiter policy."""
        if self.add_delimiters:
            return (
                (bos_token_id, *self.input_ids, eos_token_id),
                (1, *self.attention_mask, 1),
            )
        return self.input_ids, self.attention_mask


@final
class _BoundTokenization:
    """Validated process-local pairing of one native tokenizer and prompt policy."""

    __slots__ = (
        "_bos_token_id",
        "_cache_digest",
        "_eos_token_id",
        "_native",
        "_prompt_config",
        "workload_kind",
    )

    def __init__(
        self,
        native: PreTrainedTokenizerBase,
        prompt_config: _PromptConfig,
        workload_kind: WorkloadKind,
    ) -> None:
        if not isinstance(native, PreTrainedTokenizerBase):
            raise ParameterError("Tokenization requires a native Hugging Face tokenizer.")
        if not isinstance(workload_kind, WorkloadKind):
            raise ParameterError("Tokenization requires a supported workload kind.")
        if native.eos_token is None or native.eos_token_id is None:
            raise ParameterError("Runtime tokenization requires a native EOS token for record padding.")
        if native.pad_token is None or native.pad_token_id is None:
            native.pad_token = native.eos_token
        if native.pad_token is None or native.pad_token_id is None:
            raise ParameterError("Runtime tokenization could not derive a record pad token.")
        self._validate_prompt_config(native, prompt_config)
        self._native = native
        self._prompt_config = prompt_config
        self._bos_token_id = prompt_config.bos_token_id
        self._eos_token_id = prompt_config.eos_token_id
        self.workload_kind = workload_kind
        self._cache_digest: str | None = None

    @staticmethod
    def _validate_prompt_config(native: PreTrainedTokenizerBase, prompt_config: _PromptConfig) -> None:
        required = (
            prompt_config.template,
            prompt_config.bos_token,
            prompt_config.eos_token,
        )
        if not all(isinstance(value, str) and value for value in required):
            raise ParameterError("Prompt configuration requires a template and BOS/EOS token strings.")
        for placeholder in ("{instruction}", "{schema}", "{prefill}"):
            if prompt_config.template.count(placeholder) != 1:
                raise ParameterError(f"Prompt template must contain active field {placeholder} exactly once.")
        for name, token, token_id in (
            ("BOS", prompt_config.bos_token, prompt_config.bos_token_id),
            ("EOS", prompt_config.eos_token, prompt_config.eos_token_id),
        ):
            if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
                raise ParameterError(f"{name} token ID must be a non-negative integer.")
            converted = native.convert_tokens_to_ids(token)
            if not isinstance(converted, int) or converted != token_id:
                raise ParameterError(f"{name} token string does not resolve to the configured token ID.")
        if native.eos_token_id != prompt_config.eos_token_id:
            raise ParameterError("EOS token ID does not match the native tokenizer.")

    @property
    def native(self) -> PreTrainedTokenizerBase:
        """Return the sole native tokenizer authority."""
        return self._native

    @property
    def pad_token_id(self) -> int:
        """Return the prepared native pad ID; zero remains valid."""
        return cast(int, self._native.pad_token_id)

    @property
    def cache_digest(self) -> str:
        """Return a stable encoding-transform digest, computed only for caching."""
        if self._cache_digest is None:
            backend = getattr(self._native, "backend_tokenizer", None)
            to_str = getattr(backend, "to_str", None)
            if not callable(to_str):
                raise ParameterError("Reusable token caching requires a serializable fast tokenizer backend.")
            policy = (
                str(to_str()),
                str(self._native.padding_side),
                str(self._native.truncation_side),
                str(self._native.bos_token_id),
                str(self._native.eos_token_id),
                str(self._native.pad_token_id),
            )
            self._cache_digest = hashlib.sha256("\0".join(policy).encode()).hexdigest()
        return self._cache_digest

    def encode_no_special(self, text: str) -> tuple[int, ...]:
        """Encode text with native special-token insertion disabled."""
        if not isinstance(text, str):
            raise ParameterError("Text to encode must be a string.")
        return tuple(self._native.encode(text, add_special_tokens=False))

    def render_prompt(
        self,
        ordered_columns: Sequence[str],
        instruction: str,
        *,
        current_prefill: str = "",
    ) -> PromptEncoding:
        """Render and encode the exact prompt once."""
        if isinstance(ordered_columns, (str, bytes)) or not all(isinstance(column, str) for column in ordered_columns):
            raise ParameterError("Training prompt columns must be an ordered sequence of strings.")
        if len(set(ordered_columns)) != len(ordered_columns):
            raise ParameterError("Training prompt columns must be unique and ordered.")
        if not isinstance(instruction, str) or not isinstance(current_prefill, str):
            raise ParameterError("Training prompt instruction and prefill must be strings.")
        if self.workload_kind is WorkloadKind.TABULAR and current_prefill:
            raise ParameterError("Tabular training prompts do not support a prefill.")
        schema = ",".join(f'"{column}":<unk>' for column in ordered_columns)
        text = self._prompt_config.template.format(
            instruction=instruction,
            schema=schema,
            prefill=current_prefill,
        )
        return self.encode_prompt_text(text)

    def encode_prompt_text(self, text: str) -> PromptEncoding:
        """Apply the explicit prompt BOS/EOS policy to exact text."""
        ids = list(self.encode_no_special(text))
        if self._prompt_config.add_bos_token_to_prompt:
            ids.insert(0, self._bos_token_id)
        if self._prompt_config.add_eos_token_to_prompt:
            ids.append(self._eos_token_id)
        frozen = tuple(ids)
        return PromptEncoding(text, frozen, tuple(1 for _ in frozen))

    def encode_records(
        self,
        records: Sequence[Mapping[str, object]],
        *,
        exclude_columns: Sequence[str] = (),
    ) -> RecordBatch:
        """Serialize with pandas and encode all rows in one native batch call."""
        if not records:
            return RecordBatch(())
        texts = _ordered_record_texts(records, exclude_columns)
        try:
            native_batch = self._native(texts, add_special_tokens=False)
        except Exception as exc:
            raise ParameterError("The native tokenizer batch operation failed for ordered records.") from exc
        batch = _NativeBatch.parse(native_batch, len(texts))
        return RecordBatch(
            tuple(
                RecordEncoding(text.encode(), ids, tuple(1 for _ in ids))
                for text, ids in zip(texts, batch.input_ids, strict=True)
            )
        )

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
        """Calculate capacity without re-encoding the bound prompt."""
        return TrainingCapacity.from_prompt(
            prompt,
            context_limit=context_limit,
            sequence_count=sequence_count,
            maximum_sequence_count=maximum_sequence_count,
            add_sequence_delimiters=add_sequence_delimiters,
            rope_scaling_factor=rope_scaling_factor,
        )

    def validate_prompt_capacity(
        self,
        prompt: PromptEncoding,
        *,
        context_limit: int,
        rope_scaling_factor: float | None,
    ) -> None:
        """Raise the stable schema overflow error when necessary."""
        self.capacity_for(
            prompt,
            context_limit=context_limit,
            sequence_count=0,
            rope_scaling_factor=rope_scaling_factor,
        )

    def validate_record_capacity(
        self,
        prompt: PromptEncoding,
        *,
        record_token_count: int,
        context_limit: int,
        rope_scaling_factor: float | None,
    ) -> None:
        """Raise the stable record overflow error when necessary."""
        if not isinstance(record_token_count, int) or isinstance(record_token_count, bool) or record_token_count < 0:
            raise ParameterError("Record token count must be a non-negative integer.")
        capacity = self.capacity_for(prompt, context_limit=context_limit, sequence_count=1)
        if record_token_count > capacity.record_token_capacity:
            raise GenerationError(
                "At least one record requires more tokens than fit in the available context length. "
                f"{_max_tokens_action(rope_scaling_factor)}"
            )

    def can_append_sequence(
        self,
        prompt: PromptEncoding,
        *,
        current_record_tokens: int,
        candidate_record_tokens: int,
        current_sequence_count: int,
        context_limit: int,
        maximum_sequence_count: int | None = None,
    ) -> bool:
        """Return whether a candidate fits using incrementally tracked totals."""
        for name, value in (
            ("current record token count", current_record_tokens),
            ("candidate record token count", candidate_record_tokens),
            ("current sequence count", current_sequence_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ParameterError(f"Training {name} must be a non-negative integer.")
        capacity = self.capacity_for(
            prompt,
            context_limit=context_limit,
            sequence_count=current_sequence_count + 1,
            maximum_sequence_count=maximum_sequence_count,
        )
        return current_record_tokens + candidate_record_tokens <= capacity.record_token_capacity

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
        """Frame training sequences with stable masks and labels."""
        if not isinstance(prompt, PromptEncoding):
            raise ParameterError("Training framing requires a PromptEncoding.")
        delimiter_policy = _DelimiterPolicy.parse(len(sequences), add_sequence_delimiters)
        masks = _TrainingSequence.resolve_masks(sequences, sequence_attention_masks)
        if maximum_sequence_count is not None and len(sequences) > maximum_sequence_count:
            raise ParameterError(
                f"Training sequence count {len(sequences)} exceeds maximum sequence count {maximum_sequence_count}."
            )
        parsed_sequences = _TrainingSequence.parse_many(sequences, masks, delimiter_policy)
        input_ids = list(prompt.input_ids)
        attention_mask = list(prompt.attention_mask)
        labels = [_IGNORE_LABEL] * len(input_ids)
        for sequence in parsed_sequences:
            framed_ids, framed_mask = sequence.frame(self._bos_token_id, self._eos_token_id)
            input_ids.extend(framed_ids)
            attention_mask.extend(framed_mask)
            labels.extend(framed_ids)
        if context_limit is not None and len(input_ids) > context_limit:
            raise GenerationError(
                "The number of tokens in an example exceeds the available context length. "
                f"{_max_tokens_action(rope_scaling_factor)}"
            )
        return TrainingEncoding(tuple(input_ids), tuple(attention_mask), tuple(labels))

    def validate_training_length(
        self,
        token_count: int,
        *,
        context_limit: int,
        rope_scaling_factor: float | None,
    ) -> None:
        """Validate an incrementally maintained framed-example length."""
        if token_count > context_limit:
            raise GenerationError(
                "The number of tokens in an example exceeds the available context length. "
                f"{_max_tokens_action(rope_scaling_factor)}"
            )


def bind_tokenizer(
    native: PreTrainedTokenizerBase,
    metadata: _Metadata,
    *,
    workload_kind: WorkloadKind,
) -> _BoundTokenization:
    """Validate and bind one native tokenizer to the authoritative prompt config."""
    return _BoundTokenization(native, metadata.prompt_config, workload_kind)
