# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observable contracts for the native-tokenizer functional core."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.data_processing.assembler import Example
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.llm.metadata import LLMPromptConfig, ModelMetadata
from nemo_safe_synthesizer.tokenization import PromptEncoding, WorkloadKind, bind_tokenizer
from nemo_safe_synthesizer.tokenization.core import _BoundTokenization

_SMOL_TEMPLATE = "user\n {instruction} {schema} <|im_end|> \n assistant\n{prefill}"


def _native(tokenizers_dir: Path, name: str = "smollm3b") -> PreTrainedTokenizerBase:
    return cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(tokenizers_dir / name))


def _metadata(
    native: PreTrainedTokenizerBase,
    *,
    template: str = _SMOL_TEMPLATE,
    add_bos: bool = True,
    add_eos: bool = False,
) -> SimpleNamespace:
    bos_token = native.bos_token or "<|im_start|>"
    bos_token_id = native.bos_token_id
    if bos_token_id is None:
        bos_token_id = native.convert_tokens_to_ids(bos_token)
    assert isinstance(bos_token_id, int)
    assert native.eos_token is not None
    assert native.eos_token_id is not None
    return SimpleNamespace(
        prompt_config=LLMPromptConfig(
            template=template,
            add_bos_token_to_prompt=add_bos,
            add_eos_token_to_prompt=add_eos,
            bos_token=bos_token,
            bos_token_id=bos_token_id,
            eos_token=native.eos_token,
            eos_token_id=native.eos_token_id,
        )
    )


def _bound(
    tokenizers_dir: Path,
    name: str = "smollm3b",
    *,
    workload: WorkloadKind = WorkloadKind.TABULAR,
    template: str = _SMOL_TEMPLATE,
    add_bos: bool = True,
    add_eos: bool = False,
) -> tuple[PreTrainedTokenizerBase, _BoundTokenization]:
    native = _native(tokenizers_dir, name)
    metadata = _metadata(native, template=template, add_bos=add_bos, add_eos=add_eos)
    return native, bind_tokenizer(native, metadata, workload_kind=workload)


def test_exact_tabular_prompt_text_and_ids(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)

    prompt = tokenization.render_prompt(["a", "b"], "Generate a JSONL dataset with the following columns: ")

    assert prompt.text == (
        'user\n Generate a JSONL dataset with the following columns:  "a":<unk>,"b":<unk> <|im_end|> \n assistant\n'
    )
    assert prompt.input_ids == (
        128011,
        882,
        198,
        20400,
        264,
        4823,
        43,
        10550,
        449,
        279,
        2768,
        8310,
        25,
        220,
        330,
        64,
        794,
        27,
        3200,
        29,
        1359,
        65,
        794,
        27,
        3200,
        29,
        220,
        128012,
        720,
        18328,
        198,
    )
    assert prompt.attention_mask == (1,) * len(prompt.input_ids)


@pytest.mark.parametrize(
    ("add_bos", "add_eos"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_prompt_special_policy_is_explicit(tokenizers_dir: Path, add_bos: bool, add_eos: bool) -> None:
    native, tokenization = _bound(tokenizers_dir, add_bos=add_bos, add_eos=add_eos)

    prompt = tokenization.encode_prompt_text("hello")
    expected = list(native.encode("hello", add_special_tokens=False))
    if add_bos:
        expected.insert(0, native.convert_tokens_to_ids("<|im_start|>"))
    if add_eos:
        expected.append(native.eos_token_id)

    assert prompt.input_ids == tuple(expected)
    assert prompt.attention_mask == (1,) * len(expected)


def test_ordered_columns_are_not_sorted_or_normalized(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)

    prompt = tokenization.render_prompt(["z", "a b", "λ"], "instruction")

    assert '"z":<unk>,"a b":<unk>,"λ":<unk>' in prompt.text


def test_time_series_initial_and_rolling_prefills_are_verbatim(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir, workload=WorkloadKind.TIME_SERIES)
    initial = ' {"t":1,"v":2}\n'
    rolling = initial + '{"t":2,"v":3}\n'

    first = tokenization.render_prompt(["t", "v"], "instruction", current_prefill=initial)
    second = tokenization.render_prompt(["t", "v"], "instruction", current_prefill=rolling)

    assert initial in first.text
    assert rolling in second.text
    assert second.text != first.text
    assert second.input_ids != first.input_ids


def test_tabular_prefill_is_rejected(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)

    with pytest.raises(ParameterError, match="do not support a prefill"):
        tokenization.render_prompt(["x"], "instruction", current_prefill=" record")


def test_duplicate_prompt_columns_are_rejected(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)

    with pytest.raises(ParameterError, match="unique and ordered"):
        tokenization.render_prompt(["x", "x"], "instruction")


def test_established_pandas_jsonl_float_unicode_and_terminal_lf(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)

    encoded = tokenization.encode_records([{"value": 0.12345678901234567, "text": "λ"}])

    assert encoded.records[0].utf8 == b'{"value":0.123456789,"text":"\xce\xbb"}\n'
    assert encoded.records[0].attention_mask == (1,) * len(encoded.records[0].input_ids)


def test_record_column_order_and_exclusions_are_exact(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)

    encoded = tokenization.encode_records(
        [{"second": 2, "internal": "drop", "first": 1}],
        exclude_columns=("internal",),
    )

    assert encoded.records[0].utf8 == b'{"second":2,"first":1}\n'


def test_empty_record_and_empty_batch_are_distinct(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)

    assert tokenization.encode_records([]).records == ()
    assert tokenization.encode_records([{}]).records[0].utf8 == b"\n"


def test_record_encoding_uses_one_native_batch_call(tokenizers_dir: Path, monkeypatch) -> None:
    native, tokenization = _bound(tokenizers_dir)
    calls: list[tuple[list[str], bool]] = []
    original = type(native).__call__

    def recording_call(self, texts, *, add_special_tokens=True, **kwargs):
        calls.append((list(texts), add_special_tokens))
        return original(self, texts, add_special_tokens=add_special_tokens, **kwargs)

    monkeypatch.setattr(type(native), "__call__", recording_call)

    encoded = tokenization.encode_records([{"x": 1}, {"x": 2}, {"x": 3}])

    assert len(encoded.records) == 3
    assert calls == [(['{"x":1}\n', '{"x":2}\n', '{"x":3}\n'], False)]


@pytest.mark.parametrize("bad", ["x", b"x", ["x", 1]])
def test_record_exclusions_require_column_names(tokenizers_dir: Path, bad) -> None:
    _, tokenization = _bound(tokenizers_dir)

    with pytest.raises(ParameterError, match="exclusions"):
        tokenization.encode_records([{"x": 1}], exclude_columns=bad)


def test_missing_pad_is_derived_on_the_authoritative_native_tokenizer(tokenizers_dir: Path) -> None:
    native = _native(tokenizers_dir, "mistral7b")
    assert native.pad_token_id is None

    tokenization = bind_tokenizer(native, _metadata(native), workload_kind=WorkloadKind.TABULAR)

    assert tokenization.native is native
    assert native.pad_token == native.eos_token
    assert native.pad_token_id == native.eos_token_id == 2


def test_pad_id_zero_is_preserved(tokenizers_dir: Path) -> None:
    native = _native(tokenizers_dir, "tinyllama")
    native.pad_token = native.unk_token
    assert native.pad_token_id == 0

    tokenization = bind_tokenizer(native, _metadata(native), workload_kind=WorkloadKind.TABULAR)

    assert tokenization.pad_token_id == 0
    assert native.pad_token_id == 0


def test_missing_eos_fails_before_pad_derivation(tokenizers_dir: Path) -> None:
    native = _native(tokenizers_dir)
    metadata = _metadata(native)
    native.eos_token = None

    with pytest.raises(ParameterError, match="native EOS token"):
        bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)


def test_prompt_special_binding_must_match_native_token_ids(tokenizers_dir: Path) -> None:
    native = _native(tokenizers_dir)
    metadata = _metadata(native)
    metadata.prompt_config.eos_token_id += 1

    with pytest.raises(ParameterError, match="EOS token string"):
        bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)


def test_capacity_arithmetic_at_exact_edges(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir, add_bos=False, add_eos=False)
    prompt = tokenization.encode_prompt_text("hello")
    context_limit = len(prompt.input_ids) + 2 + 5

    capacity = tokenization.capacity_for(prompt, context_limit=context_limit, sequence_count=1)

    assert capacity.prompt_tokens == len(prompt.input_ids)
    assert capacity.delimiter_tokens == 2
    assert capacity.record_token_capacity == 5
    tokenization.validate_record_capacity(
        prompt,
        record_token_count=5,
        context_limit=context_limit,
        rope_scaling_factor=1,
    )


def test_record_capacity_overflow_preserves_user_error(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)
    prompt = tokenization.encode_prompt_text("hello")

    with pytest.raises(GenerationError, match="At least one record requires more tokens"):
        tokenization.validate_record_capacity(
            prompt,
            record_token_count=6,
            context_limit=len(prompt.input_ids) + 2 + 5,
            rope_scaling_factor=1,
        )


def test_schema_capacity_overflow_preserves_user_error(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)
    prompt = tokenization.encode_prompt_text("hello")

    with pytest.raises(GenerationError, match="dataset schema requires more tokens"):
        tokenization.capacity_for(prompt, context_limit=len(prompt.input_ids) - 1, sequence_count=0)


def test_maximum_sequence_count_is_exact(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)
    prompt = tokenization.encode_prompt_text("hello")

    with pytest.raises(ParameterError, match="exceeds maximum sequence count"):
        tokenization.capacity_for(
            prompt,
            context_limit=100,
            sequence_count=3,
            maximum_sequence_count=2,
        )


def test_can_append_accounts_for_all_future_delimiters(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)
    prompt = tokenization.encode_prompt_text("hello")
    existing = [(10, 11)]
    exact_limit = len(prompt.input_ids) + len(existing[0]) + 1 + 4

    assert tokenization.can_append_sequence(
        prompt,
        current_record_tokens=len(existing[0]),
        candidate_record_tokens=1,
        current_sequence_count=1,
        context_limit=exact_limit,
        maximum_sequence_count=2,
    )
    assert not tokenization.can_append_sequence(
        prompt,
        current_record_tokens=len(existing[0]),
        candidate_record_tokens=2,
        current_sequence_count=1,
        context_limit=exact_limit,
        maximum_sequence_count=2,
    )


def test_training_framing_labels_masks_and_delimiters(tokenizers_dir: Path) -> None:
    native, tokenization = _bound(tokenizers_dir, add_bos=False, add_eos=False)
    prompt = tokenization.encode_prompt_text("hello")

    framed = tokenization.frame_training(
        prompt,
        [[7, 8], [9]],
        add_sequence_delimiters=[True, False],
        sequence_attention_masks=[[0, 1], [1]],
    )

    bos_token_id = native.convert_tokens_to_ids("<|im_start|>")
    assert framed.input_ids == (*prompt.input_ids, bos_token_id, 7, 8, native.eos_token_id, 9)
    assert framed.attention_mask == (*prompt.attention_mask, 1, 0, 1, 1, 1)
    assert framed.labels == (*([-100] * len(prompt.input_ids)), bos_token_id, 7, 8, native.eos_token_id, 9)


def test_training_framing_resolves_shared_delimiter_flag_and_default_mask(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir, add_bos=False, add_eos=False)
    prompt = tokenization.encode_prompt_text("hello")

    framed = tokenization.frame_training(prompt, [[7, 8], [9]], add_sequence_delimiters=False)

    assert framed.input_ids == (*prompt.input_ids, 7, 8, 9)
    assert framed.attention_mask == (*prompt.attention_mask, 1, 1, 1)
    assert framed.labels == (*([-100] * len(prompt.input_ids)), 7, 8, 9)


def test_training_framing_rejects_bad_attention_mask(tokenizers_dir: Path) -> None:
    _, tokenization = _bound(tokenizers_dir)
    prompt = tokenization.encode_prompt_text("hello")

    with pytest.raises(ParameterError, match="attention mask"):
        tokenization.frame_training(prompt, [[1, 2]], sequence_attention_masks=[[1]])


def test_capacity_never_reencodes_static_prompt(tokenizers_dir: Path, monkeypatch) -> None:
    native, tokenization = _bound(tokenizers_dir)
    calls = 0
    original = native.encode

    def recording_encode(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(native, "encode", recording_encode)
    prompt = tokenization.render_prompt(["a"], "instruction")
    assert calls == 1

    for count in range(10):
        tokenization.capacity_for(prompt, context_limit=1000, sequence_count=count)

    assert calls == 1


def test_binding_does_not_copy_or_fingerprint_source_or_vocabulary(tokenizers_dir: Path, monkeypatch) -> None:
    native = _native(tokenizers_dir)
    metadata = _metadata(native)

    def forbidden(*args, **kwargs):
        raise AssertionError("expensive identity work is forbidden during binding")

    monkeypatch.setattr(native, "get_vocab", forbidden)
    monkeypatch.setattr(native, "get_added_vocab", forbidden)

    tokenization = bind_tokenizer(native, metadata, workload_kind=WorkloadKind.TABULAR)

    assert tokenization.native is native


def test_example_append_visits_each_new_sequence_once(tokenizers_dir: Path, monkeypatch) -> None:
    _, tokenization = _bound(tokenizers_dir, add_bos=False, add_eos=False)
    metadata = SimpleNamespace(max_seq_length=10_000, max_sequences_per_example=100, rope_scaling_factor=1)
    prompt = tokenization.encode_prompt_text("hello")
    visited = 0
    original = _BoundTokenization.frame_training

    def recording_frame(self, prompt, sequences, **kwargs):
        nonlocal visited
        visited += sum(len(sequence) for sequence in sequences)
        return original(self, prompt, sequences, **kwargs)

    monkeypatch.setattr(_BoundTokenization, "frame_training", recording_frame)
    example = Example(prompt, tokenization, cast(ModelMetadata, metadata))

    for token_id in range(50):
        example.add_sequence({"input_ids": [token_id], "attention_mask": [1]})

    assert visited == 50
    assert example.num_sequences == 50
    assert example.labels.count(-100) == len(prompt.input_ids)


def test_binding_is_process_local_and_has_no_public_wrapper_identity(tokenizers_dir: Path) -> None:
    native, first = _bound(tokenizers_dir)
    second = bind_tokenizer(native, _metadata(native), workload_kind=WorkloadKind.TABULAR)

    assert first is not second
    assert first.native is second.native is native
    assert not hasattr(first, "spec")
    assert not hasattr(first, "for_hf")


def test_prompt_encoding_is_frozen() -> None:
    prompt = PromptEncoding("x", (1,), (1,))

    with pytest.raises(AttributeError):
        prompt.text = "y"  # ty: ignore[invalid-assignment]
