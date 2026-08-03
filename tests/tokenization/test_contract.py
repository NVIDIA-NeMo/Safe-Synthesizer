# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared T1 contract tests for both built-in NSS tokenizers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.tokenization import (
    FramingPolicy,
    FrozenJsonObject,
    PaddedTokenBatch,
    PromptEncoding,
    TabularContext,
    TabularNssTokenizer,
    TimeSeriesContext,
    TimeSeriesNssTokenizer,
    TokenBatch,
    builtin_registry,
)
from nemo_safe_synthesizer.tokenization.base import native_snapshot
from nemo_safe_synthesizer.tokenization.records import columnar_batch_to_records


def _native_and_policy(tokenizers_dir, fixture_name: str):
    native = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizers_dir / fixture_name, local_files_only=True),
    )
    if fixture_name == "smollm3b":
        bos_token = "<|im_start|>"
        converted = native.convert_tokens_to_ids(bos_token)
        bos_id = converted if isinstance(converted, int) else None
    else:
        bos_token = cast(str, native.bos_token)
        bos_id = native.bos_token_id
    if native.pad_token_id is None:
        native.pad_token = native.eos_token
    policy = FramingPolicy(
        prompt_template="I:{instruction}|S:{schema}|P:{prefill}",
        add_bos_token_to_prompt=True,
        add_eos_token_to_prompt=True,
        bos_token_id=bos_id,
        eos_token_id=native.eos_token_id,
        pad_token_id=native.pad_token_id,
        bos_token=bos_token,
        eos_token=cast(str, native.eos_token),
        pad_token=cast(str, native.pad_token),
    )
    return native, policy


@pytest.fixture(params=["tinyllama", "mistral7b", "smollm3b"])
def checked_native(request, tokenizers_dir):
    return request.param, *_native_and_policy(tokenizers_dir, request.param)


@pytest.fixture(params=[TabularNssTokenizer, TimeSeriesNssTokenizer])
def implementation(request):
    return request.param


def _context(implementation):
    if implementation is TabularNssTokenizer:
        return TabularContext(("b", "a"), "generate")
    return TimeSeriesContext(
        schema=FrozenJsonObject.from_value({"type": "object", "properties": {"b": {}, "a": {}}}),
        instruction="generate",
        current_prefill='{"b":1,"a":2}\n',
    )


def test_shared_builtins_contract_on_three_checked_fixtures(checked_native, implementation, tokenizers_dir) -> None:
    fixture_name, native, policy = checked_native
    registry = builtin_registry()
    tokenizer = registry.create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )
    before = native_snapshot(native)

    prompt = tokenizer.render_prompt(_context(implementation))
    records = tokenizer.encode_records([{"b": "x\ny\u0085\u2028\u2029", "a": 1}, {}])
    training = tokenizer.frame_training(prompt, records.input_ids)

    assert tokenizer.for_hf() is native
    assert prompt.input_ids
    assert len(prompt.input_ids) == len(prompt.attention_mask)
    assert records.records[0].utf8 == b'{"b":"x\\ny\xc2\x85\xe2\x80\xa8\xe2\x80\xa9","a":1}\n'
    assert records.records[1].utf8 == b"{}\n"
    assert len(training.input_ids) == len(training.attention_mask) == len(training.labels)
    assert training.labels[: len(prompt.input_ids)] == (-100,) * len(prompt.input_ids)
    assert tokenizer.encode_records([]).records == ()
    assert tokenizer.encode_records([{}, {}]).records[0].utf8 == b"{}\n"
    assert native_snapshot(native) == before
    assert len(tokenizer.spec.cache_identity_fragment) == 64
    assert tokenizer.spec.registry_digest == registry.digest
    assert tokenizer.capabilities.no_special_encoding is True
    assert tokenizer.capabilities.prompt_encoding is True
    assert tokenizer.capabilities.training_prompt is True
    assert tokenizer.capabilities.record_jsonl is True
    assert tokenizer.capabilities.rolling_prefill is (implementation is TimeSeriesNssTokenizer)


def test_encode_records_preserves_order_excludes_without_mutation(
    checked_native, implementation, tokenizers_dir
) -> None:
    fixture_name, native, policy = checked_native
    tokenizer = builtin_registry().create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )
    records = [{"z": "line\n\u0085\u2028\u2029", "drop": 2, "a": 1}, {"drop": 3, "a": "é"}]
    before = [record.copy() for record in records]

    encoded = tokenizer.encode_records(
        records,
        exclude_columns=("drop", "unknown", "drop"),
    )

    assert tuple(record.utf8 for record in encoded.records) == (
        b'{"z":"line\\n\xc2\x85\xe2\x80\xa8\xe2\x80\xa9","a":1}\n',
        b'{"a":"\xc3\xa9"}\n',
    )
    assert records == before
    assert encoded.attention_mask == tuple(tuple(1 for _ in ids) for ids in encoded.input_ids)


def test_encode_records_uses_one_native_batch_call(checked_native, implementation, tokenizers_dir, monkeypatch) -> None:
    fixture_name, native, policy = checked_native
    tokenizer = builtin_registry().create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )
    native_type = type(native)
    original_call = native_type.__call__
    calls: list[list[str]] = []

    def recording_call(self, texts, *, add_special_tokens):
        calls.append(list(texts))
        return original_call(self, texts, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(native_type, "__call__", recording_call)

    batch = tokenizer.encode_records([{"a": "x"}, {"a": "longer"}, {}])

    assert calls == [[record.utf8.decode() for record in batch.records]]
    assert len({len(row) for row in batch.input_ids}) > 1


def test_encode_records_empty_and_all_columns_excluded(
    checked_native, implementation, tokenizers_dir, monkeypatch
) -> None:
    fixture_name, native, policy = checked_native
    tokenizer = builtin_registry().create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )
    native_type = type(native)
    original_call = native_type.__call__
    calls = 0

    def recording_call(self, texts, *, add_special_tokens):
        nonlocal calls
        calls += 1
        return original_call(self, texts, add_special_tokens=add_special_tokens)

    monkeypatch.setattr(native_type, "__call__", recording_call)

    assert tokenizer.encode_records([]).records == ()
    batch = tokenizer.encode_records([{"a": 1}, {"a": 2}], exclude_columns=("a",))

    assert tuple(record.utf8 for record in batch.records) == (b"{}\n", b"{}\n")
    assert calls == 1


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        {},
        {"input_ids": [[1]]},
        {"input_ids": [1, 2]},
        {"input_ids": [[1], ["bad"]]},
        {"input_ids": [[True], [2]]},
    ],
)
def test_encode_records_rejects_malformed_native_batches(
    checked_native, implementation, tokenizers_dir, monkeypatch, malformed
) -> None:
    fixture_name, native, policy = checked_native
    tokenizer = builtin_registry().create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )

    def malformed_call(_self, _texts, *, add_special_tokens):
        assert add_special_tokens is False
        return malformed

    monkeypatch.setattr(type(native), "__call__", malformed_call)

    with pytest.raises(ParameterError, match="native tokenizer batch"):
        tokenizer.encode_records([{"a": 1}, {"a": 2}])


@pytest.mark.parametrize(
    "bad_records",
    [
        [{"a": float("nan")}],
        [{"a": float("inf")}],
        [{"a": object()}],
        [{"a": (1, 2)}],
        [{"a": {1: "coerced"}}],
    ],
)
def test_encode_records_rejects_non_json_values(checked_native, implementation, tokenizers_dir, bad_records) -> None:
    fixture_name, native, policy = checked_native
    tokenizer = builtin_registry().create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )

    with pytest.raises(ParameterError, match="JSON"):
        tokenizer.encode_records(bad_records)


def test_encode_records_rejects_invalid_exclusions(checked_native, implementation, tokenizers_dir) -> None:
    fixture_name, native, policy = checked_native
    tokenizer = builtin_registry().create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )

    with pytest.raises(ParameterError, match="exclusions"):
        tokenizer.encode_records([{"a": 1}], exclude_columns=cast(tuple[str, ...], (1,)))
    with pytest.raises(ParameterError, match="exclusions"):
        tokenizer.encode_records([{"a": 1}], exclude_columns=cast(tuple[str, ...], {"a"}))
    with pytest.raises(ParameterError, match="exclusions"):
        tokenizer.encode_records([], exclude_columns=cast(tuple[str, ...], {"a": 1}))


def test_columnar_batch_adapter_preserves_order_and_validates_lengths() -> None:
    columns: Mapping[str, list[object]] = {"b": [1, 2], "a": ["x", "y"]}

    rows = columnar_batch_to_records(columns)

    assert rows == ({"b": 1, "a": "x"}, {"b": 2, "a": "y"})
    assert tuple(rows[0]) == ("b", "a")
    assert columns == {"b": [1, 2], "a": ["x", "y"]}

    with pytest.raises(ParameterError, match="equal lengths"):
        columnar_batch_to_records({"a": [1], "b": [2, 3]})
    with pytest.raises(ParameterError, match="column names"):
        columnar_batch_to_records(cast(Mapping[str, list[object]], {1: [1]}))


def test_columnar_batch_adapter_represents_positive_zero_column_rows() -> None:
    assert columnar_batch_to_records({}, row_count=2) == ({}, {})
    assert columnar_batch_to_records({}) == ()


def test_prompt_special_flag_combinations_are_explicit(tokenizers_dir) -> None:
    native, base_policy = _native_and_policy(tokenizers_dir, "tinyllama")
    registry = builtin_registry()
    bare = native.encode('I:generate|S:"a":<unk>|P:', add_special_tokens=False)
    for add_bos, add_eos in ((False, False), (True, False), (False, True), (True, True)):
        policy = replace(
            base_policy,
            add_bos_token_to_prompt=add_bos,
            add_eos_token_to_prompt=add_eos,
        )
        tokenizer = registry.create(
            (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
            native,
            framing=policy,
            native_source=str(tokenizers_dir / "tinyllama"),
            native_revision="fixture-v1",
        )
        expected = ([policy.bos_token_id] if add_bos else []) + bare + ([policy.eos_token_id] if add_eos else [])
        assert tokenizer.render_prompt(TabularContext(("a",), "generate")).input_ids == tuple(expected)


def test_training_prompt_and_capacity_are_base_owned(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )

    prompt = tokenizer.render_training_prompt(("a",), "generate")
    expected = tokenizer.render_prompt(TabularContext(("a",), "generate"))
    capacity = tokenizer.capacity_for(
        prompt,
        context_limit=len(prompt.input_ids) + 10,
        sequence_count=2,
        maximum_sequence_count=2,
    )

    assert prompt == expected
    assert type(capacity).__name__ == "TrainingCapacity"
    assert capacity.context_limit == len(prompt.input_ids) + 10
    assert capacity.prompt_tokens == len(prompt.input_ids)
    assert capacity.sequence_count == 2
    assert capacity.delimiter_tokens_per_sequence == 2
    assert capacity.record_token_capacity == 6
    assert (
        tokenizer.capacity_for(prompt, context_limit=len(prompt.input_ids), sequence_count=0).record_token_capacity == 0
    )


def test_training_prompt_preserves_literal_unusual_column_bytes(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )
    columns = ('quote"column', "back\\slash", "line\nbreak")
    schema = ",".join(f'"{column}":<unk>' for column in columns)
    expected_text = policy.prompt_template.format(instruction="generate", schema=schema, prefill="")

    prompt = tokenizer.render_training_prompt(columns, "generate")

    assert prompt.text == expected_text
    assert prompt.input_ids == (
        policy.bos_token_id,
        *native.encode(expected_text, add_special_tokens=False),
        policy.eos_token_id,
    )


def test_capacity_and_append_use_exact_future_delimiter_policy(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )
    prompt = tokenizer.render_training_prompt(("a",), "generate")
    exact_limit = len(prompt.input_ids) + 7

    capacity = tokenizer.capacity_for(
        prompt,
        context_limit=len(prompt.input_ids) + 10,
        sequence_count=2,
        add_sequence_delimiters=(True, False),
    )

    assert capacity.record_token_capacity == 8
    assert tokenizer.can_append_sequence(prompt, ((1,),), (2, 3), context_limit=exact_limit)
    assert not tokenizer.can_append_sequence(prompt, ((1,),), (2, 3), context_limit=exact_limit - 1)


def test_training_framing_preserves_sequence_attention_masks(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )
    prompt = tokenizer.render_training_prompt(("a",), "generate")

    framed = tokenizer.frame_training(
        prompt,
        ((10, 11),),
        sequence_attention_masks=((0, 1),),
    )

    assert framed.attention_mask == (*prompt.attention_mask, 1, 0, 1, 1)
    with pytest.raises(ParameterError, match="attention mask"):
        tokenizer.frame_training(prompt, ((10, 11),), sequence_attention_masks=((1,),))


def test_capacity_rejects_inconsistent_or_overflowing_prompt(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )
    prompt = tokenizer.render_training_prompt(("a",), "generate")
    inconsistent = PromptEncoding(prompt.text, (999,), (0,))

    with pytest.raises(ParameterError, match="PromptEncoding"):
        tokenizer.capacity_for(inconsistent, context_limit=10, sequence_count=0)
    with pytest.raises(GenerationError, match="dataset schema requires more tokens"):
        tokenizer.capacity_for(prompt, context_limit=len(prompt.input_ids) - 1, sequence_count=0)


def test_capacity_and_framing_fail_closed_on_invalid_or_overflowing_inputs(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )
    prompt = tokenizer.render_training_prompt(("a",), "generate")

    with pytest.raises(ParameterError, match="maximum sequence count"):
        tokenizer.capacity_for(
            prompt,
            context_limit=100,
            sequence_count=2,
            maximum_sequence_count=1,
        )
    with pytest.raises(GenerationError, match="dataset schema requires more tokens"):
        tokenizer.validate_prompt_capacity(prompt, context_limit=len(prompt.input_ids) - 1, rope_scaling_factor=1)
    with pytest.raises(GenerationError, match="At least one record requires more tokens"):
        tokenizer.validate_record_capacity(
            prompt,
            record_token_count=3,
            context_limit=len(prompt.input_ids) + 4,
            rope_scaling_factor=1,
        )
    with pytest.raises(GenerationError, match="number of tokens in an example exceeds"):
        tokenizer.frame_training(
            prompt,
            ((1, 2, 3),),
            context_limit=len(prompt.input_ids) + 4,
            rope_scaling_factor=1,
        )


def test_wrong_context_fails_without_subclass_leakage(checked_native, implementation, tokenizers_dir) -> None:
    fixture_name, native, policy = checked_native
    tokenizer = builtin_registry().create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )
    wrong = (
        TimeSeriesContext(FrozenJsonObject.from_value({"properties": {}}), "", "")
        if implementation is TabularNssTokenizer
        else TabularContext((), "")
    )

    with pytest.raises(ParameterError, match="requires"):
        tokenizer.render_prompt(wrong)


def test_reconstruction_on_three_checked_fixtures(checked_native, implementation, tokenizers_dir) -> None:
    fixture_name, native, policy = checked_native
    registry = builtin_registry()
    tokenizer = registry.create(
        (1, implementation.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )

    def loader(_source: str, _revision: str, _trust_remote_code: bool):
        loaded, _ = _native_and_policy(tokenizers_dir, fixture_name)
        return loaded

    reconstructed = registry.reconstruct(tokenizer.spec, native_loader=loader)

    assert reconstructed.spec == tokenizer.spec
    assert reconstructed.render_prompt(_context(implementation)) == tokenizer.render_prompt(_context(implementation))


def test_missing_pad_and_missing_eos_fail(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    with pytest.raises(ParameterError, match="pad token ID"):
        replace(policy, pad_token_id=None)

    native.pad_token = None
    with pytest.raises(ParameterError, match="declare a pad token ID"):
        builtin_registry().create(
            (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
            native,
            framing=policy,
            native_source=str(tokenizers_dir / "tinyllama"),
            native_revision="fixture-v1",
        )

    native.pad_token = native.eos_token
    native.eos_token = None
    with pytest.raises(ParameterError, match="declare a EOS token ID"):
        builtin_registry().create(
            (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
            native,
            framing=policy,
            native_source=str(tokenizers_dir / "tinyllama"),
            native_revision="fixture-v1",
        )


def test_zero_pad_id_is_not_replaced(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    native.pad_token = native.unk_token
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=replace(policy, pad_token_id=0, pad_token=cast(str, native.pad_token)),
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )

    batch = tokenizer.batch_encode_no_special(["", "prompt"], padding=True)

    assert batch.input_ids == ((0,), (9508,))
    assert batch.attention_mask == ((0,), (1,))
    assert native.pad_token_id == 0


def test_unpadded_and_padded_batches_have_truthful_types(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )

    ragged = tokenizer.batch_encode_no_special(["", "prompt"])
    padded = tokenizer.batch_encode_no_special(["", "prompt"], padding=True)

    assert isinstance(ragged, TokenBatch)
    assert tuple(map(len, ragged.input_ids)) == (0, 1)
    assert isinstance(padded, PaddedTokenBatch)
    assert tuple(map(len, padded.input_ids)) == (1, 1)


def test_special_token_string_must_match_native_not_only_unk_id(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "tinyllama")
    native.pad_token = native.unk_token

    with pytest.raises(ParameterError, match="does not match"):
        builtin_registry().create(
            (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
            native,
            framing=replace(
                policy,
                pad_token_id=0,
                pad_token="definitely-not-in-vocabulary",
            ),
            native_source=str(tokenizers_dir / "tinyllama"),
            native_revision="fixture-v1",
        )


def test_contexts_reject_mutable_and_wrong_typed_values() -> None:
    with pytest.raises(ParameterError, match="immutable tuple"):
        TabularContext(cast(tuple[str, ...], ["a"]), "generate")
    with pytest.raises(ParameterError, match="instruction"):
        TabularContext(("a",), cast(str, 1))
    with pytest.raises(ParameterError, match="FrozenJsonObject"):
        TimeSeriesContext(cast(FrozenJsonObject, {}), "generate", "")
    with pytest.raises(ParameterError, match="must be strings"):
        TimeSeriesContext(
            FrozenJsonObject.from_value({"properties": {}}),
            "generate",
            cast(str, []),
        )


@pytest.mark.parametrize(
    "template",
    [
        "{{instruction}}{{schema}}{{prefill}}",
        "{instruction}{schema}{prefill}{unknown}",
        "{instruction!r}{schema}{prefill}",
        "{instruction:>10}{schema}{prefill}",
        "{instruction.value}{schema}{prefill}",
    ],
)
def test_prompt_template_requires_exact_active_simple_fields(template: str) -> None:
    with pytest.raises(ParameterError, match="Prompt template"):
        FramingPolicy(
            prompt_template=template,
            add_bos_token_to_prompt=False,
            add_eos_token_to_prompt=False,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<unk>",
        )


def test_smollm3_uses_production_framing_without_native_bos_mutation(tokenizers_dir) -> None:
    native, policy = _native_and_policy(tokenizers_dir, "smollm3b")

    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "smollm3b"),
        native_revision="fixture-v1",
    )

    assert native.bos_token is None
    assert native.bos_token_id is None
    assert policy.bos_token == "<|im_start|>"
    assert policy.bos_token_id == 128011
    assert tokenizer.render_prompt(TabularContext(("a",), "generate")).input_ids[0] == 128011


def test_raw_mistral_requires_explicit_prebinding_pad_configuration(tokenizers_dir) -> None:
    native = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizers_dir / "mistral7b", local_files_only=True),
    )
    policy = FramingPolicy(
        prompt_template="{instruction}{schema}{prefill}",
        add_bos_token_to_prompt=False,
        add_eos_token_to_prompt=False,
        bos_token_id=cast(int, native.bos_token_id),
        eos_token_id=cast(int, native.eos_token_id),
        pad_token_id=cast(int, native.eos_token_id),
        bos_token=cast(str, native.bos_token),
        eos_token=cast(str, native.eos_token),
        pad_token=cast(str, native.eos_token),
    )

    with pytest.raises(ParameterError, match="declare a pad token ID"):
        builtin_registry().create(
            (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
            native,
            framing=policy,
            native_source=str(tokenizers_dir / "mistral7b"),
            native_revision="fixture-v1",
        )


def test_timeseries_initial_and_rolling_prefill_exact_contract(checked_native, tokenizers_dir) -> None:
    fixture_name, native, policy = checked_native
    tokenizer = builtin_registry().create(
        (1, TimeSeriesNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / fixture_name),
        native_revision="fixture-v1",
    )
    schema = FrozenJsonObject.from_value({"type": "object", "properties": {"b": {}, "a": {}}})

    initial_prefill = ' {"b":0,"a":1}\n'
    rolling_prefill = ' {"b":2,"a":3}\n'
    initial = tokenizer.render_prompt(TimeSeriesContext(schema, "generate", initial_prefill))
    rolling = tokenizer.render_prompt(TimeSeriesContext(schema, "generate", rolling_prefill))

    assert initial_prefill[:1] == rolling_prefill[:1] == " "
    assert initial_prefill[-1:] == rolling_prefill[-1:] == "\n"
    assert initial.text == 'I:generate|S:"b":<unk>,"a":<unk>|P: {"b":0,"a":1}\n'
    assert rolling.text == 'I:generate|S:"b":<unk>,"a":<unk>|P: {"b":2,"a":3}\n'
    for rendered in (initial, rolling):
        expected = (
            cast(int, policy.bos_token_id),
            *native.encode(rendered.text, add_special_tokens=False),
            cast(int, policy.eos_token_id),
        )
        assert rendered.input_ids == expected
