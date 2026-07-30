# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared T1 contract tests for both built-in NSS tokenizers."""

from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.tokenization import (
    FramingPolicy,
    FrozenJsonObject,
    PaddedTokenBatch,
    TabularContext,
    TabularNssTokenizer,
    TimeSeriesContext,
    TimeSeriesNssTokenizer,
    TokenBatch,
    builtin_registry,
)
from nemo_safe_synthesizer.tokenization.base import native_snapshot


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
    assert tokenizer.capabilities.record_jsonl is True
    assert tokenizer.capabilities.rolling_prefill is (implementation is TimeSeriesNssTokenizer)


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

    initial = tokenizer.render_prompt(TimeSeriesContext(schema, "generate", ' {"b":0,"a":1}'))
    rolling = tokenizer.render_prompt(TimeSeriesContext(schema, "generate", '{"b":2,"a":3}\n'))

    assert initial.text == 'I:generate|S:"b":<unk>,"a":<unk>|P: {"b":0,"a":1}'
    assert rolling.text == 'I:generate|S:"b":<unk>,"a":<unk>|P:{"b":2,"a":3}\n'
    for rendered in (initial, rolling):
        expected = (
            cast(int, policy.bos_token_id),
            *native.encode(rendered.text, add_special_tokens=False),
            cast(int, policy.eos_token_id),
        )
        assert rendered.input_ids == expected
