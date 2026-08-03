# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T0 executable characterization of legacy tokenizer-facing behavior."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pandas as pd
import pytest
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.data_processing.assembler import Example
from nemo_safe_synthesizer.data_processing.record_utils import records_to_jsonl
from nemo_safe_synthesizer.generation.vllm_backend import _tokens_prompt
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.llm.utils import add_bos_eos_tokens_to_tokenizer
from nemo_safe_synthesizer.tokenization import FramingPolicy, TabularNssTokenizer, builtin_registry


def _metadata(*, add_bos: bool, add_eos: bool) -> ModelMetadata:
    value = SimpleNamespace(
        prompt_config=SimpleNamespace(
            add_bos_token_to_prompt=add_bos,
            add_eos_token_to_prompt=add_eos,
            bos_token_id=1,
            eos_token_id=2,
        ),
        max_seq_length=128,
        rope_scaling_factor=1,
    )
    return cast(ModelMetadata, value)


def _load_native(path) -> PreTrainedTokenizerBase:
    return cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(path, local_files_only=True))


def _example(native: PreTrainedTokenizerBase, metadata: ModelMetadata, native_path, prompt: str) -> Example:
    if native.pad_token_id is None:
        native.pad_token = native.eos_token
    policy = FramingPolicy(
        prompt_template="{instruction}{schema}{prefill}",
        add_bos_token_to_prompt=metadata.prompt_config.add_bos_token_to_prompt,
        add_eos_token_to_prompt=metadata.prompt_config.add_eos_token_to_prompt,
        bos_token_id=native.bos_token_id,
        eos_token_id=native.eos_token_id,
        pad_token_id=native.pad_token_id,
        bos_token=cast(str, native.bos_token),
        eos_token=cast(str, native.eos_token),
        pad_token=cast(str, native.pad_token),
    )
    tokenizer = builtin_registry().create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(native_path),
        native_revision="fixture-v1",
    )
    return Example(tokenizer.encode_prompt_text(prompt), tokenizer, metadata)


def test_legacy_jsonl_bytes_and_unicode_contract() -> None:
    frame = pd.DataFrame([{"second": "x\ny\u0085z\u2028q\u2029", "first": 1}])

    serialized = records_to_jsonl(frame)

    assert serialized.encode() == b'{"second":"x\\ny\xc2\x85z\xe2\x80\xa8q\xe2\x80\xa9","first":1}\n'
    assert serialized.endswith("\n")
    assert not serialized.endswith("\n\n")


def test_legacy_zero_shape_jsonl_is_one_lf_for_both_cases() -> None:
    """Seal surprising pandas behavior; this is not the desired T1 contract."""
    assert records_to_jsonl(pd.DataFrame(columns=["a"])).encode() == b"\n"
    assert records_to_jsonl(pd.DataFrame(index=range(2))).encode() == b"\n"


@pytest.mark.parametrize(
    ("add_bos", "add_eos", "expected"),
    [
        (False, False, [9508]),
        (True, False, [1, 9508]),
        (False, True, [9508, 2]),
        (True, True, [1, 9508, 2]),
    ],
)
def test_legacy_prompt_flag_combinations_exact_ids(
    tokenizers_dir,
    add_bos: bool,
    add_eos: bool,
    expected: list[int],
) -> None:
    tokenizer = cast(PreTrainedTokenizer, _load_native(tokenizers_dir / "tinyllama"))

    example = _example(
        tokenizer,
        _metadata(add_bos=add_bos, add_eos=add_eos),
        tokenizers_dir / "tinyllama",
        "prompt",
    )

    assert example.input_ids == expected
    assert example.attention_mask == [1] * len(expected)
    assert example.labels == [-100] * len(expected)


def test_legacy_exact_record_delimiter_input_mask_and_label_ids(tokenizers_dir) -> None:
    tokenizer = cast(PreTrainedTokenizer, _load_native(tokenizers_dir / "tinyllama"))
    record_ids = [8853, 29874, 1115, 29896, 29913, 13]
    example = _example(
        tokenizer,
        _metadata(add_bos=True, add_eos=True),
        tokenizers_dir / "tinyllama",
        "prompt",
    )

    example.add_sequence({"input_ids": record_ids, "attention_mask": [1] * len(record_ids)})

    assert example.input_ids == [1, 9508, 2, 1, *record_ids, 2]
    assert example.attention_mask == [1] * 11
    assert example.labels == [-100, -100, -100, 1, *record_ids, 2]


def test_legacy_falsy_pad_zero_bug_is_incorrect_current_behavior() -> None:
    tokenizer = SimpleNamespace(add_bos_token=False, add_eos_token=False, pad_token_id=0, eos_token_id=2)

    add_bos_eos_tokens_to_tokenizer(cast(PreTrainedTokenizer, tokenizer))

    assert tokenizer.pad_token_id == 2, "incorrect legacy behavior: valid pad ID zero is overwritten"


def test_t1_pad_zero_remains_valid_without_native_mutation(tokenizers_dir) -> None:
    native = _load_native(tokenizers_dir / "tinyllama")
    native.pad_token = native.unk_token
    registry = builtin_registry()
    policy = FramingPolicy(
        prompt_template="{instruction}{schema}{prefill}",
        add_bos_token_to_prompt=False,
        add_eos_token_to_prompt=False,
        bos_token_id=native.bos_token_id,
        eos_token_id=native.eos_token_id,
        pad_token_id=0,
        bos_token=cast(str, native.bos_token),
        eos_token=cast(str, native.eos_token),
        pad_token=cast(str, native.pad_token),
    )

    tokenizer = registry.create(
        (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
        native,
        framing=policy,
        native_source=str(tokenizers_dir / "tinyllama"),
        native_revision="fixture-v1",
    )

    assert tokenizer.batch_encode_no_special(["", "prompt"], padding=True).input_ids == ((0,), (9508,))
    assert native.pad_token_id == 0


@pytest.mark.parametrize(
    ("fixture_name", "expected_class", "expected_shapes"),
    [
        ("tinyllama", "transformers.models.llama.tokenization_llama.LlamaTokenizer", (1, 5)),
        ("mistral7b", "transformers.models.llama.tokenization_llama.LlamaTokenizer", (1, 5)),
        ("smollm3b", "transformers.tokenization_utils_tokenizers.TokenizersBackend", (1, 5)),
    ],
)
def test_checked_native_classes_and_batch_shapes(
    tokenizers_dir,
    fixture_name: str,
    expected_class: str,
    expected_shapes: tuple[int, int],
) -> None:
    tokenizer = _load_native(tokenizers_dir / fixture_name)
    class_name = f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"
    batch = tokenizer(["{}", '{"a":1}'], add_special_tokens=False, padding=False)["input_ids"]

    assert class_name == expected_class
    assert tuple(len(row) for row in batch) == expected_shapes


def test_initial_and_rolling_timeseries_prefill_are_distinct_bytes_and_ids(tokenizers_dir) -> None:
    tokenizer = _load_native(tokenizers_dir / "tinyllama")
    template = "{instruction}|{schema}|{prefill}"
    initial = template.format(instruction="generate", schema='"t":<unk>', prefill=' {"t":0}')
    rolling = template.format(instruction="generate", schema='"t":<unk>', prefill='{"t":1}\n')

    assert initial.encode() == b'generate|"t":<unk>| {"t":0}'
    assert rolling.encode() == b'generate|"t":<unk>|{"t":1}\n'
    assert tokenizer.encode(initial) == [
        1,
        5706,
        29989,
        29908,
        29873,
        1115,
        0,
        29989,
        8853,
        29873,
        1115,
        29900,
        29913,
    ]
    assert tokenizer.encode(rolling) == [
        1,
        5706,
        29989,
        29908,
        29873,
        1115,
        0,
        29989,
        6377,
        29873,
        1115,
        29896,
        29913,
        13,
    ]


def test_current_vllm_static_token_prompt_shape() -> None:
    prompt = _tokens_prompt([1, 2, 3])

    assert prompt["prompt_token_ids"] == [1, 2, 3]


@pytest.mark.parametrize("fixture_name", ["tinyllama", "mistral7b", "smollm3b"])
def test_native_save_load_preserves_class_and_no_special_ids(tokenizers_dir, tmp_path, fixture_name: str) -> None:
    native = _load_native(tokenizers_dir / fixture_name)
    expected = native.encode("prompt", add_special_tokens=False)
    native.save_pretrained(tmp_path)

    loaded = _load_native(tmp_path)

    assert isinstance(loaded, PreTrainedTokenizerBase)
    assert type(loaded) is type(native)
    assert loaded.encode("prompt", add_special_tokens=False) == expected
