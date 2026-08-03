# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic NSS rendering contracts across independent construction."""

from __future__ import annotations

from typing import cast

import pytest
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_safe_synthesizer.tokenization import (
    FramingPolicy,
    TabularContext,
    TabularNssTokenizer,
    as_tabular_renderer,
    builtin_registry,
    load_nss_tokenizer,
    save_nss_tokenizer,
)


def _create(tokenizers_dir, fixture_name: str) -> TabularNssTokenizer:
    native = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(tokenizers_dir / fixture_name, local_files_only=True),
    )
    if native.pad_token_id is None:
        native.pad_token = native.eos_token
    bos_token = "<|im_start|>" if fixture_name == "smollm3b" else cast(str, native.bos_token)
    bos_token_id = native.convert_tokens_to_ids(bos_token)
    assert isinstance(bos_token_id, int)
    return cast(
        TabularNssTokenizer,
        builtin_registry().create(
            (1, TabularNssTokenizer.IMPLEMENTATION_ID, "1"),
            native,
            framing=FramingPolicy(
                prompt_template="I:{instruction}|S:{schema}|P:{prefill}",
                add_bos_token_to_prompt=True,
                add_eos_token_to_prompt=True,
                bos_token_id=bos_token_id,
                eos_token_id=native.eos_token_id,
                pad_token_id=native.pad_token_id,
                bos_token=bos_token,
                eos_token=cast(str, native.eos_token),
                pad_token=cast(str, native.pad_token),
            ),
            native_source=str(tokenizers_dir / fixture_name),
            native_revision="fixture-v1",
        ),
    )


@pytest.mark.parametrize("fixture_name", ["tinyllama", "mistral7b", "smollm3b"])
def test_independent_and_persisted_tokenizers_render_identically(tokenizers_dir, tmp_path, fixture_name) -> None:
    """Render bytes and IDs remain stable through construction and persistence."""
    context = TabularContext(("name", "value"), "Generate rows.")
    records = [{"name": "\u00e9/\u2603", "value": -1.5}, {"name": "line\nbreak", "value": 2}]
    first = _create(tokenizers_dir, fixture_name)
    second = _create(tokenizers_dir, fixture_name)

    first_prompt = first.render_prompt(context)
    second_prompt = second.render_prompt(context)
    first_records = first.encode_records(records)
    second_records = second.encode_records(records)

    assert first_prompt == second_prompt
    assert first_records == second_records

    artifact = tmp_path / fixture_name
    save_nss_tokenizer(first, artifact)
    restored = load_nss_tokenizer(artifact)
    assert restored is not None
    assert as_tabular_renderer(restored).render_prompt(context) == first_prompt
    assert restored.encode_records(records) == first_records
