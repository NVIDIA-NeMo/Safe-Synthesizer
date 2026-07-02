# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test helpers for validating XGrammar Structural Tag constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

__all__ = ["structural_tag_accepts_text"]


def structural_tag_accepts_text(
    text: str,
    structural_tag_json: str,
    tokenizer: PreTrainedTokenizerBase,
) -> bool:
    """Return whether *text* is fully accepted by an XGrammar Structural Tag.

    Mirrors regex round-trip tests that use ``re.fullmatch`` against
    ``build_json_based_regex`` output. Requires xgrammar and a Hugging Face
    tokenizer compatible with the generation backend.
    """
    xgr = pytest.importorskip("xgrammar", reason="xgrammar is required for structural tag acceptance tests")

    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
    compiled = compiler.compile_structural_tag(structural_tag_json)
    matcher = xgr.GrammarMatcher(compiled)
    matcher.reset()
    return bool(matcher.accept_string(text) and matcher.is_completed())
