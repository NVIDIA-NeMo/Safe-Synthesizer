# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal type stubs for unsloth (optional dependency, not always installed)."""

from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizer

class FastLanguageModel:
    @staticmethod
    def from_pretrained(**kwargs: Any) -> tuple[PreTrainedModel, PreTrainedTokenizer]: ...
    @staticmethod
    def get_peft_model(model: PreTrainedModel, **kwargs: Any) -> PreTrainedModel: ...
    @staticmethod
    def for_inference(model: PreTrainedModel) -> None: ...
    @staticmethod
    def for_training(model: PreTrainedModel) -> None: ...
