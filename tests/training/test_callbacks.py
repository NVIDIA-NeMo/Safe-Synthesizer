# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from nemo_safe_synthesizer.data_processing.record_utils import ParsedRecord
from nemo_safe_synthesizer.generation.processors import ParsedResponse, TabularDataProcessor
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.training import callbacks as callbacks_module
from nemo_safe_synthesizer.training.callbacks import InferenceEvalCallback, SafeSynthesizerWorkerCallback


@pytest.fixture
def fixture_mock_metadata():
    """``ModelMetadata`` stub exposing the fields ``InferenceEvalCallback`` reads."""
    metadata = MagicMock(spec=ModelMetadata)
    metadata.instruction = "Generate a record."
    metadata.prompt_config = MagicMock()
    metadata.prompt_config.template = "{instruction} {schema} {prefill}"
    metadata.prompt_config.add_bos_token_to_prompt = False
    metadata.prompt_config.add_eos_token_to_prompt = False
    metadata.prompt_config.bos_token_id = 1
    metadata.prompt_config.eos_token_id = 2
    return metadata


@pytest.fixture
def fixture_mock_processor():
    """Minimal ``Processor`` stub that yields a single valid ``ParsedResponse`` per call."""
    processor = MagicMock(spec=TabularDataProcessor)
    processor.name = "mock-processor"
    processor.return_value = ParsedResponse(
        records=[ParsedRecord(text='{"col_a": "value"}', parsed={"col_a": "value"})],
        prompt_number=0,
    )
    return processor


@pytest.fixture
def fixture_mock_model():
    """Model stub whose ``generate`` returns a single dummy sequence."""
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.tensor([[0, 1, 2]])
    return model


def _build_tokenizer(model_max_length: int, prompt_token_ids: list[int]) -> MagicMock:
    """Build a tokenizer stub that returns ``prompt_token_ids`` for any prompt."""
    tokenizer = MagicMock()
    tokenizer.model_max_length = model_max_length
    tokenizer.return_value = {
        "input_ids": torch.tensor([prompt_token_ids]),
        "attention_mask": torch.ones((1, len(prompt_token_ids)), dtype=torch.long),
    }
    tokenizer.batch_decode.return_value = ["{}"]
    # ``None`` routes the post-generate completion-token counter through the
    # shape-based fallback, which works with the integer-tensor stub above.
    tokenizer.pad_token_id = None
    return tokenizer


def _invoke_on_evaluate(
    callback: InferenceEvalCallback,
    model: MagicMock,
    tokenizer: MagicMock,
) -> tuple[MagicMock, MagicMock]:
    """Drive ``on_evaluate`` with world-process-zero state."""
    state = MagicMock()
    state.is_world_process_zero = True
    state.log_history = []
    control = MagicMock()
    callback.on_evaluate(
        args=MagicMock(),
        state=state,
        control=control,
        model=model,
        tokenizer=tokenizer,
    )
    return state, control


def test_worker_callback_logs_compact_structured_progress(monkeypatch):
    """Training progress uses a compact message without flattening JSON metrics."""
    log_info = MagicMock()
    monkeypatch.setattr(callbacks_module.logger.runtime, "info", log_info)
    state = MagicMock(is_local_process_zero=True, global_step=2, max_steps=10, epoch=0.2)

    SafeSynthesizerWorkerCallback().on_log(
        args=MagicMock(gradient_accumulation_steps=4),
        state=state,
        control=MagicMock(),
        logs={"loss": 0.123456},
    )

    log_info.assert_called_once_with(
        "Training Progress | Progress: 20.00% | Epoch: 0.20 | Step: 8 | Loss: 0.12",
        extra={
            "ctx": {
                "tabular_data": {"progress": 0.2, "epoch": 0.2, "step": 8, "loss": 0.1235},
                "title": "Training Progress",
            }
        },
    )


class TestInferenceEvalCallbackMaxNewTokens:
    """Ensure ``max_new_tokens`` is derived from ``metadata.generation_max_tokens_for``."""

    def test_callback_uses_generation_max_tokens_for_when_helper_is_tighter(
        self,
        fixture_mock_metadata,
        fixture_mock_processor,
        fixture_mock_model,
    ):
        """Training-informed helper output drives ``max_new_tokens`` directly."""
        fixture_mock_metadata.generation_max_tokens_for.return_value = 500
        prompt_token_ids = list(range(10))
        tokenizer = _build_tokenizer(model_max_length=5000, prompt_token_ids=prompt_token_ids)

        callback = InferenceEvalCallback(
            schema={"properties": {"col_a": {"type": "string"}}},
            metadata=fixture_mock_metadata,
            processor=fixture_mock_processor,
            num_prompts_per_batch=1,
            num_batches=1,
        )

        _invoke_on_evaluate(callback, fixture_mock_model, tokenizer)

        assert fixture_mock_model.generate.call_args.kwargs["max_new_tokens"] == 500
        fixture_mock_metadata.generation_max_tokens_for.assert_called_once_with(len(prompt_token_ids))
        tokenizer.assert_called_once_with(
            [callback.templated_prompt],
            add_special_tokens=False,
            return_tensors="pt",
        )

    def test_callback_applies_only_metadata_owned_prompt_boundaries(
        self,
        fixture_mock_metadata,
        fixture_mock_processor,
        fixture_mock_model,
    ):
        """Callback prompt IDs match assembly instead of tokenizer-global defaults."""
        fixture_mock_metadata.prompt_config.add_bos_token_to_prompt = True
        fixture_mock_metadata.prompt_config.add_eos_token_to_prompt = True
        fixture_mock_metadata.prompt_config.bos_token_id = 101
        fixture_mock_metadata.prompt_config.eos_token_id = 102
        fixture_mock_metadata.generation_max_tokens_for.return_value = 20
        tokenizer = _build_tokenizer(model_max_length=100, prompt_token_ids=[7, 8])

        callback = InferenceEvalCallback(
            schema={"properties": {"col_a": {"type": "string"}}},
            metadata=fixture_mock_metadata,
            processor=fixture_mock_processor,
            num_prompts_per_batch=1,
            num_batches=1,
        )

        _invoke_on_evaluate(callback, fixture_mock_model, tokenizer)

        generate_kwargs = fixture_mock_model.generate.call_args.kwargs
        assert generate_kwargs["input_ids"].tolist() == [[101, 7, 8, 102]]
        assert generate_kwargs["attention_mask"].tolist() == [[1, 1, 1, 1]]
        fixture_mock_metadata.generation_max_tokens_for.assert_called_once_with(4)

    def test_callback_uses_helper_remaining_context_when_stat_unset(
        self,
        fixture_mock_metadata,
        fixture_mock_processor,
        fixture_mock_model,
    ):
        """Legacy metadata routes through the helper's ``max_seq_length - prompt_len`` clamp."""
        prompt_token_ids = list(range(10))
        # Helper would return ``max_seq_length - prompt_len`` -- mirror that here.
        remaining = 500 - len(prompt_token_ids)
        fixture_mock_metadata.generation_max_tokens_for.return_value = remaining
        tokenizer = _build_tokenizer(model_max_length=500, prompt_token_ids=prompt_token_ids)

        callback = InferenceEvalCallback(
            schema={"properties": {"col_a": {"type": "string"}}},
            metadata=fixture_mock_metadata,
            processor=fixture_mock_processor,
            num_prompts_per_batch=1,
            num_batches=1,
        )

        _invoke_on_evaluate(callback, fixture_mock_model, tokenizer)

        assert fixture_mock_model.generate.call_args.kwargs["max_new_tokens"] == remaining
        fixture_mock_metadata.generation_max_tokens_for.assert_called_once_with(len(prompt_token_ids))

    @pytest.mark.parametrize("prompt_length", [1, 16, 128])
    def test_callback_forwards_prompt_length_to_helper(
        self,
        prompt_length,
        fixture_mock_metadata,
        fixture_mock_processor,
        fixture_mock_model,
    ):
        """The helper is invoked with the actual tokenized prompt length, regardless of size."""
        model_max_length = 1024
        fixture_mock_metadata.generation_max_tokens_for.side_effect = lambda prompt_len: model_max_length - prompt_len
        tokenizer = _build_tokenizer(
            model_max_length=model_max_length,
            prompt_token_ids=list(range(prompt_length)),
        )

        callback = InferenceEvalCallback(
            schema={"properties": {"col_a": {"type": "string"}}},
            metadata=fixture_mock_metadata,
            processor=fixture_mock_processor,
            num_prompts_per_batch=1,
            num_batches=1,
        )

        _invoke_on_evaluate(callback, fixture_mock_model, tokenizer)

        assert fixture_mock_model.generate.call_args.kwargs["max_new_tokens"] == model_max_length - prompt_length
        fixture_mock_metadata.generation_max_tokens_for.assert_called_once_with(prompt_length)


class TestInferenceEvalCallbackTerminalStatus:
    """Ensure terminal generation states stop the evaluation loop immediately."""

    def test_callback_stops_after_no_records_status(
        self,
        fixture_mock_metadata,
        fixture_mock_processor,
        fixture_mock_model,
    ):
        """A no-records batch should not trigger extra generation batches."""
        fixture_mock_processor.return_value = ParsedResponse(records=[], prompt_number=0)
        fixture_mock_metadata.generation_max_tokens_for.return_value = 500
        tokenizer = _build_tokenizer(model_max_length=5000, prompt_token_ids=list(range(10)))

        callback = InferenceEvalCallback(
            schema={"properties": {"col_a": {"type": "string"}}},
            metadata=fixture_mock_metadata,
            processor=fixture_mock_processor,
            num_prompts_per_batch=1,
            num_batches=3,
        )

        state, control = _invoke_on_evaluate(callback, fixture_mock_model, tokenizer)

        assert fixture_mock_model.generate.call_count == 1
        assert control.should_training_stop is True
        assert state.log_history == [{"training_incomplete": "no_records"}]
