# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU generation smoke tests -- generate, schema prompt, processor.

All tests run on CPU. They exercise HuggingFace model.generate() and
NSS generation/processing utilities.
"""

from unittest.mock import MagicMock

from nemo_safe_synthesizer.config import SafeSynthesizerParameters
from nemo_safe_synthesizer.defaults import DEFAULT_INSTRUCTION, PROMPT_TEMPLATE
from nemo_safe_synthesizer.generation.processors import create_processor
from nemo_safe_synthesizer.utils import create_schema_prompt


def test_generate_with_schema_prompt(tiny_model, stub_tokenizer):
    """Exercises: create_schema_prompt() + model.generate()."""
    prompt = create_schema_prompt(
        columns=["col1", "col2", "col3"],
        instruction=DEFAULT_INSTRUCTION,
        prompt_template=PROMPT_TEMPLATE,
    )
    input_ids = stub_tokenizer(prompt, return_tensors="pt")["input_ids"]
    output = tiny_model.generate(input_ids, max_new_tokens=20, do_sample=False)
    decoded = stub_tokenizer.decode(output[0], skip_special_tokens=True)
    assert len(decoded) > len(prompt)  # model produced something


def test_processor_parses_generated_text(tiny_model, stub_tokenizer):
    """Exercises: create_processor() does not crash on garbage text from random model."""
    # Generate some text (will be garbage from random model)
    input_ids = stub_tokenizer("test", return_tensors="pt")["input_ids"]
    output = tiny_model.generate(input_ids, max_new_tokens=50, do_sample=False)
    generated_text = stub_tokenizer.decode(output[0], skip_special_tokens=True)
    assert len(generated_text) > 0

    # Create a schema and processor
    schema = {"properties": {"col1": {"type": "string"}, "col2": {"type": "integer"}}}

    # Build minimal config and metadata mock for create_processor
    config = SafeSynthesizerParameters.from_params(
        use_unsloth=False,
        rope_scaling_factor=1,
    )
    mock_metadata = MagicMock()
    mock_metadata.prompt_config.prompt_template = PROMPT_TEMPLATE
    mock_metadata.instruction = DEFAULT_INSTRUCTION
    mock_metadata.base_max_seq_length = 128

    processor = create_processor(schema=schema, metadata=mock_metadata, config=config)

    # The processor should NOT crash even on garbage text.
    # It's OK if valid_records is empty -- we just need no exceptions.
    # Processor.__call__ signature: (prompt_number: int, text: str)
    result = processor(prompt_number=0, text=generated_text)
    assert result is not None
    assert hasattr(result, "valid_records")
    assert hasattr(result, "invalid_records")
