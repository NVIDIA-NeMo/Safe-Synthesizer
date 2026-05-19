# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Basic e2e tests for NeMo Safe Synthesizer package.

WARNING: Tests are not currently hermetic and require internet access for:
- fetching the financial transactions dataset from github
- loading model weights from huggingface hub
"""

# ruff: noqa: E402
import sys

import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers and a GPU are required for these tests (install a CUDA runtime extra such as: uv sync --extra cu129 or uv sync --extra cu130)",
)

# Skip all tests in this module if vllm is not properly available.
vllm = pytest.importorskip(
    "vllm",
    reason="vllm with GPU support is required for these tests (install a CUDA runtime extra such as: uv sync --extra cu129 or uv sync --extra cu130)",
)

try:
    from vllm import LLM  # noqa: F401
except ImportError:
    skip_reason = "vllm with GPU support is required for these tests (install a CUDA runtime extra such as: uv sync --extra cu129 or uv sync --extra cu130)"
    pytest.skip(skip_reason, allow_module_level=True)  # ty: ignore[invalid-argument-type,too-many-positional-arguments]


from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.observability import get_logger
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

logger = get_logger(__name__)


@pytest.mark.e2e
@pytest.mark.requires_gpu
@pytest.mark.timeout(1000)
@pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS")
def test_train_and_generate_dp(fixture_financial_transactions_dataset, fixture_save_path):
    df = fixture_financial_transactions_dataset
    config = SafeSynthesizerParameters.from_params(
        replace_pii=None,
        num_input_records_to_sample=1500,
        dp_enabled=True,
        epsilon=100.0,
        num_records=100,
        use_structured_generation=True,
        structured_generation_backend="outlines",
    )
    logger.info(f"Running DP test with config: {config}")

    nss = SafeSynthesizer(config=config, save_path=fixture_save_path).with_data_source(df)
    nss.run()
    result = nss.results

    assert result.synthetic_data is not None
    assert result.synthetic_data.shape == (config.generation.num_records, df.shape[1])
    assert result.summary.timing.training_time_sec is not None and result.summary.timing.training_time_sec > 0
    assert result.summary.timing.generation_time_sec is not None and result.summary.timing.generation_time_sec > 0
    assert result.summary.timing.evaluation_time_sec is not None and result.summary.timing.evaluation_time_sec > 0


@pytest.mark.e2e
@pytest.mark.requires_gpu
@pytest.mark.timeout(900)
@pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS")
def test_train_and_generate_defaults(fixture_financial_transactions_dataset, fixture_save_path):
    df = fixture_financial_transactions_dataset
    config = SafeSynthesizerParameters.from_params(
        replace_pii=None,
        num_input_records_to_sample=5000,
    )
    logger.info(f"Running test_train_and_generate_defaults with config: {config}")

    nss = SafeSynthesizer(config=config, save_path=fixture_save_path).with_data_source(df)
    nss.run()
    result = nss.results

    assert result.synthetic_data is not None
    assert result.synthetic_data.shape == (config.generation.num_records, df.shape[1])
    assert result.summary.timing.training_time_sec is not None and result.summary.timing.training_time_sec > 0
    assert result.summary.timing.generation_time_sec is not None and result.summary.timing.generation_time_sec > 0
    assert result.summary.timing.evaluation_time_sec is not None and result.summary.timing.evaluation_time_sec > 0
