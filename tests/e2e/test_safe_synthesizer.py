# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Basic e2e tests for NeMo Safe Synthesizer package.

EXTREME WARNING: Due to unsloth's invasive patching of other libraries,
this style of e2e test depends on the order pytest executes the tests.
Running the default test first (which uses unsloth) will cause the DP
test to subsequently fail because unsloth has patched transformers'
modules in a way that's incompatible with DP.

Recommended to run each test individually, as its own pytest invocation:
uv run --frozen --extra cu128 pytest -s packages/nemo_safe_synthesizer/tests/e2e/ -k default
uv run --frozen --extra cu128 pytest -s packages/nemo_safe_synthesizer/tests/e2e/ -k dp

WARNING: Tests are not currently hermetic and require internet access for:
- fetching the financial transactions dataset from github
- loading model weights from huggingface hub
"""

# ruff: noqa: E402
import sys
from pathlib import Path

import pytest

# Skip all tests in this module if sentence_transformers is not available
pytest.importorskip(
    "sentence_transformers",
    reason="sentence_transformers and a GPU are required for these tests (install with: uv sync --extra cu128)",
)

# Skip all tests in this module if vllm is not properly available.
vllm = pytest.importorskip(
    "vllm", reason="vllm with GPU support is required for these tests (install with: uv sync --extra cu128)"
)

try:
    from vllm import LLM  # noqa: F401
except ImportError:
    pytest.skip(
        "vllm with GPU support is required for these tests (install with: uv sync --extra cu128)",
        allow_module_level=True,
    )


from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.observability import get_logger
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

logger = get_logger(__name__)

# Path to config files
CONFIG_DIR = Path(__file__).parent / "required_configs"


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
    assert result.summary.timing.training_time_sec > 0
    assert result.summary.timing.generation_time_sec > 0
    assert result.summary.timing.evaluation_time_sec > 0


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
    assert result.summary.timing.training_time_sec > 0
    assert result.summary.timing.generation_time_sec > 0
    assert result.summary.timing.evaluation_time_sec > 0
