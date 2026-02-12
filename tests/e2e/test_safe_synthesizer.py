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
import gc
import sys
import tempfile
from pathlib import Path

import pandas as pd
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


from nemo_safe_synthesizer.config.autoconfig import AutoConfigResolver
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.observability import get_logger
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

logger = get_logger(__name__)

# Path to config files
CONFIG_DIR = Path(__file__).parent / "required_configs"


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Clean up GPU memory between tests to prevent OOM errors.

    This fixture ensures that GPU memory from training models and vLLM
    engines is properly released between test runs. Without this cleanup,
    subsequent tests may fail with "Engine core initialization failed"
    due to insufficient available KV cache memory.
    """
    import torch

    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def update_group_by_config(
    config: SafeSynthesizerParameters,
    group_by_column: str,
    order_by_column: str,
) -> SafeSynthesizerParameters:
    config_dict = config.model_dump()
    config_dict["data"]["group_training_examples_by"] = group_by_column
    config_dict["data"]["order_training_examples_by"] = order_by_column
    config = SafeSynthesizerParameters.model_validate(config_dict)
    return config


@pytest.fixture
def fixture_save_path():
    return Path(tempfile.mkdtemp(prefix="nemo_safe_synthesizer_tmp"))


@pytest.fixture
def fixture_financial_transactions_dataset():
    return pd.read_csv(
        "https://raw.githubusercontent.com/gretelai/gretel-blueprints/refs/heads/main/sample_data/financial_transactions.csv"
    )


@pytest.mark.e2e
@pytest.mark.gpu_integration
@pytest.mark.timeout(1000)
@pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS")
def test_train_and_generate_dp(fixture_financial_transactions_dataset, fixture_save_path):
    df = fixture_financial_transactions_dataset
    config = SafeSynthesizerParameters.from_params(
        enable_synthesis=True,
        enable_replace_pii=False,
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
@pytest.mark.gpu_integration
@pytest.mark.timeout(500)
@pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS")
def test_train_and_generate_defaults(fixture_financial_transactions_dataset, fixture_save_path):
    df = fixture_financial_transactions_dataset
    config = SafeSynthesizerParameters.from_params(
        enable_synthesis=True,
        enable_replace_pii=False,
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


# Config-based tests using two datasets:
#   - clinc_oos: free text dataset
#   - enth: numeric/categorical and group-by dataset
# Using pytest.mark.parametrize to DRY out repetitive tests
# Some tests are expected to fail, so we mark them with xfail.
# As we fix those issues, we can remove the xfail marker and the test will be run.


@pytest.mark.nss_clinc_oos
@pytest.mark.timeout(7200)
@pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS")
@pytest.mark.parametrize(
    "config_file,quality_threshold,privacy_threshold",
    [
        ("mistral-dp.yaml", 4.5, 8.0),
        ("mistral-nodp.yaml", 5.0, 8.5),
        ("smollm3-dp.yaml", 7.0, 8.5),
        ("smollm3-unsloth.yaml", 4.5, 8.5),
        ("tinyllama-dp.yaml", 7.0, 8.0),
        ("tinyllama-unsloth.yaml", 8.0, 9.5),
    ],
    ids=[
        "mistral-dp",
        "mistral-nodp",
        "smollm3-dp",
        "smollm3-unsloth",
        "tinyllama-dp",
        "tinyllama-unsloth",
    ],
)
def test_clinc_oos_dataset(fixture_clinc_oos_dataset, config_file, quality_threshold, privacy_threshold):
    """
    Test CLINC OOS, a free text dataset, with different models and DP settings.
    """
    df = fixture_clinc_oos_dataset
    config = SafeSynthesizerParameters.from_yaml(CONFIG_DIR / config_file)
    config = AutoConfigResolver(df, config)()
    logger.info(f"Running test with config: {config}")

    nss = SafeSynthesizer(config=config).with_data_source(df)
    nss.run()
    result = nss.results

    assert result.synthetic_data is not None, "Synthetic data should be generated"
    assert result.synthetic_data.shape == (config.generation.num_records, df.shape[1]), (
        f"Expected {config.generation.num_records} rows and {df.shape[1]} columns"
    )
    assert result.summary.timing.training_time_sec > 0, "Training time should be recorded"
    assert result.summary.timing.generation_time_sec > 0, "Generation time should be recorded"
    assert result.summary.timing.evaluation_time_sec > 0, "Evaluation time should be recorded"
    assert result.summary.synthetic_data_quality_score is not None, "Quality score should be computed"
    assert result.summary.synthetic_data_quality_score >= quality_threshold, (
        f"Quality score {result.summary.synthetic_data_quality_score} below threshold of {quality_threshold}"
    )
    assert result.summary.data_privacy_score is not None, "Privacy score should be computed"
    assert result.summary.data_privacy_score >= privacy_threshold, (
        f"Privacy score {result.summary.data_privacy_score} below threshold of {privacy_threshold}"
    )
    print(result.summary.model_dump_json())


@pytest.mark.nss_dow_jones_index
@pytest.mark.timeout(7200)
@pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS")
@pytest.mark.parametrize(
    "config_file,quality_threshold,privacy_threshold",
    [
        ("mistral-dp.yaml", 4.5, 6.0),
        ("mistral-nodp.yaml", 7.5, 8.0),
        pytest.param(
            "smollm3-dp.yaml",
            5.0,
            6.0,
            marks=pytest.mark.xfail(reason="SmolLM3 DP sometimes fails with no or low valid records or timeout"),
        ),
        pytest.param(
            "smollm3-unsloth.yaml",
            6.0,
            6.0,
            marks=pytest.mark.xfail(reason="SmolLM3 Unsloth sometimes fails with no or low valid records or timeout"),
        ),
        ("tinyllama-dp.yaml", 5.5, 6.0),
        ("tinyllama-unsloth.yaml", 7.5, 7.0),
    ],
    ids=[
        "mistral-dp",
        "mistral-nodp",
        "smollm3-dp",
        "smollm3-unsloth",
        "tinyllama-dp",
        "tinyllama-unsloth",
    ],
)
def test_dow_jones_index_dataset(fixture_dow_jones_index_dataset, config_file, quality_threshold, privacy_threshold):
    """
    Test Dow Jones Index dataset, a group-by-order-by dataset, with different models and DP settings.
    """
    df = fixture_dow_jones_index_dataset.to_pandas()
    config = SafeSynthesizerParameters.from_yaml(CONFIG_DIR / config_file)
    config = update_group_by_config(config, "stock", "date")
    config = AutoConfigResolver(df, config)()
    if config_file == "smollm3-dp.yaml":
        config.generation.structured_generation_backend = "outlines"
    logger.info(f"Running test with config: {config}")

    nss = SafeSynthesizer(config=config).with_data_source(df)
    nss.run()

    result = nss.results

    assert result.synthetic_data is not None, "Synthetic data should be generated"
    # When group by is used, we don't currently truncate the synthetic data to the number of records specified in the config.
    assert result.synthetic_data.shape[0] >= config.generation.num_records
    assert result.synthetic_data.shape[1] == df.shape[1]
    assert result.summary.timing.training_time_sec > 0, "Training time should be recorded"
    assert result.summary.timing.generation_time_sec > 0, "Generation time should be recorded"
    assert result.summary.timing.evaluation_time_sec > 0, "Evaluation time should be recorded"
    assert result.summary.synthetic_data_quality_score is not None, "Quality score should be computed"
    assert result.summary.synthetic_data_quality_score >= quality_threshold, (
        f"Quality score {result.summary.synthetic_data_quality_score} below threshold of {quality_threshold}"
    )
    assert result.summary.data_privacy_score is not None, "Privacy score should be computed"
    assert result.summary.data_privacy_score >= privacy_threshold, (
        f"Privacy score {result.summary.data_privacy_score} below threshold of {privacy_threshold}"
    )
