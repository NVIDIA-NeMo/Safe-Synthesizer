# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402
import sys
from pathlib import Path

import pytest

from nemo_safe_synthesizer.config.autoconfig import AutoConfigResolver
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.observability import get_logger
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

logger = get_logger(__name__)

# Path to config files
CONFIG_DIR = Path(__file__).parent / "required_configs"

llm = pytest.importorskip(
    "vllm", reason="vllm with GPU support is required for these tests (install with: uv sync --extra cu128)"
)

try:
    from vllm import LLM  # noqa: F401
except ImportError:
    pytest.skip(
        "vllm with GPU support is required for these tests (install with: uv sync --extra cu128)",
        allow_module_level=True,
    )


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


# Config-based tests using two datasets:
#   - clinc_oos: free text dataset
#   - enth: numeric/categorical and group-by dataset
# Using pytest.mark.parametrize to DRY out repetitive tests
# Some tests are expected to fail, so we mark them with xfail.
# As we fix those issues, we can remove the xfail marker and the test will be run.


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
    """Test CLINC OOS, a free text dataset, with different models and DP settings."""
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
    assert result.summary.timing.training_time_sec is not None and result.summary.timing.training_time_sec > 0, (
        "Training time should be recorded"
    )
    assert result.summary.timing.generation_time_sec is not None and result.summary.timing.generation_time_sec > 0, (
        "Generation time should be recorded"
    )
    assert result.summary.timing.evaluation_time_sec is not None and result.summary.timing.evaluation_time_sec > 0, (
        "Evaluation time should be recorded"
    )
    assert result.summary.synthetic_data_quality_score is not None, "Quality score should be computed"
    assert result.summary.synthetic_data_quality_score >= quality_threshold, (
        f"Quality score {result.summary.synthetic_data_quality_score} below threshold of {quality_threshold}"
    )
    assert result.summary.data_privacy_score is not None, "Privacy score should be computed"
    assert result.summary.data_privacy_score >= privacy_threshold, (
        f"Privacy score {result.summary.data_privacy_score} below threshold of {privacy_threshold}"
    )
    print(result.summary.model_dump_json())


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
    """Test Dow Jones Index dataset, a group-by-order-by dataset, with different models and DP settings."""
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
    assert result.summary.timing.training_time_sec is not None and result.summary.timing.training_time_sec > 0, (
        "Training time should be recorded"
    )
    assert result.summary.timing.generation_time_sec is not None and result.summary.timing.generation_time_sec > 0, (
        "Generation time should be recorded"
    )
    assert result.summary.timing.evaluation_time_sec is not None and result.summary.timing.evaluation_time_sec > 0, (
        "Evaluation time should be recorded"
    )
    assert result.summary.synthetic_data_quality_score is not None, "Quality score should be computed"
    assert result.summary.synthetic_data_quality_score >= quality_threshold, (
        f"Quality score {result.summary.synthetic_data_quality_score} below threshold of {quality_threshold}"
    )
    assert result.summary.data_privacy_score is not None, "Privacy score should be computed"
    assert result.summary.data_privacy_score >= privacy_threshold, (
        f"Privacy score {result.summary.data_privacy_score} below threshold of {privacy_threshold}"
    )
