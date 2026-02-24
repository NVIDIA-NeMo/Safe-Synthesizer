# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import tempfile
from pathlib import Path

import pandas as pd
import pytest


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


@pytest.fixture
def fixture_save_path():
    return Path(tempfile.mkdtemp(prefix="nemo_safe_synthesizer_tmp"))


@pytest.fixture
def fixture_financial_transactions_dataset():
    return pd.read_csv(
        "https://raw.githubusercontent.com/gretelai/gretel-blueprints/refs/heads/main/sample_data/financial_transactions.csv"
    )
