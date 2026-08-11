# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.pii_replacement import ReplacePiiConfig
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.preflight import PreflightContext


def make_ctx(
    config: SafeSynthesizerParameters | None = None,
    data: pd.DataFrame | None = None,
    metadata: object | None = None,
) -> PreflightContext:
    """Build a ``PreflightContext`` with sensible defaults for tests.

    The default metadata mock is ``spec=ModelMetadata`` so that typoed
    attribute access (e.g. ``ctx.metadata.autoconfigg``) raises
    ``AttributeError`` at test time rather than silently returning a
    fresh auto-mock and masking production bugs.
    """
    return PreflightContext(
        data=pd.DataFrame() if data is None else data,
        config=SafeSynthesizerParameters() if config is None else config,
        metadata=MagicMock(spec=ModelMetadata) if metadata is None else metadata,  # ty: ignore[invalid-argument-type] -- MagicMock stands in for ModelMetadata in tests
    )


@pytest.fixture
def ctx_factory():
    """Return the ``make_ctx`` helper (fixture wrapper for test ergonomics)."""
    return make_ctx


@pytest.fixture
def default_config():
    """Default SafeSynthesizerParameters with all defaults."""
    return SafeSynthesizerParameters()


@pytest.fixture
def llm_pii_config(default_config):
    """Config with LLM PII enhancement enabled (inference env checks apply)."""
    return default_config.model_copy(
        update={"replace_pii": ReplacePiiConfig.model_construct(llm_enhancement=True)},
        deep=True,
    )


@pytest.fixture
def pretrained_config(default_config):
    """Return a helper that updates the default pretrained model."""

    def make_config(model: str) -> SafeSynthesizerParameters:
        default_config.training.pretrained_model = model
        return default_config

    return make_config


@pytest.fixture
def sample_df():
    """Simple 5-column, 500-row DataFrame for testing."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "id": range(500),
            "name": [f"item_{i}" for i in range(500)],
            "value": rng.random(500),
            "category": rng.choice(["A", "B", "C", "D"], 500),
            "score": rng.integers(0, 100, 500),
        }
    )


@pytest.fixture
def small_df():
    """50-row DataFrame for size-check testing."""
    return pd.DataFrame(
        {
            "col_a": range(50),
            "col_b": ["x"] * 50,
        }
    )


@pytest.fixture
def tiny_df():
    """5-row DataFrame for extreme-small testing."""
    return pd.DataFrame(
        {
            "col_a": range(5),
            "col_b": ["x"] * 5,
        }
    )
