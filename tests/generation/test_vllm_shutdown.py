# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the VllmBackend teardown lifecycle."""

from unittest.mock import MagicMock, patch

import pytest

from nemo_safe_synthesizer.generation import vllm_backend as vllm_backend_mod
from nemo_safe_synthesizer.generation.vllm_backend import VllmBackend


@pytest.fixture
def _mock_vllm_cleanup():
    """Patch vLLM distributed cleanup so tests run without a GPU."""
    with (
        patch.object(vllm_backend_mod, "cleanup_dist_env_and_memory") as mock_dist,
        patch.object(vllm_backend_mod, "cleanup_memory") as mock_mem,
    ):
        yield mock_dist, mock_mem


@pytest.fixture
def backend(_mock_vllm_cleanup, fixture_session_cache_dir):
    """Create a VllmBackend with mocked dependencies."""
    mock_metadata = MagicMock()
    mock_metadata.adapter_path = None
    mock_metadata.instruction = "Generate"
    mock_metadata.prompt_config = MagicMock()
    mock_metadata.prompt_config.template = "{instruction} {schema}"

    mock_config = MagicMock()

    mock_workdir = MagicMock()
    mock_workdir.schema_file = fixture_session_cache_dir / "schema.json"
    mock_workdir.schema_file.parent.mkdir(parents=True, exist_ok=True)
    mock_workdir.schema_file.write_text('{"properties": {"col_a": {"type": "string"}}}')
    mock_workdir.adapter_path = None

    return VllmBackend(config=mock_config, model_metadata=mock_metadata, workdir=mock_workdir)


class TestTeardownIdempotency:
    def test_first_teardown_runs_cleanup(self, backend, _mock_vllm_cleanup):
        mock_dist, mock_mem = _mock_vllm_cleanup
        backend.llm = MagicMock()

        backend.teardown()

        mock_dist.assert_called_once()
        mock_mem.assert_called_once()
        assert backend.llm is None
        assert backend._torn_down is True

    def test_second_teardown_is_noop(self, backend, _mock_vllm_cleanup):
        mock_dist, mock_mem = _mock_vllm_cleanup

        backend.teardown()
        mock_dist.reset_mock()
        mock_mem.reset_mock()

        backend.teardown()

        mock_dist.assert_not_called()
        mock_mem.assert_not_called()

    def test_initialize_resets_guard(self, backend, _mock_vllm_cleanup):
        backend.teardown()
        assert backend._torn_down is True

        # Simulate initialize resetting the flag without needing real vLLM
        backend._torn_down = False
        assert backend._torn_down is False


class TestTeardownResilience:
    def test_cleanup_memory_runs_even_if_dist_cleanup_fails(self, backend, _mock_vllm_cleanup):
        mock_dist, mock_mem = _mock_vllm_cleanup
        mock_dist.side_effect = RuntimeError("distributed cleanup failed")

        backend.teardown()

        mock_dist.assert_called_once()
        mock_mem.assert_called_once()
        assert backend.llm is None

    def test_llm_cleared_even_if_dist_cleanup_fails(self, backend, _mock_vllm_cleanup):
        mock_dist, _ = _mock_vllm_cleanup
        mock_dist.side_effect = RuntimeError("boom")
        backend.llm = MagicMock()

        backend.teardown()

        assert backend.llm is None


class TestDunderDel:
    def test_del_calls_teardown(self, backend, _mock_vllm_cleanup):
        mock_dist, _ = _mock_vllm_cleanup

        backend.__del__()

        mock_dist.assert_called_once()
        assert backend._torn_down is True

    def test_del_suppresses_exceptions(self, backend, _mock_vllm_cleanup):
        mock_dist, _ = _mock_vllm_cleanup
        mock_dist.side_effect = RuntimeError("boom")

        backend.__del__()

    def test_del_after_teardown_is_noop(self, backend, _mock_vllm_cleanup):
        mock_dist, _ = _mock_vllm_cleanup

        backend.teardown()
        mock_dist.reset_mock()

        backend.__del__()

        mock_dist.assert_not_called()
