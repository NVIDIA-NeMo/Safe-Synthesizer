# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle tests for ``HuggingFaceBackend.teardown``.

Covers the two contract guarantees that don't need a GPU to verify:

- Idempotency: a second call after ``_torn_down`` is set is a no-op, so
  callers that wrap ``train()`` in ``try/finally`` can call ``teardown``
  on every exit path without double-freeing resources.
- Per-step error isolation: each cleanup step is wrapped in its own
  ``try/except`` so a torch / NCCL failure in one step doesn't prevent
  the others from running and doesn't escape the call.

The branch on ``dist.is_initialized()`` is also exercised here because
calling ``destroy_process_group()`` when no group is initialised would
raise -- a regression in the conditional would surface in test runs that
never set up distributed training (i.e. all of them).

The actual GPU / distributed cleanup behaviour is covered end-to-end by
the integration suite; here everything below the lifecycle logic is
mocked out.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from nemo_safe_synthesizer.training.huggingface_backend import HuggingFaceBackend


def _bare_hf_backend() -> HuggingFaceBackend:
    """Build a ``HuggingFaceBackend`` without running ``__init__``.

    ``__init__`` requires a populated ``SafeSynthesizerParameters`` and a
    ``Workdir``, neither of which the lifecycle tests need. Bypassing it lets
    each test install only the attributes it cares about.
    """
    return HuggingFaceBackend.__new__(HuggingFaceBackend)


class TestHuggingFaceBackendTeardown:
    def test_idempotent_after_first_call(self):
        backend = _bare_hf_backend()
        backend.trainer = MagicMock()
        backend.model = MagicMock()

        with (
            patch("nemo_safe_synthesizer.training.huggingface_backend.cleanup_memory") as cleanup,
            patch("torch.distributed.is_initialized", return_value=False),
        ):
            backend.teardown()
            backend.teardown()

        assert cleanup.call_count == 1
        assert backend._torn_down is True

    def test_destroys_process_group_only_when_initialized(self):
        # Calling destroy_process_group() with no group initialised raises;
        # the conditional is the only thing keeping non-distributed runs sane.
        backend = _bare_hf_backend()

        with (
            patch("nemo_safe_synthesizer.training.huggingface_backend.cleanup_memory"),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            backend.teardown()

        destroy.assert_not_called()

        backend2 = _bare_hf_backend()
        with (
            patch("nemo_safe_synthesizer.training.huggingface_backend.cleanup_memory"),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            backend2.teardown()

        destroy.assert_called_once()

    def test_isolates_per_step_failures(self):
        # Each cleanup step is wrapped independently so a torch / NCCL
        # failure in one (e.g. cleanup_memory or destroy_process_group)
        # doesn't prevent the rest from running, and ``_torn_down`` still
        # flips so a retry no-ops instead of compounding the failure.
        backend = _bare_hf_backend()
        backend.trainer = MagicMock()
        backend.model = MagicMock()

        with (
            patch(
                "nemo_safe_synthesizer.training.huggingface_backend.cleanup_memory",
                side_effect=RuntimeError("simulated cleanup failure"),
            ) as cleanup,
            patch("torch.distributed.is_initialized", return_value=True),
            patch(
                "torch.distributed.destroy_process_group",
                side_effect=RuntimeError("simulated nccl teardown failure"),
            ) as destroy,
        ):
            backend.teardown()

        cleanup.assert_called_once()
        destroy.assert_called_once()
        assert backend._torn_down is True
