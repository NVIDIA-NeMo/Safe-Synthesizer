# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in smoke test for sending one telemetry payload to the configured endpoint."""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from nemo_safe_synthesizer.telemetry import (
    NSSTrainingAndGenerationEvent,
    QueuedEvent,
    TaskStatusEnum,
    _telemetry_endpoint,
    build_payload,
)

pytestmark = pytest.mark.smoke


def _telemetry_smoke_enabled() -> bool:
    return os.getenv("NSS_TELEMETRY_SMOKE_SEND", "").lower() in {"1", "true", "yes"}


@pytest.mark.skipif(
    not _telemetry_smoke_enabled(),
    reason="Set NSS_TELEMETRY_SMOKE_SEND=1 to send a real telemetry smoke event.",
)
async def test_telemetry_endpoint_accepts_smoke_event() -> None:
    """Send a single telemetry event to the configured endpoint.

    This is intentionally opt-in because it contacts the live telemetry endpoint
    unless ``NEMO_TELEMETRY_ENDPOINT`` points at a local or controlled server.
    """
    httpx = pytest.importorskip("httpx")

    event = NSSTrainingAndGenerationEvent(
        task="telemetry_smoke",
        task_status=TaskStatusEnum.COMPLETED,
    )
    payload = build_payload(
        [QueuedEvent(event=event, timestamp=datetime.now(timezone.utc))],
        source_client_version="telemetry-smoke",
        session_id=os.getenv("NEMO_SESSION_ID", "telemetry-smoke"),
    )
    endpoint = _telemetry_endpoint()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(endpoint, json=payload)

    print(f"Telemetry smoke POST {endpoint} -> HTTP {response.status_code}")
    if response.text:
        print(response.text[:1000])

    assert response.is_success, f"telemetry endpoint rejected smoke event: HTTP {response.status_code}"
