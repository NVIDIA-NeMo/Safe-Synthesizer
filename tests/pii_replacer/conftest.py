# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType

import pytest

from nemo_safe_synthesizer.pii_replacer.llm_client import InferenceSettings
from nemo_safe_synthesizer.pii_replacer.local_inference import LocalVllmProfile
from nemo_safe_synthesizer.pii_replacer.local_inference import managed as managed_module


class FakeLocalServer:
    """Managed-server stand-in that records its launch settings and whether it is running."""

    def __init__(
        self,
        profile: LocalVllmProfile,
        *,
        host: str,
        port: int | None,
        environ: Mapping[str, str] | None,
        request_options: Mapping[str, object] | None,
    ) -> None:
        self.profile = profile
        self.host = host
        self.port = port
        self.request_options = request_options
        self.running = False

    def __enter__(self) -> FakeLocalServer:
        self.running = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.running = False

    def inference_settings(self, *, max_workers: int = 1, timeout_seconds: float | None = None) -> InferenceSettings:
        return InferenceSettings(
            endpoint_url="http://127.0.0.1:8123/v1",
            model_id=self.profile.served_name,
            max_workers=max_workers,
            timeout_seconds=self.profile.request_timeout_seconds if timeout_seconds is None else timeout_seconds,
            request_options=dict(self.request_options or {}),
        )


@pytest.fixture
def fixture_fake_local_servers(monkeypatch: pytest.MonkeyPatch) -> list[FakeLocalServer]:
    """Replace the managed vLLM server with fakes and return each one created, in order."""
    created: list[FakeLocalServer] = []

    def make_server(
        profile: LocalVllmProfile,
        *,
        host: str,
        port: int | None,
        environ: Mapping[str, str] | None,
        request_options: Mapping[str, object] | None,
    ) -> FakeLocalServer:
        created.append(FakeLocalServer(profile, host=host, port=port, environ=environ, request_options=request_options))
        return created[-1]

    monkeypatch.setattr(managed_module, "LocalVllmServer", make_server)
    return created
