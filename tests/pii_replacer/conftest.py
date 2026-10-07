# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType

import pytest

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
    ) -> None:
        self.profile = profile
        self.host = host
        self.port = port
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

    def inference_environ(self) -> dict[str, str]:
        return {"NSS_INFERENCE_ENDPOINT": "http://127.0.0.1:8123/v1", "NSS_INFERENCE_MODEL": self.profile.served_name}


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
    ) -> FakeLocalServer:
        created.append(FakeLocalServer(profile, host=host, port=port, environ=environ))
        return created[-1]

    monkeypatch.setattr(managed_module, "LocalVllmServer", make_server)
    return created
