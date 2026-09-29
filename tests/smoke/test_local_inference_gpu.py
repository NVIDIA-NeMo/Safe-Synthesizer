# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test for the managed local vLLM PII planning server.

Requires CUDA and internet access on first run (downloads SmolLM2-135M-Instruct,
about 270 MB). Checks the server lifecycle, not classification quality.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import httpx
import pandas as pd
import pytest
import torch

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.replace_pii import LLMConfig, ReplacePiiConfig
from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.local_inference import LocalVllmServer, load_profile
from nemo_safe_synthesizer.pii_replacer.planning import resolve_plan

pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.vllm,
    pytest.mark.smollm2,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform != "linux", reason="vLLM serving is Linux-only"),
]


@pytest.fixture
def fixture_tiny_profile_path(test_data_dir: Path) -> Path:
    return test_data_dir / "local_inference" / "smollm2-135m-instruct.yaml"


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_server_starts_answers_and_releases_its_port(fixture_tiny_profile_path: Path) -> None:
    profile = load_profile(fixture_tiny_profile_path)

    with LocalVllmServer(profile) as server:
        environ = server.inference_environ()
        response = httpx.get(
            f"{server.endpoint_url}/models",
            headers={"Authorization": f"Bearer {environ['NSS_INFERENCE_KEY']}"},
        )
        port = int(server.endpoint_url.rsplit(":", 1)[1].split("/", 1)[0])
        assert not _port_is_free(port)

    assert response.status_code == 200
    assert [item["id"] for item in response.json()["data"]] == [profile.served_name]
    assert _port_is_free(port)


def test_planning_stops_the_server_before_returning(
    fixture_tiny_profile_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    port = _free_port()
    monkeypatch.setenv("NSS_INFERENCE_LOCAL_PROFILE", str(fixture_tiny_profile_path))
    monkeypatch.setenv("NSS_INFERENCE_ENDPOINT", f"http://127.0.0.1:{port}/v1")
    monkeypatch.delenv("NSS_INFERENCE_MODEL", raising=False)
    df = pd.DataFrame(
        {
            "name": ["Ada Lovelace", "Grace Hopper", "Alan Turing"],
            "email": ["ada@example.com", "grace@example.com", "alan@example.com"],
        }
    )

    # A 135M model may produce plans that fail semantic validation; this test
    # checks only that the server is gone once planning returns or raises.
    try:
        resolve_plan(df, ReplacePiiConfig(llm=LLMConfig()), DataParameters())
    except (GenerationError, ParameterError):
        pass

    assert _port_is_free(port)
