# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.local_inference import (
    LocalVllmProfile,
    LocalVllmServer,
    build_serve_command,
    local_runtime_problem,
)
from nemo_safe_synthesizer.pii_replacer.local_inference import server as server_module

# Captured at import, before the root conftest's autouse guard replaces it.
REAL_LAUNCH = server_module._launch

PROFILE = LocalVllmProfile(
    model_id="org/tiny",
    revision="abc123",
    served_model_name="tiny",
    max_model_len=4096,
    extra_args=("--enforce-eager",),
    startup_timeout_seconds=30,
    shutdown_timeout_seconds=0.05,
)


@dataclass
class FakeProcessGroup:
    """Process group that exits on SIGTERM (unless told to ignore it) or SIGKILL."""

    pgid: int = 4242
    ignore_sigterm: bool = False
    alive: bool = True
    exit_code: int | None = None
    signals: list[signal.Signals] = field(default_factory=list)

    def killpg(self, pgid: int, signum: int) -> None:
        assert pgid == self.pgid
        if not self.alive:
            raise ProcessLookupError
        if signum == 0:
            return
        self.signals.append(signal.Signals(signum))
        if signum == signal.SIGKILL or not self.ignore_sigterm:
            self.exit(-signum)

    def exit(self, code: int) -> None:
        self.alive = False
        self.exit_code = code


class FakeProcess:
    def __init__(self, group: FakeProcessGroup, output: str) -> None:
        self.pid = group.pgid
        self.stdout = io.StringIO(output)
        self._group = group

    def poll(self) -> int | None:
        return self._group.exit_code


@dataclass
class Harness:
    group: FakeProcessGroup
    popen_calls: list[tuple[list[str], dict[str, str]]] = field(default_factory=list)
    models: list[Callable[[], httpx.Response]] = field(default_factory=list)
    probe_content: str = '{"ready": true}'
    probe_payloads: list[dict[str, object]] = field(default_factory=list)
    probe_timeouts: list[float] = field(default_factory=list)

    @property
    def argv(self) -> list[str]:
        return self.popen_calls[0][0]

    @property
    def child_env(self) -> dict[str, str]:
        return self.popen_calls[0][1]


def _models_response(*names: str) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(200, json={"object": "list", "data": [{"id": name} for name in names]})


@pytest.fixture
def fixture_server_harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    """Fake the vLLM launch, process group, and HTTP endpoints, and record what the server does."""
    state = Harness(group=FakeProcessGroup())

    def launch(argv: list[str], environ: dict[str, str]) -> FakeProcess:
        state.popen_calls.append((argv, environ))
        return FakeProcess(state.group, output="INFO starting\nINFO loading weights\n")

    def get(url: str, **kwargs: object) -> httpx.Response:
        assert url.endswith("/v1/models")
        headers = cast(dict[str, str], kwargs["headers"])
        assert headers["Authorization"] == f"Bearer {state.child_env['VLLM_API_KEY']}"
        assert kwargs["trust_env"] is False
        response = state.models.pop(0) if len(state.models) > 1 else state.models[0]
        return response()

    def post(url: str, **kwargs: object) -> httpx.Response:
        payload = cast(dict[str, object], kwargs["json"])
        assert payload["model"] == "tiny"
        assert kwargs["trust_env"] is False
        state.probe_timeouts.append(cast(float, kwargs["timeout"]))
        state.probe_payloads.append(payload)
        return httpx.Response(200, json={"choices": [{"message": {"content": state.probe_content}}]})

    state.models.append(_models_response("tiny"))
    monkeypatch.setattr(server_module, "local_runtime_problem", lambda profile=None: None)
    monkeypatch.setattr(server_module, "_launch", launch)
    monkeypatch.setattr(server_module.os, "killpg", state.group.killpg)
    monkeypatch.setattr(server_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", post)
    yield state


@pytest.mark.unit
class TestLocalVllmServerLifecycle:
    def test_ready_server_points_planner_at_itself_and_stops_on_exit(
        self, fixture_server_harness: Harness, tmp_path: Path
    ) -> None:
        cache_dir = tmp_path / "outlines"
        base = {
            "PATH": "/bin",
            "NSS_INFERENCE_KEY": "hosted-secret",
            "NSS_INFERENCE_ENDPOINT": "http://x",
            "OUTLINES_CACHE_DIR": str(cache_dir),
            "VLLM_V1_USE_OUTLINES_CACHE": "1",
            "VLLM_DEBUG_LOG_API_SERVER_RESPONSE": "true",
        }

        with LocalVllmServer(PROFILE, port=None, environ=base) as server:
            settings = server.inference_settings(max_workers=4)
            assert fixture_server_harness.group.signals == []

        assert fixture_server_harness.group.signals == [signal.SIGTERM]
        assert settings.endpoint_url == server.endpoint_url
        assert (settings.model_id, settings.max_workers, settings.timeout_seconds) == ("tiny", 4, 60)
        assert settings.api_key == fixture_server_harness.child_env["VLLM_API_KEY"]
        assert fixture_server_harness.child_env["PATH"] == "/bin"
        assert "NSS_INFERENCE_KEY" not in fixture_server_harness.child_env
        assert "NSS_INFERENCE_ENDPOINT" not in fixture_server_harness.child_env
        assert fixture_server_harness.child_env["VLLM_API_KEY"] not in " ".join(fixture_server_harness.argv)
        # The server gets the same diskcache protections as in-process generation.
        assert fixture_server_harness.child_env["VLLM_V1_USE_OUTLINES_CACHE"] == "0"
        assert stat.S_IMODE(cache_dir.stat().st_mode) == 0o700
        # Responses can contain cell samples, so response logging stays off.
        assert fixture_server_harness.child_env["VLLM_DEBUG_LOG_API_SERVER_RESPONSE"] == "false"
        # The schema probe stays within the profile's startup deadline.
        assert 0 < fixture_server_harness.probe_timeouts[0] <= PROFILE.startup_timeout_seconds

    @pytest.mark.parametrize(
        ("request_options", "expected_options"),
        [
            (None, {"temperature": 1.0, "thinking_token_budget": 500}),
            ({"temperature": 0.2, "thinking_token_budget": None}, {"temperature": 0.2}),
        ],
        ids=["profile-options", "given-options"],
    )
    def test_request_options_reach_the_planner_and_the_probe(
        self,
        fixture_server_harness: Harness,
        request_options: dict[str, object] | None,
        expected_options: dict[str, object],
    ) -> None:
        profile = PROFILE.model_copy(
            update={
                "request_options": {"temperature": 1.0, "thinking_token_budget": 500},
                "environment": {"VLLM_USE_V2_MODEL_RUNNER": "0"},
            }
        )

        with LocalVllmServer(profile, request_options=request_options) as server:
            planner_options = server.inference_settings().request_options

        probe_options = {
            key: value
            for key, value in fixture_server_harness.probe_payloads[0].items()
            if key not in {"model", "messages", "response_format"}
        }
        assert planner_options == expected_options
        assert probe_options == expected_options
        assert fixture_server_harness.child_env["VLLM_USE_V2_MODEL_RUNNER"] == "0"

    def test_polls_until_the_model_is_listed(self, fixture_server_harness: Harness) -> None:
        def refused() -> httpx.Response:
            raise httpx.ConnectError("refused")

        fixture_server_harness.models[:] = [refused, lambda: httpx.Response(503), _models_response("tiny")]

        with LocalVllmServer(PROFILE):
            pass

        assert fixture_server_harness.group.signals == [signal.SIGTERM]

    def test_early_exit_reports_server_output(self, fixture_server_harness: Harness) -> None:
        def crashed() -> httpx.Response:
            fixture_server_harness.group.exit(1)
            raise httpx.ConnectError("refused")

        fixture_server_harness.models[:] = [crashed]

        with pytest.raises(GenerationError, match=r"exited with code 1(.|\n)*INFO loading weights"):
            LocalVllmServer(PROFILE).start()

    def test_startup_timeout_stops_the_server(self, fixture_server_harness: Harness) -> None:
        fixture_server_harness.models[:] = [lambda: httpx.Response(503)]
        profile = PROFILE.model_copy(update={"startup_timeout_seconds": 0.01})

        with pytest.raises(GenerationError, match="did not become ready within"):
            LocalVllmServer(profile).start()

        assert fixture_server_harness.group.signals == [signal.SIGTERM]

    @pytest.mark.parametrize(
        ("response", "match"),
        [
            pytest.param(_models_response("other"), "does not serve 'tiny'", id="wrong-model"),
            pytest.param(lambda: httpx.Response(401), "rejected the launch credential", id="foreign-server"),
            pytest.param(lambda: httpx.Response(200, json={"models": []}), "invalid model list", id="bad-list"),
        ],
    )
    def test_foreign_or_invalid_servers_fail_fast(
        self,
        fixture_server_harness: Harness,
        response: Callable[[], httpx.Response],
        match: str,
    ) -> None:
        fixture_server_harness.models[:] = [response]

        with pytest.raises(GenerationError, match=match):
            LocalVllmServer(PROFILE).start()

        assert fixture_server_harness.group.signals == [signal.SIGTERM]

    def test_schema_probe_failure_stops_the_server(self, fixture_server_harness: Harness) -> None:
        fixture_server_harness.probe_content = '{"state": "ok"}'

        with pytest.raises(GenerationError, match="did not return JSON matching a strict schema"):
            LocalVllmServer(PROFILE).start()

        assert fixture_server_harness.group.signals == [signal.SIGTERM]

    def test_interrupt_during_startup_stops_the_server(self, fixture_server_harness: Harness) -> None:
        def interrupted() -> httpx.Response:
            raise KeyboardInterrupt

        fixture_server_harness.models[:] = [interrupted]

        with pytest.raises(KeyboardInterrupt):
            LocalVllmServer(PROFILE).start()

        assert fixture_server_harness.group.signals == [signal.SIGTERM]

    def test_server_ignoring_sigterm_is_killed(self, fixture_server_harness: Harness) -> None:
        fixture_server_harness.group.ignore_sigterm = True

        with LocalVllmServer(PROFILE):
            pass

        assert fixture_server_harness.group.signals == [signal.SIGTERM, signal.SIGKILL]
        assert not fixture_server_harness.group.alive

    def test_unavailable_runtime_fails_before_launch_and_points_to_an_endpoint(
        self,
        fixture_server_harness: Harness,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(server_module, "local_runtime_problem", lambda profile=None: "no CUDA GPU is available")

        with pytest.raises(ParameterError, match="no CUDA GPU is available. Set NSS_INFERENCE_ENDPOINT"):
            LocalVllmServer(PROFILE).start()

        assert fixture_server_harness.popen_calls == []

    def test_busy_port_fails_before_launch(self, fixture_server_harness: Harness) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]

            with pytest.raises(ParameterError, match="address is in use"):
                LocalVllmServer(PROFILE, port=port).start()

        assert fixture_server_harness.popen_calls == []

    def test_requested_free_port_is_used(self, fixture_server_harness: Harness) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]

        with LocalVllmServer(PROFILE, port=port) as server:
            assert server.endpoint_url == f"http://127.0.0.1:{port}/v1"


@pytest.mark.unit
class TestLocalRuntimeProblem:
    @pytest.fixture
    def fixture_visible_gpus(self, monkeypatch: pytest.MonkeyPatch) -> list[float]:
        """Fake vLLM as installed and CUDA as available; return the visible GPUs' memory in GiB to fill in."""
        import torch

        gpus: list[float] = []
        monkeypatch.setattr(server_module, "is_vllm_installed", lambda: True)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: len(gpus))
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda index: SimpleNamespace(name="Fake GPU", total_memory=int(gpus[index] * 1024**3)),
        )
        return gpus

    @pytest.mark.parametrize(
        ("gpus", "tensor_parallel_size", "match"),
        [
            pytest.param([79.25], 1, None, id="80gb-gpu"),
            pytest.param([44.4], 1, r"at least 75 GiB, but GPU 0 \(Fake GPU\) has 44.4 GiB", id="small-gpu"),
            pytest.param([79.25], 2, "needs 2 GPUs, but only 1 is visible", id="too-few-gpus"),
            pytest.param([79.25, 44.4], 2, "GPU 1", id="second-gpu-too-small"),
        ],
    )
    def test_profile_needs_enough_gpus_with_enough_memory(
        self,
        fixture_visible_gpus: list[float],
        gpus: list[float],
        tensor_parallel_size: int,
        match: str | None,
    ) -> None:
        fixture_visible_gpus.extend(gpus)
        profile = PROFILE.model_copy(update={"min_gpu_memory_gib": 75, "tensor_parallel_size": tensor_parallel_size})

        problem = local_runtime_problem(profile)

        if match is None:
            assert problem is None
        else:
            assert problem is not None
            assert re.search(match, problem)

    def test_profile_without_minimum_skips_the_memory_check(self, fixture_visible_gpus: list[float]) -> None:
        fixture_visible_gpus.append(8.0)

        assert local_runtime_problem(PROFILE) is None


@pytest.mark.unit
def test_launch_starts_the_server_in_its_own_session(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(server_module.subprocess, "Popen", lambda argv, **kwargs: calls.append(kwargs))

    REAL_LAUNCH(["vllm"], {"A": "1"})

    assert calls[0]["start_new_session"] is True
    assert calls[0]["env"] == {"A": "1"}
    assert calls[0]["stderr"] is subprocess.STDOUT


@pytest.mark.unit
class TestServeCommand:
    def test_runs_vllm_serve_with_profile_settings_through_the_current_interpreter(self) -> None:
        argv = build_serve_command(PROFILE, host="::1", port=8123, parent_pid=77)

        assert argv[0] == sys.executable
        assert argv[3] == "77"
        serve_args = argv[argv.index("serve") + 1 :]
        assert serve_args == [
            "org/tiny",
            "--revision",
            "abc123",
            "--served-model-name",
            "tiny",
            "--host",
            "::1",
            "--port",
            "8123",
            "--gpu-memory-utilization",
            "0.9",
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "4096",
            "--enforce-eager",
        ]

    @pytest.mark.skipif(sys.platform != "linux", reason="parent-death signal is Linux-only")
    def test_launcher_execs_the_target_when_the_parent_is_alive(self) -> None:
        launcher = build_serve_command(PROFILE, host="127.0.0.1", port=1, parent_pid=os.getpid())[:4]
        result = subprocess.run(
            [*launcher, "-c", "import json, sys; print(json.dumps(sys.argv))", "x"],
            capture_output=True,
            text=True,
            check=True,
        )

        assert json.loads(result.stdout) == ["-c", "x"]

    @pytest.mark.skipif(sys.platform != "linux", reason="parent-death signal is Linux-only")
    def test_launcher_exits_when_its_parent_already_changed(self) -> None:
        launcher = build_serve_command(PROFILE, host="127.0.0.1", port=1, parent_pid=1)[:4]
        result = subprocess.run([*launcher, "-c", "print('ran')"], capture_output=True, text=True)

        assert result.returncode == 1
        assert result.stdout == ""
