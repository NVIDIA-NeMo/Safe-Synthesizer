# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
import json
import os
import signal
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import cast

import httpx
import pytest

from nemo_safe_synthesizer.errors import GenerationError, ParameterError
from nemo_safe_synthesizer.pii_replacer.local_inference import LocalVllmProfile, LocalVllmServer, build_serve_command
from nemo_safe_synthesizer.pii_replacer.local_inference import server as server_module

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
    popen_calls: list[tuple[list[str], dict[str, object]]] = field(default_factory=list)
    models: list[Callable[[], httpx.Response]] = field(default_factory=list)
    probe_content: str = '{"ready": true}'

    @property
    def argv(self) -> list[str]:
        return self.popen_calls[0][0]

    @property
    def child_env(self) -> dict[str, str]:
        return cast(dict[str, str], self.popen_calls[0][1]["env"])


def _models_response(*names: str) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(200, json={"object": "list", "data": [{"id": name} for name in names]})


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Iterator[Harness]:
    state = Harness(group=FakeProcessGroup())

    def popen(argv: list[str], **kwargs: object) -> FakeProcess:
        state.popen_calls.append((argv, kwargs))
        return FakeProcess(state.group, output="INFO starting\nINFO loading weights\n")

    def get(url: str, **kwargs: object) -> httpx.Response:
        assert url.endswith("/v1/models")
        headers = cast(dict[str, str], kwargs["headers"])
        assert headers["Authorization"] == f"Bearer {state.child_env['VLLM_API_KEY']}"
        response = state.models.pop(0) if len(state.models) > 1 else state.models[0]
        return response()

    def post(url: str, **kwargs: object) -> httpx.Response:
        payload = cast(dict[str, object], kwargs["json"])
        assert payload["model"] == "tiny"
        return httpx.Response(200, json={"choices": [{"message": {"content": state.probe_content}}]})

    state.models.append(_models_response("tiny"))
    monkeypatch.setattr(server_module, "is_vllm_installed", lambda: True)
    monkeypatch.setattr(server_module.subprocess, "Popen", popen)
    monkeypatch.setattr(server_module.os, "killpg", state.group.killpg)
    monkeypatch.setattr(server_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setattr(httpx, "post", post)
    yield state


@pytest.mark.unit
class TestLocalVllmServerLifecycle:
    def test_ready_server_points_planner_at_itself_and_stops_on_exit(self, harness: Harness) -> None:
        base = {"PATH": "/bin", "NSS_INFERENCE_KEY": "hosted-secret", "NSS_INFERENCE_ENDPOINT": "http://x"}

        with LocalVllmServer(PROFILE, port=None, environ=base) as server:
            environ = server.inference_environ()
            assert harness.group.signals == []

        assert harness.group.signals == [signal.SIGTERM]
        assert environ["NSS_INFERENCE_ENDPOINT"] == server.endpoint_url
        assert environ["NSS_INFERENCE_MODEL"] == "tiny"
        assert environ["NSS_INFERENCE_KEY"] == harness.child_env["VLLM_API_KEY"]
        assert environ["PATH"] == "/bin"
        assert "NSS_INFERENCE_KEY" not in harness.child_env
        assert "NSS_INFERENCE_ENDPOINT" not in harness.child_env
        assert harness.child_env["VLLM_API_KEY"] not in " ".join(harness.argv)
        assert harness.popen_calls[0][1]["start_new_session"] is True

    def test_polls_until_the_model_is_listed(self, harness: Harness) -> None:
        def refused() -> httpx.Response:
            raise httpx.ConnectError("refused")

        harness.models[:] = [refused, lambda: httpx.Response(503), _models_response("tiny")]

        with LocalVllmServer(PROFILE):
            pass

        assert harness.group.signals == [signal.SIGTERM]

    def test_early_exit_reports_server_output(self, harness: Harness) -> None:
        def crashed() -> httpx.Response:
            harness.group.exit(1)
            raise httpx.ConnectError("refused")

        harness.models[:] = [crashed]

        with pytest.raises(GenerationError, match=r"exited with code 1(.|\n)*INFO loading weights"):
            LocalVllmServer(PROFILE).start()

    def test_startup_timeout_stops_the_server(self, harness: Harness) -> None:
        harness.models[:] = [lambda: httpx.Response(503)]
        profile = PROFILE.model_copy(update={"startup_timeout_seconds": 0.01})

        with pytest.raises(GenerationError, match="did not become ready within"):
            LocalVllmServer(profile).start()

        assert harness.group.signals == [signal.SIGTERM]

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
        harness: Harness,
        response: Callable[[], httpx.Response],
        match: str,
    ) -> None:
        harness.models[:] = [response]

        with pytest.raises(GenerationError, match=match):
            LocalVllmServer(PROFILE).start()

        assert harness.group.signals == [signal.SIGTERM]

    @pytest.mark.parametrize("content", ["not json", '{"ready": true, "extra": 1}', '{"state": "ok"}'])
    def test_schema_probe_failure_stops_the_server(self, harness: Harness, content: str) -> None:
        harness.probe_content = content

        with pytest.raises(GenerationError, match="did not return JSON matching a strict schema"):
            LocalVllmServer(PROFILE).start()

        assert harness.group.signals == [signal.SIGTERM]

    def test_interrupt_during_startup_stops_the_server(self, harness: Harness) -> None:
        def interrupted() -> httpx.Response:
            raise KeyboardInterrupt

        harness.models[:] = [interrupted]

        with pytest.raises(KeyboardInterrupt):
            LocalVllmServer(PROFILE).start()

        assert harness.group.signals == [signal.SIGTERM]

    def test_server_ignoring_sigterm_is_killed(self, harness: Harness) -> None:
        harness.group.ignore_sigterm = True

        with LocalVllmServer(PROFILE):
            pass

        assert harness.group.signals == [signal.SIGTERM, signal.SIGKILL]
        assert not harness.group.alive

    def test_stop_is_idempotent(self, harness: Harness) -> None:
        server = LocalVllmServer(PROFILE)
        server.start()
        server.stop()
        server.stop()

        assert harness.group.signals == [signal.SIGTERM]

    def test_missing_vllm_fails_before_launch(self, harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server_module, "is_vllm_installed", lambda: False)

        with pytest.raises(ParameterError, match="requires vLLM"):
            LocalVllmServer(PROFILE).start()

        assert harness.popen_calls == []

    def test_busy_port_fails_before_launch(self, harness: Harness) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]

            with pytest.raises(ParameterError, match="address is in use"):
                LocalVllmServer(PROFILE, port=port).start()

        assert harness.popen_calls == []

    def test_requested_free_port_is_used(self, harness: Harness) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]

        with LocalVllmServer(PROFILE, port=port) as server:
            assert server.endpoint_url == f"http://127.0.0.1:{port}/v1"


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
