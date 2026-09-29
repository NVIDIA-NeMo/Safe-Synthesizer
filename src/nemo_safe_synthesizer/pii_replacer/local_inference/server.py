# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start, probe, and stop one managed loopback vLLM server."""

from __future__ import annotations

import importlib.util
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from types import TracebackType
from typing import IO, Self

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from ...errors import GenerationError, InternalError, ParameterError
from ...observability import get_logger, heartbeat
from ..llm_client import InferenceSettings, OpenAICompatibleTransport
from .profile import LocalVllmProfile

__all__ = [
    "DEFAULT_HOST",
    "LocalVllmServer",
    "build_serve_command",
    "is_vllm_installed",
    "local_runtime_problem",
]

logger = get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
_POLL_INTERVAL_SECONDS = 0.5
_MODELS_REQUEST_TIMEOUT_SECONDS = 5.0
_PROBE_TIMEOUT_SECONDS = 300.0
_KILL_WAIT_SECONDS = 10.0
_OUTPUT_TAIL_LINES = 50
_OUTPUT_JOIN_SECONDS = 2.0

# Runs in the child before it execs vLLM. PR_SET_PDEATHSIG (1) asks Linux to
# SIGTERM the server if NSS dies without reaching its cleanup, for example on
# SIGKILL; the setting survives exec. The parent PID check closes the race
# where NSS exits before prctl runs. The signal fires when the thread that
# started the server exits, which cannot happen while its ``with`` block runs.
_PARENT_DEATH_LAUNCHER = """\
import ctypes, os, signal, sys
ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM)
if os.getppid() != int(sys.argv[1]):
    sys.exit(1)
os.execv(sys.executable, [sys.executable, *sys.argv[2:]])
"""


class _ReadinessProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ready: bool


def is_vllm_installed() -> bool:
    """Return whether vLLM can be imported, without importing it."""
    return importlib.util.find_spec("vllm") is not None


def local_runtime_problem() -> str | None:
    """Return why this machine cannot run a managed vLLM server, or ``None`` if it can."""
    if not is_vllm_installed():
        return "vLLM is not installed (install the engine extra, for example `uv sync --extra cu129 --extra engine`)"
    import torch

    if not torch.cuda.is_available():
        return "no CUDA GPU is available"
    return None


def build_serve_command(profile: LocalVllmProfile, *, host: str, port: int, parent_pid: int) -> list[str]:
    """Return the argv that launches ``vllm serve`` for ``profile``.

    The command runs vLLM with the current interpreter, so it uses the same
    environment as NSS. The API key is passed separately through
    ``VLLM_API_KEY`` so it never appears in the process list.

    Args:
        profile: Model and engine settings.
        host: Loopback host to bind.
        port: Port to bind.
        parent_pid: PID the child checks before exec; the server stops when
            this process dies.
    """
    options = [
        "--revision",
        profile.revision,
        "--served-model-name",
        profile.served_name,
        "--host",
        host,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(profile.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(profile.tensor_parallel_size),
    ]
    if profile.max_model_len is not None:
        options += ["--max-model-len", str(profile.max_model_len)]
    if profile.max_num_seqs is not None:
        options += ["--max-num-seqs", str(profile.max_num_seqs)]
    return [
        sys.executable,
        "-c",
        _PARENT_DEATH_LAUNCHER,
        str(parent_pid),
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        profile.model_id,
        *options,
        *profile.extra_args,
    ]


def _launch(command: list[str], environ: Mapping[str, str]) -> subprocess.Popen[str]:
    """Start the server in its own session, merging stderr into a text stdout pipe."""
    return subprocess.Popen(
        command,
        env=dict(environ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


def _available_port(host: str, port: int | None) -> int:
    """Return ``port`` if it can be bound on ``host``, or a free port when ``port`` is ``None``."""
    family, socktype, proto, _, sockaddr = socket.getaddrinfo(host, port or 0, type=socket.SOCK_STREAM)[0]
    with socket.socket(family, socktype, proto) as probe:
        # Match vLLM's own listener so a recently stopped server in TIME_WAIT
        # does not look busy; an active listener still fails to bind.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(sockaddr)
        except OSError as exc:
            raise ParameterError(
                f"Cannot start the local PII inference server on {_url_host(host)}:{port} because the "
                "address is in use. Unset NSS_INFERENCE_LOCAL_PROFILE to use the server already listening "
                "there, or choose another port in NSS_INFERENCE_ENDPOINT."
            ) from exc
        return int(probe.getsockname()[1])


def _signal_group(process_group: int, signum: signal.Signals) -> None:
    try:
        os.killpg(process_group, signum)
    except ProcessLookupError:
        pass


def _group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


class LocalVllmServer:
    """Own one loopback vLLM server process for the duration of a ``with`` block.

    Entering starts ``vllm serve`` in its own process group, forwards its
    output to the NSS log at debug level, and waits until the server lists the
    profile's model and answers one strict-JSON-schema request. Exiting, or
    any startup failure including ``KeyboardInterrupt``, stops the whole
    process group: SIGTERM first, SIGKILL after the profile's shutdown timeout.
    The server also receives SIGTERM if NSS dies without running its cleanup.

    Args:
        profile: Model and engine settings.
        host: Loopback host to bind.
        port: Port to bind, or ``None`` to choose a free port.
        environ: Base environment for the server process; defaults to
            ``os.environ``. ``NSS_INFERENCE_*`` values are not passed on.

    Raises:
        ParameterError: On entry, if vLLM or a CUDA GPU is unavailable, or the port is in use.
        GenerationError: On entry, if the server exits, times out, or fails
            its readiness checks.
    """

    def __init__(
        self,
        profile: LocalVllmProfile,
        *,
        host: str = DEFAULT_HOST,
        port: int | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._profile = profile
        self._host = host
        self._requested_port = port
        self._base_environ: Mapping[str, str] = os.environ if environ is None else environ
        self._api_key = secrets.token_urlsafe(32)
        self._port: int | None = None
        self._process: subprocess.Popen[str] | None = None
        self._output_thread: threading.Thread | None = None
        self._output_tail: deque[str] = deque(maxlen=_OUTPUT_TAIL_LINES)
        self._stopped = False

    @property
    def endpoint_url(self) -> str:
        """OpenAI-compatible base URL, available once the server has started."""
        if self._port is None:
            raise InternalError("The local vLLM server has not started")
        return f"http://{_url_host(self._host)}:{self._port}/v1"

    def inference_environ(self) -> dict[str, str]:
        """Return the base environment with ``NSS_INFERENCE_*`` pointing at this server."""
        return {
            **self._base_environ,
            "NSS_INFERENCE_ENDPOINT": self.endpoint_url,
            "NSS_INFERENCE_KEY": self._api_key,
            "NSS_INFERENCE_MODEL": self._profile.served_name,
        }

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def start(self) -> None:
        """Launch the server and block until it is ready.

        Raises:
            ParameterError: If vLLM or a CUDA GPU is unavailable, or the port is in use.
            GenerationError: If the server exits, times out, or fails readiness checks.
        """
        if self._process is not None or self._stopped:
            raise InternalError("A LocalVllmServer can only be started once")
        if problem := local_runtime_problem():
            raise ParameterError(
                f"LLM-assisted PII planning runs a local vLLM server unless NSS_INFERENCE_ENDPOINT is set, "
                f"but {problem}. Set NSS_INFERENCE_ENDPOINT, plus NSS_INFERENCE_KEY if it requires one, "
                "to use a remote OpenAI-compatible service instead."
            )
        self._port = _available_port(self._host, self._requested_port)
        command = build_serve_command(self._profile, host=self._host, port=self._port, parent_pid=os.getpid())
        logger.user.info(f"Starting local vLLM server for {self._profile.served_name!r} at {self.endpoint_url}")
        started = time.monotonic()
        try:
            self._process = _launch(command, self._server_environ())
            if self._process.stdout is not None:
                self._output_thread = threading.Thread(
                    target=self._forward_output,
                    args=(self._process.stdout,),
                    name="nss-local-vllm-output",
                    daemon=True,
                )
                self._output_thread.start()
            with heartbeat("Local vLLM server startup", logger_name=__name__, model=self._profile.served_name):
                self._wait_until_ready()
        except BaseException:
            self.stop()
            raise
        logger.user.info(
            f"Local vLLM server ready after {time.monotonic() - started:.0f}s",
            extra={"model": self._profile.served_name, "endpoint": self.endpoint_url},
        )

    def stop(self) -> None:
        """Stop the server's process group; safe to call more than once."""
        if self._stopped:
            return
        self._stopped = True
        process = self._process
        if process is None:
            return
        # start_new_session makes the server its own process group leader.
        process_group = process.pid
        if process.poll() is None or _group_alive(process_group):
            _signal_group(process_group, signal.SIGTERM)
            if not self._wait_for_group_exit(process, process_group, self._profile.shutdown_timeout_seconds):
                logger.user.warning(
                    f"Local vLLM server did not stop within {self._profile.shutdown_timeout_seconds:g}s; killing it"
                )
                _signal_group(process_group, signal.SIGKILL)
                if not self._wait_for_group_exit(process, process_group, _KILL_WAIT_SECONDS):
                    logger.user.warning(f"Local vLLM server process group {process_group} is still running")
        if self._output_thread is not None:
            self._output_thread.join(timeout=_OUTPUT_JOIN_SECONDS)
        logger.user.info("Local vLLM server stopped")

    def _server_environ(self) -> dict[str, str]:
        environ = {key: value for key, value in self._base_environ.items() if not key.startswith("NSS_INFERENCE_")}
        environ["VLLM_API_KEY"] = self._api_key
        return environ

    def _forward_output(self, stream: IO[str]) -> None:
        for line in stream:
            text = line.rstrip()
            self._output_tail.append(text)
            logger.runtime.debug(f"vllm: {text}")

    def _output_summary(self) -> str:
        if self._output_thread is not None:
            self._output_thread.join(timeout=_OUTPUT_JOIN_SECONDS)
        if not self._output_tail:
            return ""
        return "\nLast server output:\n" + "\n".join(self._output_tail)

    def _raise_if_exited(self) -> None:
        assert self._process is not None
        returncode = self._process.poll()
        if returncode is not None:
            raise GenerationError(
                f"The local vLLM server exited with code {returncode} before it became ready.{self._output_summary()}"
            )

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._profile.startup_timeout_seconds
        while True:
            self._raise_if_exited()
            if self._model_listed():
                break
            if time.monotonic() >= deadline:
                raise GenerationError(
                    f"The local vLLM server did not become ready within "
                    f"{self._profile.startup_timeout_seconds:g}s.{self._output_summary()}"
                )
            time.sleep(_POLL_INTERVAL_SECONDS)
        self._probe_structured_output()

    def _model_listed(self) -> bool:
        """Return whether ``/models`` lists the served model; ``False`` while the server is starting."""
        try:
            response = httpx.get(
                f"{self.endpoint_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=_MODELS_REQUEST_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            return False
        if response.status_code in {401, 403}:
            raise GenerationError(
                f"The server at {self.endpoint_url} rejected the launch credential; another process may own the port"
            )
        if response.status_code != 200:
            return False
        try:
            served = {item["id"] for item in response.json()["data"]}
        except (ValueError, KeyError, TypeError) as exc:
            raise GenerationError(f"The server at {self.endpoint_url} returned an invalid model list") from exc
        if self._profile.served_name not in served:
            raise GenerationError(
                f"The server at {self.endpoint_url} does not serve {self._profile.served_name!r}; "
                "another process may own the port"
            )
        return True

    def _probe_structured_output(self) -> None:
        settings = InferenceSettings(
            endpoint_url=self.endpoint_url,
            model_id=self._profile.served_name,
            max_workers=1,
            api_key=self._api_key,
        )
        transport = OpenAICompatibleTransport(settings, timeout=_PROBE_TIMEOUT_SECONDS)
        try:
            content = transport.complete(
                messages=[{"role": "user", "content": 'Reply with a JSON object whose "ready" field is true.'}],
                response_model=_ReadinessProbe,
            )
            _ReadinessProbe.model_validate_json(content)
        except (GenerationError, ParameterError, ValidationError) as exc:
            raise GenerationError(
                f"The local vLLM server for {self._profile.served_name!r} started but did not return JSON "
                f"matching a strict schema. Check the profile's extra_args.{self._output_summary()}"
            ) from exc

    def _wait_for_group_exit(self, process: subprocess.Popen[str], process_group: int, timeout: float) -> bool:
        """Wait until every process in the group has exited; return ``False`` on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            # Reap the leader first: an unreaped zombie still counts as a group member.
            process.poll()
            if not _group_alive(process_group):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)
