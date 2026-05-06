#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "pydantic",
#     "starlette",
#     "structlog",
#     "uvicorn",
# ]
# ///
# pyright: reportMissingImports=false
"""Guard proxy for checking that a command does not contact Hugging Face.

Warning: This is an agent-made diagnostic tool with minimal test coverage. It
has worked for the NSS/Hugging Face checks it was built for, but it is not a
general-purpose HTTP proxy and may miss clients that ignore proxy variables.

The proxy is intentionally not a forwarding proxy. It records requests, blocks
Hugging Face destinations, and returns 502 for other proxied traffic. Use it to
confirm that an offline/local-only code path stays local, or to prove where a
real Hugging Face request first occurs.

One-shot command wrapper::

    uv run --script tools/hf_network_guard_proxy.py run -- \
      uv run --frozen pytest tests/llm/test_utils.py -n0

Manual watcher mode::

    uv run --script tools/hf_network_guard_proxy.py serve --port 8765

    export HTTP_PROXY='http://127.0.0.1:8765'
    export HTTPS_PROXY='http://127.0.0.1:8765'
    export ALL_PROXY='http://127.0.0.1:8765'
    export NO_PROXY='127.0.0.1,localhost'
    uv run --frozen pytest tests/llm/test_utils.py -n0

NSS uncached-model capture example::

    DATA="$HOME/dev/data/your-data.csv"
    MODEL="nvidia/Nemotron-4-340B-Instruct"
    ARTIFACTS="$HOME/dev/nss-proxy-capture"

    uv run --script tools/hf_network_guard_proxy.py run -- \
      uv run --frozen --extra engine --group dev safe-synthesizer run train \
        --data-source "$DATA" \
        --artifact-path "$ARTIFACTS" \
        --training__pretrained_model "$MODEL" \
        --training__attn_implementation sdpa \
        --replace_pii__globals__classify__enable_classify false \
        --log-format plain \
        -vv

Expected pass output includes::

    hf network guard passed
    No proxied requests observed.

Expected Hugging Face capture output includes::

    Hugging Face network requests observed:
    - CONNECT huggingface.co huggingface.co:443

For machine-readable output, add ``--json`` before the command separator::

    uv run --script tools/hf_network_guard_proxy.py run --json -- <command>
"""

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from urllib.parse import urlsplit

import click
import structlog
import uvicorn
from pydantic import BaseModel, ConfigDict, Field
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
HF_DOMAINS = ("huggingface.co", "hf.co")
STARTUP_TIMEOUT_SECONDS = 5.0

logger = structlog.get_logger(__name__)


class ExitCode(IntEnum):
    """Process exit codes."""

    ok = 0
    failure = 1
    bad_input = 125


class ProxyRequest(BaseModel):
    """Single request observed by the guard proxy."""

    model_config = ConfigDict(frozen=True)

    method: str = Field(description="HTTP method observed by the proxy.")
    target: str = Field(description="Raw proxy request target.")
    host: str = Field(description="Destination host inferred from the request.")
    is_huggingface: bool = Field(description="Whether the host belongs to Hugging Face.")


class ProxySummary(BaseModel):
    """Summary of requests observed by the guard proxy."""

    request_count: int = Field(description="Total proxied requests observed.")
    huggingface_request_count: int = Field(description="Hugging Face requests observed.")
    requests: list[ProxyRequest] = Field(description="Observed proxy requests.")

    @property
    def huggingface_requests(self) -> list[ProxyRequest]:
        """Return only requests that targeted Hugging Face."""
        return [request for request in self.requests if request.is_huggingface]


class ProxyEnv(BaseModel):
    """Environment variables needed to route traffic through the guard."""

    proxy_url: str = Field(description="Proxy URL.")
    variables: dict[str, str] = Field(description="Proxy-related environment variables.")


@dataclass(slots=True)
class ProxyState:
    """Thread-safe request log for the guard proxy."""

    requests: list[ProxyRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def record(self, request: ProxyRequest) -> None:
        """Record an observed request."""
        with self.lock:
            self.requests.append(request)

    def summary(self) -> ProxySummary:
        """Return a stable summary of observed requests."""
        with self.lock:
            requests = list(self.requests)
        return ProxySummary(
            request_count=len(requests),
            huggingface_request_count=sum(request.is_huggingface for request in requests),
            requests=requests,
        )


@dataclass(slots=True)
class RunningServer:
    """Background uvicorn server instance."""

    server: uvicorn.Server
    thread: threading.Thread
    host: str
    port: int
    state: ProxyState

    @property
    def proxy_url(self) -> str:
        """Return the URL clients should use for proxy variables."""
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        """Stop the background server."""
        self.server.should_exit = True
        self.thread.join(timeout=STARTUP_TIMEOUT_SECONDS)
        if self.thread.is_alive():
            logger.warning("guard proxy server thread did not stop promptly")


class GuardProxyMiddleware:
    """Record and reject all proxied traffic before Starlette routing."""

    def __init__(self, app: ASGIApp, state: ProxyState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or _is_status_request(scope):
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive)
        proxy_request = _proxy_request_from_starlette(request)
        self.state.record(proxy_request)
        response = _proxy_response(proxy_request)
        await response(scope, receive, send)


def _create_app(state: ProxyState) -> Starlette:
    async def status(_: Request) -> JSONResponse:
        return JSONResponse(state.summary().model_dump())

    starlette_app = Starlette(routes=[Route("/__status__", status, methods=["GET"])])
    starlette_app.add_middleware(GuardProxyMiddleware, state=state)
    return starlette_app


def _is_status_request(scope: Scope) -> bool:
    return scope.get("method") == "GET" and scope.get("path") == "/__status__"


def _proxy_response(request: ProxyRequest) -> Response:
    if request.is_huggingface:
        return Response(f"Blocked Hugging Face request to {request.host}\n", status_code=502)
    return Response("Guard proxy does not forward requests\n", status_code=502)


def _proxy_request_from_starlette(request: Request) -> ProxyRequest:
    target = _request_target(request)
    host = _request_host(request.method, target, request.headers.get("host", ""))
    return ProxyRequest(
        method=request.method,
        target=target,
        host=host,
        is_huggingface=_is_huggingface_host(host),
    )


def _request_target(request: Request) -> str:
    raw_path = request.scope.get("raw_path", b"")
    if isinstance(raw_path, bytes) and raw_path:
        return raw_path.decode("latin-1")
    return request.url.path


def _request_host(method: str, target: str, host_header: str) -> str:
    if method == "CONNECT":
        return target.rsplit(":", 1)[0].strip("[]").casefold()

    parsed = urlsplit(target)
    if parsed.hostname:
        return parsed.hostname.casefold()
    return host_header.rsplit(":", 1)[0].strip("[]").casefold()


def _is_huggingface_host(host: str) -> bool:
    return any(host == domain or host.endswith(f".{domain}") for domain in HF_DOMAINS)


def _start_server(host: str, port: int) -> RunningServer:
    state = ProxyState()
    config = uvicorn.Config(
        _create_app(state),
        host=host,
        port=port,
        access_log=False,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    actual_host, actual_port = _wait_for_server(server, thread)
    return RunningServer(server=server, thread=thread, host=actual_host, port=actual_port, state=state)


def _wait_for_server(server: uvicorn.Server, thread: threading.Thread) -> tuple[str, int]:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if server.started and server.servers:
            sockname = server.servers[0].sockets[0].getsockname()
            return str(sockname[0]), int(sockname[1])
        if not thread.is_alive():
            raise RuntimeError("guard proxy server stopped during startup")
        time.sleep(0.01)
    raise TimeoutError("guard proxy server did not start")


def _proxy_env(proxy_url: str) -> ProxyEnv:
    variables = {
        "HTTP_PROXY": proxy_url,
        "HTTPS_PROXY": proxy_url,
        "ALL_PROXY": proxy_url,
        "http_proxy": proxy_url,
        "https_proxy": proxy_url,
        "all_proxy": proxy_url,
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    return ProxyEnv(proxy_url=proxy_url, variables=variables)


def _write_env_instructions(env: ProxyEnv) -> None:
    sys.stderr.write(f"HF network guard proxy listening on {env.proxy_url}\n")
    sys.stderr.write("Export these variables for a manual run:\n")
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        sys.stderr.write(f"export {key}={env.variables[key]!r}\n")
    sys.stderr.write(f"Status endpoint: {env.proxy_url}/__status__\n")


def _write_run_summary(summary: ProxySummary, *, as_json: bool) -> None:
    if as_json:
        sys.stdout.write(summary.model_dump_json(indent=2) + "\n")
        return
    _write_human_summary(summary)


def _log_guard_pass(summary: ProxySummary) -> None:
    if summary.huggingface_requests:
        return
    logger.info(
        "hf network guard passed",
        request_count=summary.request_count,
        proxied_request_count=summary.request_count,
    )


def _write_human_summary(summary: ProxySummary) -> None:
    hf_requests = summary.huggingface_requests
    if hf_requests:
        sys.stderr.write("Hugging Face network requests observed:\n")
        for request in hf_requests:
            sys.stderr.write(f"- {request.method} {request.host} {request.target}\n")
        return

    if summary.requests:
        sys.stderr.write(f"No Hugging Face requests observed. Total proxied requests: {summary.request_count}\n")
    else:
        sys.stderr.write("No proxied requests observed.\n")


def test_proxy_response_blocks_huggingface_request() -> None:
    request = ProxyRequest(
        method="CONNECT",
        target="huggingface.co:443",
        host="huggingface.co",
        is_huggingface=True,
    )

    response = _proxy_response(request)

    assert response.status_code == 502
    assert response.body == b"Blocked Hugging Face request to huggingface.co\n"


def test_proxy_response_rejects_non_huggingface_without_forwarding() -> None:
    request = ProxyRequest(
        method="GET",
        target="http://example.com/models",
        host="example.com",
        is_huggingface=False,
    )

    response = _proxy_response(request)

    assert response.status_code == 502
    assert response.body == b"Guard proxy does not forward requests\n"


def test_proxy_state_summary_counts_huggingface_and_non_huggingface_requests() -> None:
    state = ProxyState()
    huggingface_request = ProxyRequest(
        method="CONNECT",
        target="huggingface.co:443",
        host="huggingface.co",
        is_huggingface=True,
    )
    non_huggingface_request = ProxyRequest(
        method="GET",
        target="http://example.com/models",
        host="example.com",
        is_huggingface=False,
    )

    state.record(huggingface_request)
    state.record(non_huggingface_request)
    summary = state.summary()
    state.record(non_huggingface_request)

    assert summary.request_count == 2
    assert summary.huggingface_request_count == 1
    assert summary.requests == [huggingface_request, non_huggingface_request]
    assert summary.huggingface_requests == [huggingface_request]


def _normalize_command(command: tuple[str, ...]) -> list[str]:
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    return normalized


@click.group(help=__doc__)
def cli() -> None:
    """Command line entry point."""


@cli.command()
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Host interface for the guard proxy.")
@click.option("--port", default=DEFAULT_PORT, show_default=True, type=int, help="Port for the guard proxy.")
def serve(*, host: str, port: int) -> None:
    """Run the guard proxy until interrupted."""
    server = _start_server(host, port)
    _write_env_instructions(_proxy_env(server.proxy_url))
    try:
        while server.thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()


def run(
    *command: str,
    host: str = DEFAULT_HOST,
    port: int = 0,
    json_output: bool = False,
) -> int:
    """Run a command through the guard proxy."""
    normalized = _normalize_command(command)
    if not normalized:
        sys.stderr.write("error: missing command after `run --`\n")
        return int(ExitCode.bad_input)

    server = _start_server(host, port)
    try:
        completed = subprocess.run(
            normalized, env={**os.environ, **_proxy_env(server.proxy_url).variables}, check=False
        )
        summary = server.state.summary()
    finally:
        server.stop()

    if not json_output:
        _log_guard_pass(summary)
    _write_run_summary(summary, as_json=json_output)
    if summary.huggingface_requests:
        return int(ExitCode.failure)
    return completed.returncode


@cli.command("run", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Host interface for the guard proxy.")
@click.option(
    "--port", default=0, show_default=True, type=int, help="Port for the guard proxy. Use 0 for any free port."
)
@click.option("--json", "json_output", is_flag=True, help="Write a machine-readable JSON summary to stdout.")
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def run_cli(command: tuple[str, ...], *, host: str, port: int, json_output: bool) -> None:
    """Run a command through the guard proxy."""
    raise click.exceptions.Exit(run(*command, host=host, port=port, json_output=json_output))


def main() -> int:
    return int(cli(standalone_mode=False) or ExitCode.ok)


if __name__ == "__main__":
    raise SystemExit(main())
