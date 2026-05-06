#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "click",
#     "pydantic",
#     "proxy.py>=2.4.10",
#     "structlog",
# ]
# ///
# pyright: reportMissingImports=false
"""Guard proxy for checking that a command does not contact Hugging Face.

Warning: This is an agent-made diagnostic tool with minimal test coverage. It
has worked for the NSS/Hugging Face checks it was built for, but it is not a
general-purpose HTTP proxy and may miss clients that ignore proxy variables.

The proxy records requests and blocks Hugging Face destinations by default.
Other proxied traffic passes through and is included in the summary. Use it to
confirm that an offline/local-only code path stays local, or to prove where a
real Hugging Face request first occurs. Pass ``--allow-passthrough-hf`` when you
need Hugging Face requests to complete while still tracking them.

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
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import IntEnum
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from typing import Any, ClassVar
from unittest.mock import patch

import click
import structlog
from proxy import Proxy
from proxy.http import httpStatusCodes
from proxy.http.exception import HttpRequestRejected
from proxy.http.parser import HttpParser
from proxy.http.proxy import HttpProxyBasePlugin
from pydantic import BaseModel, ConfigDict, Field

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

    @property
    def other_requests(self) -> list[ProxyRequest]:
        """Return proxied requests that did not target Hugging Face."""
        return [request for request in self.requests if not request.is_huggingface]


class ProxyEnv(BaseModel):
    """Environment variables needed to route traffic through the guard."""

    proxy_url: str = Field(description="Proxy URL.")
    variables: dict[str, str] = Field(description="Proxy-related environment variables.")


@dataclass(slots=True)
class ProxyState:
    """Thread-safe request log for the guard proxy."""

    requests: Any = field(default_factory=list)
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
    """Background proxy.py server instance."""

    server: Proxy
    host: str
    port: int
    state: ProxyState
    allow_passthrough_hf: bool
    manager: SyncManager | None = None

    @property
    def proxy_url(self) -> str:
        """Return the URL clients should use for proxy variables."""
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        """Stop the background server."""
        self.server.__exit__(None, None, None)
        if self.manager is not None:
            self.manager.shutdown()


class GuardProxyPlugin(HttpProxyBasePlugin):
    """Record proxied traffic and optionally pass through Hugging Face requests."""

    state: ClassVar[ProxyState] = ProxyState()
    allow_passthrough_hf: ClassVar[bool] = False

    @classmethod
    def configure(cls, *, state: ProxyState, allow_passthrough_hf: bool) -> None:
        """Configure shared state for the next embedded proxy instance."""
        cls.state = state
        cls.allow_passthrough_hf = allow_passthrough_hf

    def before_upstream_connection(self, request: HttpParser) -> HttpParser | None:
        """Record the destination and decide whether proxy.py may connect upstream."""
        proxy_request = _proxy_request_from_proxy_py(request)
        self.state.record(proxy_request)
        if not proxy_request.is_huggingface or self.allow_passthrough_hf:
            return request
        raise _proxy_rejection(proxy_request)


def _proxy_rejection(request: ProxyRequest) -> HttpRequestRejected:
    return HttpRequestRejected(
        status_code=httpStatusCodes.BAD_GATEWAY,
        reason=b"Bad Gateway",
        headers={b"content-type": b"text/plain; charset=utf-8"},
        body=_proxy_response_body(request),
    )


def _proxy_response_body(request: ProxyRequest) -> bytes:
    if request.is_huggingface:
        return f"Blocked Hugging Face request to {request.host}\n".encode()
    return b"Guard proxy blocked request\n"


def _proxy_request_from_proxy_py(request: HttpParser) -> ProxyRequest:
    method = _request_value(request.method)
    host = _normalize_host(_request_value(request.host))
    target = _proxy_py_target(request, method=method, host=host)
    return ProxyRequest(
        method=method,
        target=target,
        host=host,
        is_huggingface=_is_huggingface_host(host),
    )


def _proxy_py_target(request: HttpParser, *, method: str, host: str) -> str:
    if method == "CONNECT":
        return f"{host}:{request.port}"
    path = _request_value(request.path)
    if path:
        return path
    return host


def _request_value(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return value


def _normalize_host(host: str) -> str:
    return host.strip().strip("[]").rstrip(".").casefold()


def _is_huggingface_host(host: str) -> bool:
    normalized = _normalize_host(host)
    return any(normalized == domain or normalized.endswith(f".{domain}") for domain in HF_DOMAINS)


def _start_server(host: str, port: int, *, allow_passthrough_hf: bool = False) -> RunningServer:
    manager = Manager()
    state = ProxyState(requests=manager.list())
    GuardProxyPlugin.configure(state=state, allow_passthrough_hf=allow_passthrough_hf)
    server = Proxy(
        input_args=[
            "--hostname",
            host,
            "--port",
            str(port),
            "--num-workers",
            "1",
            "--num-acceptors",
            "1",
            "--threaded",
            "--log-level",
            "warning",
        ],
        plugins=[GuardProxyPlugin],
    )
    server.__enter__()
    return RunningServer(
        server=server,
        host=str(server.flags.hostname),
        port=int(server.flags.port),
        state=state,
        allow_passthrough_hf=allow_passthrough_hf,
        manager=manager,
    )


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
    other_requests = summary.other_requests
    if hf_requests:
        sys.stderr.write("Hugging Face network requests observed:\n")
        for request in hf_requests:
            sys.stderr.write(f"- {request.method} {request.host} {request.target}\n")
    else:
        sys.stderr.write("No Hugging Face requests observed.\n")

    if other_requests:
        sys.stderr.write("Other proxied requests observed:\n")
        for request in other_requests:
            sys.stderr.write(f"- {request.method} {request.host} {request.target}\n")
    elif not summary.requests:
        sys.stderr.write("No proxied requests observed.\n")


def test_proxy_response_body_blocks_huggingface_request() -> None:
    request = ProxyRequest(
        method="CONNECT",
        target="huggingface.co:443",
        host="huggingface.co",
        is_huggingface=True,
    )

    rejection = _proxy_rejection(request)

    assert rejection.status_code == 502
    assert rejection.body == b"Blocked Hugging Face request to huggingface.co\n"


def test_proxy_plugin_blocks_case_insensitive_huggingface_host() -> None:
    from proxy.http.parser import httpParserTypes

    state = ProxyState()
    parser = HttpParser(httpParserTypes.REQUEST_PARSER)
    parser.parse(memoryview(b"CONNECT HuggingFace.co:443 HTTP/1.1\r\nHost: HuggingFace.co:443\r\n\r\n"))
    plugin = object.__new__(GuardProxyPlugin)

    GuardProxyPlugin.configure(state=state, allow_passthrough_hf=False)

    try:
        plugin.before_upstream_connection(parser)
    except HttpRequestRejected as rejection:
        assert rejection.status_code == 502
        assert rejection.body == b"Blocked Hugging Face request to huggingface.co\n"
    else:
        raise AssertionError("expected mixed-case Hugging Face request to be rejected")
    assert state.summary().huggingface_requests == [
        ProxyRequest(
            method="CONNECT",
            target="huggingface.co:443",
            host="huggingface.co",
            is_huggingface=True,
        )
    ]


def test_proxy_plugin_blocks_huggingface_by_default() -> None:
    from proxy.http.parser import httpParserTypes

    state = ProxyState()
    parser = HttpParser(httpParserTypes.REQUEST_PARSER)
    parser.parse(memoryview(b"CONNECT huggingface.co:443 HTTP/1.1\r\nHost: huggingface.co:443\r\n\r\n"))
    plugin = object.__new__(GuardProxyPlugin)

    GuardProxyPlugin.configure(state=state, allow_passthrough_hf=False)

    try:
        plugin.before_upstream_connection(parser)
    except HttpRequestRejected as rejection:
        assert rejection.status_code == 502
        assert rejection.body == b"Blocked Hugging Face request to huggingface.co\n"
    else:
        raise AssertionError("expected Hugging Face request to be rejected")
    assert state.summary().requests == [
        ProxyRequest(
            method="CONNECT",
            target="huggingface.co:443",
            host="huggingface.co",
            is_huggingface=True,
        )
    ]


def test_proxy_plugin_allows_huggingface_passthrough_when_enabled() -> None:
    from proxy.http.parser import httpParserTypes

    state = ProxyState()
    parser = HttpParser(httpParserTypes.REQUEST_PARSER)
    parser.parse(memoryview(b"CONNECT huggingface.co:443 HTTP/1.1\r\nHost: huggingface.co:443\r\n\r\n"))
    plugin = object.__new__(GuardProxyPlugin)

    GuardProxyPlugin.configure(state=state, allow_passthrough_hf=True)

    assert plugin.before_upstream_connection(parser) is parser
    assert state.summary().huggingface_requests == [
        ProxyRequest(
            method="CONNECT",
            target="huggingface.co:443",
            host="huggingface.co",
            is_huggingface=True,
        )
    ]


def test_proxy_plugin_allows_non_huggingface_passthrough_by_default() -> None:
    from proxy.http.parser import httpParserTypes

    state = ProxyState()
    parser = HttpParser(httpParserTypes.REQUEST_PARSER)
    parser.parse(memoryview(b"GET http://example.com/models HTTP/1.1\r\nHost: example.com\r\n\r\n"))
    plugin = object.__new__(GuardProxyPlugin)

    GuardProxyPlugin.configure(state=state, allow_passthrough_hf=False)

    assert plugin.before_upstream_connection(parser) is parser
    assert state.summary().other_requests == [
        ProxyRequest(
            method="GET",
            target="/models",
            host="example.com",
            is_huggingface=False,
        )
    ]


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
    assert summary.other_requests == [non_huggingface_request]


def test_run_wires_proxy_env_and_returns_child_status() -> None:
    state = ProxyState()
    state.record(
        ProxyRequest(
            method="GET",
            target="/models",
            host="example.com",
            is_huggingface=False,
        )
    )
    server = RunningServer(
        server=nullcontext(),
        host="127.0.0.1",
        port=43210,
        state=state,
        allow_passthrough_hf=False,
    )

    with (
        patch(f"{__name__}._start_server", return_value=server) as start_server,
        patch(f"{__name__}.subprocess.run", return_value=subprocess.CompletedProcess(["child"], 7)) as run_child,
        patch(f"{__name__}._write_run_summary") as write_summary,
    ):
        exit_code = run("child", host="127.0.0.1", port=0)

    assert exit_code == 7
    start_server.assert_called_once_with("127.0.0.1", 0, allow_passthrough_hf=False)
    assert run_child.call_args.kwargs["check"] is False
    assert run_child.call_args.kwargs["env"]["HTTP_PROXY"] == "http://127.0.0.1:43210"
    summary = write_summary.call_args.args[0]
    assert summary.request_count == 1
    assert summary.huggingface_request_count == 0


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
@click.option(
    "--allow-passthrough-hf/--no-allow-passthrough-hf",
    default=False,
    show_default=True,
    help="Forward Hugging Face requests after recording them.",
)
def serve(*, host: str, port: int, allow_passthrough_hf: bool) -> None:
    """Run the guard proxy until interrupted."""
    server = _start_server(host, port, allow_passthrough_hf=allow_passthrough_hf)
    _write_env_instructions(_proxy_env(server.proxy_url))
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()
        _write_run_summary(server.state.summary(), as_json=False)


def run(
    *command: str,
    host: str = DEFAULT_HOST,
    port: int = 0,
    json_output: bool = False,
    allow_passthrough_hf: bool = False,
) -> int:
    """Run a command through the guard proxy."""
    normalized = _normalize_command(command)
    if not normalized:
        sys.stderr.write("error: missing command after `run --`\n")
        return int(ExitCode.bad_input)

    server = _start_server(host, port, allow_passthrough_hf=allow_passthrough_hf)
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
    if summary.huggingface_requests and not allow_passthrough_hf:
        return int(ExitCode.failure)
    return completed.returncode


@cli.command("run", context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Host interface for the guard proxy.")
@click.option(
    "--port", default=0, show_default=True, type=int, help="Port for the guard proxy. Use 0 for any free port."
)
@click.option("--json", "json_output", is_flag=True, help="Write a machine-readable JSON summary to stdout.")
@click.option(
    "--allow-passthrough-hf/--no-allow-passthrough-hf",
    default=False,
    show_default=True,
    help="Forward Hugging Face requests after recording them.",
)
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def run_cli(
    command: tuple[str, ...],
    *,
    host: str,
    port: int,
    json_output: bool,
    allow_passthrough_hf: bool,
) -> None:
    """Run a command through the guard proxy."""
    raise click.exceptions.Exit(
        run(*command, host=host, port=port, json_output=json_output, allow_passthrough_hf=allow_passthrough_hf)
    )


def main() -> int:
    return int(cli(standalone_mode=False) or ExitCode.ok)


if __name__ == "__main__":
    raise SystemExit(main())
