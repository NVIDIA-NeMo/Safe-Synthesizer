# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke tests for the remote generation backend against a local stub server.

These run the full ``RemoteBackend`` path -- real ``httpx`` client, real
``ThreadPoolExecutor`` concurrency, the shared ``generate()`` batch loop, the
real ``TabularDataProcessor``, retry/backoff, and JSON compaction -- over a real
loopback socket. The server is an in-process stdlib HTTP server, so there is no
GPU, no model load, and no external network. The mocked unit suite in
``tests/generation/test_remote_backend.py`` covers fine-grained behavior; this
suite proves the pieces work together end-to-end over the wire.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.generation import remote_backend
from nemo_safe_synthesizer.generation.remote_backend import RemoteBackend

from .conftest import build_remote_config, build_remote_metadata

# Responder contract: (path, parsed_body) -> (status_code, json_payload, extra_headers | None)
Responder = Callable[[str, dict[str, Any]], tuple[int, dict[str, Any], dict[str, str] | None]]

SCHEMA = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}


@contextmanager
def stub_completions_server(responder: Responder) -> Iterator[str]:
    """Run a loopback HTTP server driven by ``responder``; yield its ``/v1`` base URL.

    Bound to an ephemeral port on 127.0.0.1 and served from a daemon thread, so
    concurrent requests from the backend's worker pool are handled in parallel.
    """

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 -- BaseHTTPRequestHandler API
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
            status, payload, headers = responder(self.path, body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            for key, value in (headers or {}).items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 -- match base signature
            pass  # silence per-request stderr noise

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        address = server.server_address
        yield f"http://{address[0]}:{address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _completion(text: str, *, completion_tokens: int = 6) -> dict[str, Any]:
    """A minimal OpenAI-style ``/completions`` response body wrapping ``text``."""
    return {
        "choices": [{"text": text, "finish_reason": "stop"}],
        "usage": {"completion_tokens": completion_tokens},
    }


def make_backend(config: Any, schema_path: Path) -> RemoteBackend:
    """Build a ``RemoteBackend`` reading a real on-disk schema (everything else real)."""
    workdir = MagicMock(spec=Workdir)
    workdir.schema_file = schema_path
    return RemoteBackend(config=config, model_metadata=build_remote_metadata(), workdir=workdir)


@pytest.fixture
def schema_path(tmp_path: Path) -> Path:
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(SCHEMA))
    return path


def test_happy_path_generates_records(schema_path: Path) -> None:
    """Full initialize -> generate loop yields the requested records over a real socket."""
    counter, lock = [0], threading.Lock()

    def responder(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], None]:
        assert path == "/v1/completions"
        assert body["prompt"], "prompt should be baked into the request body"
        assert body["n"] == 1
        with lock:
            counter[0] += 1
            n = counter[0]
        return 200, _completion(json.dumps({"name": f"person_{n}", "age": 20 + (n % 50)})), None

    with stub_completions_server(responder) as base_url:
        config = build_remote_config(endpoint_url=base_url, model="stub-model", num_records=5)
        backend = make_backend(config, schema_path)
        try:
            backend.initialize()
            results = backend.generate()
        finally:
            backend.teardown()
            backend.teardown()  # idempotent

    assert results.num_valid_records >= 5
    assert list(results.df.columns) == ["name", "age"]
    assert len(results.df) == 5  # truncated to num_records on completion


def test_recovers_from_transient_failures(schema_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """503 responses are retried with backoff until the server recovers."""
    monkeypatch.setattr(remote_backend, "_BACKOFF_BASE_SECONDS", 0.001)
    monkeypatch.setattr(remote_backend, "_BACKOFF_MAX_SECONDS", 0.01)
    counter, lock = [0], threading.Lock()
    fail_first = 2

    def responder(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, str] | None]:
        with lock:
            counter[0] += 1
            n = counter[0]
        if n <= fail_first:
            return 503, {"error": "server warming up"}, {"Retry-After": "0"}
        return 200, _completion(json.dumps({"name": f"person_{n}", "age": 20 + (n % 50)})), None

    with stub_completions_server(responder) as base_url:
        # Serialize requests so the transient failures land on the first record's attempts.
        config = build_remote_config(
            endpoint_url=base_url, model="stub-model", num_records=3, max_concurrency=1, max_retries=5
        )
        backend = make_backend(config, schema_path)
        try:
            backend.initialize()
            results = backend.generate()
        finally:
            backend.teardown()

    assert results.num_valid_records >= 3
    assert counter[0] > 3, "expected retries beyond the initial 503 responses"


def test_compacts_pretty_printed_json(schema_path: Path) -> None:
    """Multi-line JSON from a json_schema-constrained server is compacted before parsing.

    The line-oriented record extractor cannot match an object spanning newlines,
    so without ``_compact_json`` the run would yield zero records. ``openai``
    dialect can't send the vLLM ``disable_any_whitespace`` source fix, making the
    client-side compaction net the only thing that rescues the output.
    """
    counter, lock = [0], threading.Lock()

    def responder(path: str, body: dict[str, Any]) -> tuple[int, dict[str, Any], None]:
        with lock:
            counter[0] += 1
            n = counter[0]
        pretty = json.dumps({"name": f"person_{n}", "age": 20 + (n % 50)}, indent=2)
        assert "\n" in pretty
        return 200, _completion(pretty, completion_tokens=12), None

    with stub_completions_server(responder) as base_url:
        config = build_remote_config(
            endpoint_url=base_url,
            model="stub-model",
            num_records=3,
            dialect="openai",
            use_structured_generation=True,
            structured_generation_schema_method="json_schema",
        )
        backend = make_backend(config, schema_path)
        try:
            backend.initialize()
            results = backend.generate()
        finally:
            backend.teardown()

    assert results.num_valid_records >= 3
    assert len(results.df) == 3
