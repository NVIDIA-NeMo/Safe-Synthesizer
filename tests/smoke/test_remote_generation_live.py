# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Opt-in live smoke test for the remote backend against build.nvidia.com.

Disabled by default because it makes real, billable network calls. Enable with::

    NSS_REMOTE_SMOKE_SEND=1 NVIDIA_API_KEY=nvapi-... \
        uv run --frozen pytest tests/smoke/test_remote_generation_live.py -vvs -n0

Overridable via env: ``NSS_REMOTE_SMOKE_ENDPOINT`` (default
``https://integrate.api.nvidia.com/v1``) and ``NSS_REMOTE_SMOKE_MODEL``. The
model must expose the OpenAI ``/v1/completions`` route (not all catalog models
do); the test skips rather than fails when the endpoint rejects that route.

This validates the live HTTP transport (auth header, request body, response
parsing, retry) -- not record quality, since an instruct model prompted on the
raw completions route is not expected to emit clean JSONL.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.errors import GenerationError
from nemo_safe_synthesizer.generation.remote_backend import RemoteBackend

from .conftest import build_remote_config, build_remote_metadata

LIVE_ENDPOINT = os.environ.get("NSS_REMOTE_SMOKE_ENDPOINT", "https://integrate.api.nvidia.com/v1")
LIVE_MODEL = os.environ.get("NSS_REMOTE_SMOKE_MODEL", "meta/llama-3.1-8b-instruct")
LIVE_API_KEY_ENV = "NVIDIA_API_KEY"

pytestmark = pytest.mark.skipif(
    os.environ.get("NSS_REMOTE_SMOKE_SEND") != "1",
    reason="opt-in live remote smoke; set NSS_REMOTE_SMOKE_SEND=1 (and NVIDIA_API_KEY) to run",
)

SCHEMA = {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}


def test_live_build_nvidia_completions(tmp_path: Path) -> None:
    if not os.environ.get(LIVE_API_KEY_ENV):
        pytest.skip(f"{LIVE_API_KEY_ENV} not set")

    schema_path = tmp_path / "schema.json"
    schema_path.write_text(json.dumps(SCHEMA))
    workdir = MagicMock(spec=Workdir)
    workdir.schema_file = schema_path

    config = build_remote_config(
        endpoint_url=LIVE_ENDPOINT,
        model=LIVE_MODEL,
        dialect="openai",  # NIM is a strict OpenAI server; drop the vLLM-only fields
        api_key_env=LIVE_API_KEY_ENV,
        num_records=5,
        max_concurrency=2,
        max_retries=2,
    )
    backend = RemoteBackend(config=config, model_metadata=build_remote_metadata(), workdir=workdir)
    try:
        backend.initialize()
        try:
            results = backend.generate()
        except GenerationError as exc:
            # A model that does not serve /v1/completions answers 400/404 on every
            # request; that is an availability gap, not a transport bug, so skip.
            if "400" in str(exc) or "404" in str(exc):
                pytest.skip(f"model {LIVE_MODEL!r} did not accept /v1/completions: {exc}")
            raise
    finally:
        backend.teardown()

    # Transport round-tripped and the batch loop ran. Record validity is not
    # asserted: quality from an instruct model on a raw completions prompt varies.
    assert results.num_prompts > 0
