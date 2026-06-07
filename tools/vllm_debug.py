#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "cyclopts>=3",
#     "httpx>=0.27",
#     "pydantic>=2",
#     "rich>=13",
#     "structlog>=24",
# ]
# ///
r"""vllm-debug: spin up a vLLM model and fire a few debug calls against it.

A standalone companion to the VllmBackend / RemoteBackend code: serve a base
model (optionally with a trained LoRA adapter attached) and poke it over the
OpenAI-compatible API to see exactly what it emits. ``serve`` runs vLLM from the
project venv with the workarounds this A100 box needs (offline HF cache,
FLASH_ATTN, eager mode); ``call`` and ``models`` are lightweight and need only
this script's own deps.

Usage::

    # Serve the local Nemotron text model on :8000 (Ctrl-C to stop)
    uv run tools/vllm_debug.py serve

    # Serve a base model + a trained LoRA adapter, registered as model `lora`
    uv run tools/vllm_debug.py serve meta-llama/Llama-3.2-1B \
        --adapter artifacts/.../train/adapter --max-lora-rank 32

    # Print the launch command without starting it (no GPU touched)
    uv run tools/vllm_debug.py serve --dry-run

    # One chat call against a running server, reasoning disabled
    uv run tools/vllm_debug.py call "Say hello" --base-url http://localhost:8000/v1

    # Debug structured generation: constrain a text completion to a JSON schema
    uv run tools/vllm_debug.py call "a person" --mode text --json-schema schema.json

    # Constrain to a regex, repeat 3 times, emit machine-readable output
    uv run tools/vllm_debug.py call "a person" --mode text --regex '\{.*\}' -n 3 --json

    # Health check / list served models
    uv run tools/vllm_debug.py models --base-url http://localhost:8000/v1

Exit codes: 0 success, 1 a call failed, 125 bad input.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, Self

import cyclopts
import httpx
import structlog
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
DEFAULT_BASE_URL = "http://localhost:8000/v1"
REPO_ROOT = Path(__file__).resolve().parents[1]

LogFormat = Literal["plain", "json"]
CallMode = Literal["chat", "text"]

console = Console()
log = structlog.get_logger()


def configure_logging(fmt: LogFormat) -> None:
    """Set up structlog to render to stderr in the chosen format."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer() if fmt == "plain" else structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
    )


def bad_input(message: str) -> NoReturn:
    """Log a usage error and exit with code 125."""
    log.error("bad-input", detail=message)
    raise SystemExit(125)


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


class ServeConfig(BaseModel):
    """Resolved arguments for launching a vLLM OpenAI-compatible server."""

    model: str
    port: int
    host: str
    max_model_len: int
    gpu_util: float
    max_num_seqs: int
    attention_backend: str
    enforce_eager: bool
    prefix_caching: bool
    trust_remote_code: bool
    adapter: Path | None
    lora_name: str
    max_lora_rank: int
    python: Path

    def argv(self) -> list[str]:
        """Build the vLLM api_server command line."""
        cmd = [
            str(self.python), "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--served-model-name", self.model,
            "--port", str(self.port),
            "--host", self.host,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_util),
            "--max-num-seqs", str(self.max_num_seqs),
            "--attention-config", json.dumps({"backend": self.attention_backend}),
        ]  # fmt: skip
        if self.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.enforce_eager:
            cmd.append("--enforce-eager")
        if self.prefix_caching:
            cmd.append("--enable-prefix-caching")
        if self.adapter is not None:
            cmd += ["--enable-lora", "--lora-modules", f"{self.lora_name}={self.adapter}"]
            cmd += ["--max-lora-rank", str(self.max_lora_rank)]
        cmd.append("--no-enable-log-requests")
        return cmd

    def server_env(self) -> dict[str, str]:
        """Environment for the server: force the offline HF cache, keep FP8 off the Triton path."""
        return os.environ | {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_TEST_FORCE_FP8_MARLIN": "1",
        }


def _default_python() -> Path:
    """Path to the project venv's interpreter (where vLLM is installed)."""
    return REPO_ROOT / ".venv" / "bin" / "python"


app = cyclopts.App(name="vllm-debug", help="Serve a vLLM model and fire debug calls against it.")


@app.command
def serve(
    model: str = DEFAULT_MODEL,
    *,
    port: int = 8000,
    host: str = "0.0.0.0",
    adapter: Annotated[Path | None, cyclopts.Parameter(help="LoRA adapter dir to attach and serve")] = None,
    lora_name: Annotated[str, cyclopts.Parameter(help="Model name to register the adapter under")] = "lora",
    max_lora_rank: int = 32,
    max_model_len: int = 8192,
    gpu_util: Annotated[float, cyclopts.Parameter(name="--gpu-util")] = 0.90,
    max_num_seqs: int = 128,
    attention_backend: str = "FLASH_ATTN",
    eager: Annotated[bool, cyclopts.Parameter(help="Use --enforce-eager (FlashInfer is broken in this venv)")] = True,
    prefix_caching: bool = True,
    trust_remote_code: bool = True,
    python: Annotated[Path | None, cyclopts.Parameter(help="Interpreter to run vLLM (default: project venv)")] = None,
    dry_run: Annotated[bool, cyclopts.Parameter(name="--dry-run", help="Print the command and exit")] = False,
) -> None:
    """Launch a vLLM OpenAI-compatible server, optionally with a LoRA adapter attached.

    Replaces this process with the server (so Ctrl-C / logs behave normally).
    Defaults bake in the workarounds this A100 box needs: a small context window,
    high GPU fraction, FLASH_ATTN, and eager mode.
    """
    if adapter is not None and not adapter.exists():
        bad_input(f"adapter path does not exist: {adapter}")
    interpreter = python or _default_python()
    cfg = ServeConfig(
        model=model, port=port, host=host, max_model_len=max_model_len, gpu_util=gpu_util,
        max_num_seqs=max_num_seqs, attention_backend=attention_backend, enforce_eager=eager,
        prefix_caching=prefix_caching, trust_remote_code=trust_remote_code, adapter=adapter,
        lora_name=lora_name, max_lora_rank=max_lora_rank, python=interpreter,
    )  # fmt: skip
    argv, env = cfg.argv(), cfg.server_env()
    served_as = lora_name if adapter is not None else model

    if dry_run:
        console.print("[bold]HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_TEST_FORCE_FP8_MARLIN=1[/bold]")
        console.print(" ".join(argv))
        console.print(f"[dim]→ call it with  --base-url http://localhost:{port}/v1  --model {served_as}[/dim]")
        return

    if not interpreter.exists():
        bad_input(f"interpreter not found: {interpreter} (pass --python or run `uv sync`)")
    log.info("serving", model=model, port=port, adapter=str(adapter) if adapter else None, served_as=served_as)
    os.execve(argv[0], argv, env)  # replace this process with the server


# --------------------------------------------------------------------------- #
# call
# --------------------------------------------------------------------------- #


class TokenUsage(BaseModel):
    """Token counts reported by the server's ``usage`` block."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_openai(cls, usage: dict[str, Any]) -> Self:
        def _i(value: Any) -> int:
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return 0

        prompt, completion = _i(usage.get("prompt_tokens")), _i(usage.get("completion_tokens"))
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=_i(usage.get("total_tokens")) or (prompt + completion),
        )


class CallResult(BaseModel):
    """Outcome of a single debug call."""

    content: str = ""
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    latency_s: float = 0.0
    error: str | None = None


def build_structured_outputs(json_schema: Path | None, regex: str | None, structural_tag: Path | None) -> dict | None:
    """Build the vLLM ``structured_outputs`` field from at most one constraint flag."""
    chosen: list[tuple[str, Any]] = []
    if json_schema is not None:
        chosen.append(("json", json.loads(json_schema.read_text())))
    if regex is not None:
        chosen.append(("regex", regex))
    if structural_tag is not None:
        chosen.append(("structural_tag", structural_tag.read_text()))
    if not chosen:
        return None
    if len(chosen) > 1:
        bad_input("pass at most one of --json-schema / --regex / --structural-tag")
    key, value = chosen[0]
    return {key: value}


def build_call_payload(
    prompt: str,
    *,
    mode: CallMode,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    think: bool,
    structured_outputs: dict | None,
) -> dict[str, Any]:
    """Assemble the request body for chat or text completion."""
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty,
    }
    if mode == "chat":
        payload["messages"] = [{"role": "user", "content": prompt}]
        if not think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        payload["prompt"] = prompt
    if structured_outputs is not None:
        payload["structured_outputs"] = structured_outputs
    return payload


def parse_call_response(mode: CallMode, data: dict[str, Any]) -> tuple[str, str | None, str | None]:
    """Extract ``(content, reasoning, finish_reason)`` from a completion response."""
    choice = data["choices"][0]
    if mode == "chat":
        message = choice.get("message") or {}
        return message.get("content") or "", message.get("reasoning_content"), choice.get("finish_reason")
    return choice.get("text") or "", None, choice.get("finish_reason")


def do_call(client: httpx.Client, payload: dict[str, Any], *, mode: CallMode, api_key: str) -> CallResult:
    """Issue one completion request and capture text, usage, and latency."""
    path = "/chat/completions" if mode == "chat" else "/completions"
    started = time.monotonic()
    try:
        response = client.post(path, json=payload, headers={"Authorization": f"Bearer {api_key}"})
        response.raise_for_status()
        data = response.json()
        content, reasoning, finish = parse_call_response(mode, data)
        return CallResult(
            content=content,
            reasoning=reasoning,
            finish_reason=finish,
            usage=TokenUsage.from_openai(data.get("usage") or {}),
            latency_s=round(time.monotonic() - started, 3),
        )
    except Exception as exc:  # noqa: BLE001 -- debug tool: report any failure as a result
        return CallResult(latency_s=round(time.monotonic() - started, 3), error=f"{type(exc).__name__}: {exc}")


def render_calls(results: list[CallResult], *, json_output: bool) -> None:
    """Print call results as rich panels, or as a JSON array with --json."""
    if json_output:
        print(json.dumps([r.model_dump() for r in results], indent=2))
        return
    table = Table(title="vllm-debug call", show_lines=True)
    for col in ("#", "finish", "prompt tok", "completion tok", "latency s", "content"):
        table.add_column(col, overflow="fold")
    for i, r in enumerate(results, 1):
        body = r.error and f"[red]{r.error}[/red]" or r.content
        if r.reasoning:
            body = f"[dim](reasoning: {len(r.reasoning)} chars hidden)[/dim]\n{body}"
        table.add_row(
            str(i), r.finish_reason or "-", str(r.usage.prompt_tokens),
            str(r.usage.completion_tokens), f"{r.latency_s:.2f}", body,
        )  # fmt: skip
    console.print(table)


@app.command
def call(
    prompt: Annotated[str, cyclopts.Parameter(help="Prompt text, or '-' to read stdin")],
    *,
    base_url: Annotated[str, cyclopts.Parameter(name="--base-url")] = DEFAULT_BASE_URL,
    model: str = DEFAULT_MODEL,
    mode: CallMode = "chat",
    max_tokens: Annotated[int, cyclopts.Parameter(name="--max-tokens")] = 256,
    temperature: float = 0.0,
    top_p: Annotated[float, cyclopts.Parameter(name="--top-p")] = 1.0,
    repetition_penalty: Annotated[float, cyclopts.Parameter(name="--repetition-penalty")] = 1.0,
    think: Annotated[bool, cyclopts.Parameter(help="Allow reasoning CoT (chat mode; default off)")] = False,
    json_schema: Annotated[Path | None, cyclopts.Parameter(name="--json-schema", help="Constrain to JSON schema")] = None,
    regex: Annotated[str | None, cyclopts.Parameter(help="Constrain output to a regex")] = None,
    structural_tag: Annotated[Path | None, cyclopts.Parameter(name="--structural-tag", help="XGrammar tag file")] = None,
    n: Annotated[int, cyclopts.Parameter(help="Number of times to repeat the call")] = 1,
    api_key: Annotated[str, cyclopts.Parameter(name="--api-key")] = "EMPTY",
    request_timeout: Annotated[float, cyclopts.Parameter(name="--request-timeout")] = 600.0,
    json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
) -> None:
    """Send one or more debug calls to a running server and show text + token usage.

    Structured generation maps to vLLM's ``structured_outputs`` field; pass at
    most one of --json-schema / --regex / --structural-tag.
    """
    if n < 1:
        bad_input("-n must be >= 1")
    text = sys.stdin.read() if prompt == "-" else prompt
    structured = build_structured_outputs(json_schema, regex, structural_tag)
    payload = build_call_payload(
        text, mode=mode, model=model, max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, repetition_penalty=repetition_penalty, think=think, structured_outputs=structured,
    )  # fmt: skip

    with httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(request_timeout)) as client:
        results = [do_call(client, payload, mode=mode, api_key=api_key) for _ in range(n)]
    render_calls(results, json_output=json_output)
    if any(r.error for r in results):
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# models
# --------------------------------------------------------------------------- #


@app.command
def models(
    *,
    base_url: Annotated[str, cyclopts.Parameter(name="--base-url")] = DEFAULT_BASE_URL,
    api_key: Annotated[str, cyclopts.Parameter(name="--api-key")] = "EMPTY",
    json_output: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
) -> None:
    """List the models a running server is serving (also a health check)."""
    try:
        with httpx.Client(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(30.0)) as client:
            response = client.get("/models", headers={"Authorization": f"Bearer {api_key}"})
            response.raise_for_status()
            entries = response.json().get("data", [])
    except Exception as exc:  # noqa: BLE001
        log.error("models-failed", base_url=base_url, detail=f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    if json_output:
        print(json.dumps(entries, indent=2))
        return
    table = Table(title=f"served models @ {base_url}")
    table.add_column("id")
    table.add_column("max_model_len")
    table.add_column("root", overflow="fold")
    for entry in entries:
        table.add_row(entry.get("id", "?"), str(entry.get("max_model_len", "?")), entry.get("root", ""))
    console.print(table)


@app.meta.default
def _launcher(
    *tokens: Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    log_format: Annotated[LogFormat, cyclopts.Parameter(name="--log-format")] = "plain",
) -> None:
    """Configure logging, then dispatch to the requested subcommand."""
    configure_logging(log_format)
    app(tokens)


if __name__ == "__main__":
    app.meta()
