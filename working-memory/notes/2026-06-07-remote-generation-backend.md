---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Remote generation backend — design, verification, and inference-serving roadmap
date: 2026-06-07
branch: agonzales/remote-generation-backend
status: in-review
kb:
  tags: [generation, backend, vllm, remote-inference, nim, triton, dynamo, tensorrt-llm, lora, structured-outputs]
  summary: >
    Added RemoteBackend (GPU-free generation against a vLLM OpenAI-compatible
    endpoint), lifted the batch loop into GeneratorBackend as a template method,
    and built tools/vllm_debug.py. Captures the structured-output field
    compatibility matrix across vLLM versions and a roadmap for targeting
    NIM / Triton / Dynamo / TensorRT-LLM.
---

# Remote generation backend

Branch `agonzales/remote-generation-backend`, 4 commits off `main`. Reviewed in-session
via an ad-hoc reviewer panel (the `/council` skill is user-invocation-only, so it was
emulated with parallel review agents). Not yet merged or pushed.

## What shipped

1. **`GeneratorBackend.generate()` is now a concrete template method**
   (`src/nemo_safe_synthesizer/generation/backend.py`). The batch loop, stopping
   conditions, and result aggregation moved up from `VllmBackend`. Subclasses implement
   only the engine seams: `initialize`, `prepare_params`, `_generate_batch`,
   `_get_prompt_token_count`, `teardown`. `VllmBackend` now inherits `generate()`;
   `TimeseriesBackend(VllmBackend)` keeps its own `generate()` override (its parallel-group
   loop is genuinely different — confirmed via astnav: it overrides `generate` at
   `timeseries_backend.py:893`).
2. **`RemoteBackend`** (`generation/remote_backend.py`) — GPU-free; POSTs to a vLLM
   OpenAI-compatible `/v1/completions`. One request per record, concurrent up to
   `max_concurrency` via a `ThreadPoolExecutor`; per-completion token counts come from the
   server `usage` block. Uses **httpx** (already a dep), not the `openai` SDK (transitive only).
3. **`RemoteParameters`** nested under `GenerateParameters` (`config/generate.py`):
   `endpoint_url`, `model`, `api_key_env`, `timeout_seconds`, `max_concurrency`. Selection
   wired in `SafeSynthesizer.generate()` (`sdk/library_builder.py`): remote wins, else
   time-series, else local vLLM. Remote + time-series is rejected at **config-validation
   time** via a `SafeSynthesizerParameters` model validator (`config/parameters.py`).
4. **`tools/vllm_debug.py`** — PEP 723 uv script (cyclopts/httpx/rich/structlog).
   Subcommands: `serve` (launch vLLM from the project venv, optional `--adapter` LoRA,
   A100 workarounds baked in, `--dry-run`), `call` (chat/text, thinking toggle,
   `--json-schema`/`--regex`/`--structural-tag` -> `structured_outputs`, `--json`),
   `models`. Distilled from `dev/evals/route-eval/{serve_vllm.sh,vllm_probe.py}`.

## Verified facts (with sources)

- **Structured-output request fields differ by vLLM version.** Tested two live servers:
  | field | old Nano server (localhost:8000) | new Super-120B server (gpu-dev-pod-serve-svc:8000) |
  | --- | --- | --- |
  | legacy `guided_json` / `guided_regex` (top-level) | honored | **silently ignored** |
  | `structured_outputs: {json\|regex\|structural_tag}` | honored | honored |
  | `response_format: {json_schema}` | honored | honored |
  The repo's offline `vllm_backend.py` already uses the modern `StructuredOutputsConfig`/
  `StructuredOutputsParams` API (vLLM 0.20+), so `RemoteBackend` uses **`structured_outputs`**
  — portable across both servers and an exact mirror of the offline params. `structural_tag`
  must be sent as its JSON-encoded **string** (parsed-object 400s).
- **`structural_tag` is NOT inherently unsupported over HTTP.** It was initially rejected on
  a stale assumption (the legacy `guided_*` family had no `guided_structural_tag`). Verified
  `structured_outputs: {structural_tag: <string>}` returns 200 and yields multi-record JSONL.
  All three methods are now supported, so the **`auto` default works out-of-the-box** for
  remote (auto -> structural_tag on xgrammar/auto backends).
- **httpx preserves the base path on join.** `base_url=".../v1"` + `"/completions"` ->
  `.../v1/completions` (NOT RFC `urljoin` behavior). No `/v1` bug.
- **Live end-to-end:** against Super-120B, default `auto` (structural_tag) produced 5 valid,
  realistic records via `TabularDataProcessor` in ~13s. `json_schema` works at transport level
  but the server **pretty-prints** JSON across lines, which the JSONL processor rejects (0 valid);
  `regex` and `structural_tag` enforce the compact single-line shape the pipeline needs.
- `tests/generation/test_remote_backend.py` — 25 unit tests (mocked httpx, GPU-free), incl. an
  end-to-end `generate()` loop through a real `TabularDataProcessor`. ruff + ty clean.

## Runtime validation against real adapters (2026-06-07, GPU freed mid-session)

Proved RemoteBackend is runtime-agnostic against two engine families, both serving a real
Safe Synthesizer LoRA, generation with **structured generation OFF** (the adapter emits JSONL):

- **vLLM + SmolLM3-3B `financial_transactions` LoRA** (served via `tools/vllm_debug.py serve
  --adapter`): 10/10 valid, realistic records (real names/cities, valid enum currencies/types,
  in-range IDs). The eager→CUDA-graphs perf result is below.
- **NIM (`mistral-7b-instruct-v0.3:1.12.0`) + Mistral-7B `ai_generated_essays` LoRA**: NIM
  discovered and served our adapter (`essays-lora` listed in `/v1/models`); 3/3 valid, coherent
  essay records. **The only code that needed to change was `prepare_params`** — the generate
  loop, processor, batch, and results machinery worked unchanged. Strongest evidence the
  abstraction is sound.

**Dialect finding (new compatibility-matrix axis).** NIM's OpenAI server accepts only a *subset*
of vLLM's sampling extensions; it 400s on the rest:

  | sampling field | vLLM | NIM (TRT-LLM) |
  | --- | --- | --- |
  | temperature, top_p, max_tokens, n | OK | OK |
  | min_p, ignore_eos | OK | OK |
  | repetition_penalty, top_k | OK | **400** |
  | skip_special_tokens, include_stop_str_in_output | OK | **400** |

Fix shipped: `RemoteParameters.dialect` (`"vllm"` default | `"openai"`). `prepare_params` emits
the vLLM extensions only for `"vllm"`; `"openai"` sends just the universal fields, for NIM/Triton.
Resolved sampling values are identical across dialects — only the wire fields differ.

NIM operational notes: needs `NGC_API_KEY` (env `NVIDIA_API_KEY` here) for `docker login nvcr.io`
+ runtime; mount a **writable** model cache (`chmod 777`, container runs as uid 1000) or it can't
download its profile; LoRA via `-e NIM_PEFT_SOURCE` with adapters in subdirs (subdir name = served
model), and rewrite the adapter_config `base_model_name_or_path` from a local snapshot path to the
canonical repo id. First boot ~10 min (profile download + TRT-LLM engine build).

## Quality & performance (what was and wasn't measured)

- **Quality**: only *face-validity* was observed — 100% schema-valid across all real-adapter runs
  (10/10, 10/10, 3/3), `status=complete`, realistic content. Safe Synthesizer's real evaluation
  phase (SQS, distribution fidelity, MI/AIA privacy) was **not** run, so there are no rigorous
  quality scores. NIM-vs-vLLM quality is **not** isolated: different adapters (SmolLM3-financial
  vs Mistral-essays), temp 0.9.
- **Performance**: the one clean comparison (same model/adapter/workload) is **vLLM eager vs
  CUDA-graphs: 118.73s → 15.24s for 10 records (~7.8x)** — the payoff of the FlashInfer repair
  (see below) + the `vllm_debug` eager-off default. NIM's number (404s / 68k tokens / 3 essays
  ≈ 169 tok/s aggregate) is **not** comparable: 7B vs 3B, ~22k-token essays vs ~600-token rows.
  A fair runtime comparison needs the *same* adapter on both engines + the eval metrics — not yet done.

## FlashInfer was broken in the venv (fixed, not a pyproject change)

Symptom: `import flashinfer` raised `AttributeError: module 'flashinfer_cubin' has no attribute
'__version__'`, crashing vLLM's backend probe (catches only `ImportError`). Root cause was **not**
versions: the `0.6.8.post1` PyPI wheel ships `flashinfer_cubin/__init__.py` (verified by remote
wheel listing) and the uv cache copy was intact, but the `.venv` install was **missing that file**
(only the 295MB `cubins/` dir linked) — a partial hardlink-fallback copy (`/stable-cache` cache and
`/root` venv are different filesystems). Fix: `uv sync --reinstall-package flashinfer-cubin` (no
pyproject edit; vLLM 0.20.0 hard-pins `==0.6.8.post1` anyway). Durable hardening: `UV_LINK_MODE=copy`
or colocate cache+venv. With FlashInfer importable, `vllm_debug serve` now defaults to CUDA graphs
(`--eager` is the opt-in escape hatch).

## Decisions and why

- **Template method on the base over a mixin.** Both were acceptable to the operator; base
  class is the cleaner end state and TimeseriesBackend's override means no behavioral risk.
- **httpx over openai SDK.** Avoids depending on an undeclared transitive dep; keeps the
  GPU-free backend lightweight.
- **`_get_prompt_token_count()` returns 0** (no local tokenizer). Server enforces its own
  context window; `generation_max_tokens_for` is sized from training-time example length.
  Open question below.
- **Config-time validation** for remote+time-series so it fails before training, not after.
  Note: pydantic wraps the `ParameterError` (a `ValueError`) into `ValidationError` — tests
  assert `ValidationError`.

## Meta development flow (worth reusing)

The session followed a feasibility -> evidence -> build -> adversarial-verify -> harden loop:
1. **Feasibility first**, no edits: parallel `Explore` agents mapped backend selection, the
   LoRA/metadata flow, and how tightly the batch loop coupled to vLLM. Conclusion: the
   `GeneratorBackend` seam + `remote: bool` placeholder made this additive.
2. **Hardened claims with astnav** (`callers`, `locate --hierarchy`, `refs`) before editing —
   this corrected one Explore finding (TimeseriesBackend extends VllmBackend, not the ABC) and
   confirmed `.remote` was written-once/never-read.
3. **Verified the load-bearing assumption** (the reused modules are vLLM/torch-free at import)
   before committing to a GPU-free backend.
4. **Live-tested against a real server** rather than trusting mocks — which is how the
   `guided_*` vs `structured_outputs` version split and the pretty-print/JSONL mismatch were
   found. Testing against a *second* server surfaced the compatibility bug a single server hid.
5. **Council-style review** (4 lenses: design / correctness / style / tests) -> triaged
   findings, applied the real ones (base docstring, response-parse hardening, config validator,
   added tests), declined over-engineering with reasons.
6. Operator constraint honored throughout: **never load a model on the local GPU** (a server
   already occupied it) — used `--dry-run`, mocked unit tests, and HTTP-only live calls.

## Ideas: adopt RemoteBackend for NIM / Dynamo / Triton testing

The whole point of the OpenAI-compatible target is that `RemoteBackend` is **server-agnostic**.
Concrete next steps to exploit that:

- **NIM (NVIDIA Inference Microservices).** OpenAI-compatible and supports dynamic multi-LoRA.
  `RemoteBackend` should connect unchanged: set `endpoint_url` to the NIM `/v1`, `model` to the
  served/LoRA name, `api_key_env` for the NGC/bearer token (NIM usually wants auth — our
  `Authorization: Bearer` path covers it). This is the cleanest "adapter already served" demo.
- **Triton** (vLLM backend or TensorRT-LLM backend). Triton's OpenAI-compatible frontend or its
  generate endpoint — confirm it speaks `/v1/completions` + `structured_outputs`; if it only
  does `response_format`, add a small adapter (see roadmap).
- **Dynamo** — same OpenAI surface, scale-out; useful for a throughput/concurrency stress of the
  `ThreadPoolExecutor` batching path.
- **A compatibility matrix is the deliverable**: run `tools/vllm_debug.py call --mode text
  --json-schema/--regex/--structural-tag` against each server and record which structured-output
  field each honors. We already have vLLM-old vs vLLM-new; NIM/Triton/Dynamo are the next rows.
  This matrix should drive whether `RemoteBackend` needs a pluggable structured-output dialect.

## Prove the design by adding a serving layer

To validate the abstraction beyond vLLM, add a thin serving target and point `RemoteBackend` at it:

- **Lowest effort:** stand up **NIM for LLMs** with a Safe-Synthesizer-trained LoRA adapter
  (NGC container, multi-LoRA), then run a normal `generate()` with `config.generation.remote`.
  If it works without code changes, the abstraction holds. `tools/vllm_debug.py serve --adapter`
  is the local rehearsal of this (vLLM `--enable-lora`).
- **Higher effort / more proof:** export the adapter+base to a **TensorRT-LLM** engine and serve
  via `trtllm-serve` or **Triton + tensorrtllm_backend**. This tests a genuinely different runtime
  (compiled engine, FP8/INT4) behind the same OpenAI API. Watch for: structured-output dialect
  differences (may be `response_format` only), and tokenization/`usage` reporting differences.
- **If a target only speaks `response_format` / `json_object`** (not `structured_outputs`), the
  clean extension is a small **structured-output dialect** seam on `RemoteBackend`
  (`prepare_params` chooses the field shape per a `remote.dialect` config: `vllm` |
  `openai_response_format`). Keep the resolved method (regex/json/structural_tag) the same; only
  the wire field changes. The verified matrix above is the input to this design.

## Open questions / TODO

- Should `_get_prompt_token_count()` optionally load the base tokenizer for an exact prompt-length
  clamp, or is "server enforces context" sufficient? (Currently returns 0.)
- `json_schema` pretty-print mismatch: document that remote users should prefer `regex`/`auto`,
  or post-process multi-line JSON before the JSONL processor. `auto` (-> structural_tag) sidesteps it.
- Is `RemoteParameters` better as its own config module vs nested under `GenerateParameters`?
  (Left nested; matches the time_series precedent.)
- No SDK-level test that `RemoteBackend` is selected over `VllmBackend` (only the config validator
  is tested); a heavier integration test could cover the selection branch.
- `tools/vllm_debug.py` has no unit tests (consistent with other agent-made tools in `tools/`);
  validated by `--dry-run` + live runs only.

## Pointers

- Commits (newest first): `vllm_debug` tool, support `structural_tag`, `structured_outputs` fix,
  feature + template-method refactor.
- Reference utilities this drew on: `/root/dev/agent-stuff/dev/evals/route-eval/serve_vllm.sh`
  (A100/FlashInfer serve workarounds) and `vllm_probe.py` (async OpenAI probe patterns).
- Live servers used this session: `localhost:8000` (Nemotron-3-Nano, later down) and
  `gpu-dev-pod-serve-svc:8000` (Nemotron-3-Super-120B, vLLM 0.20+).
