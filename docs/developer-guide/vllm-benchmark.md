<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vLLM Benchmark Harness

The vLLM benchmark harness replays a captured generation trace against one or
more benchmark candidates. It is a developer tool for comparing engine and
sampling configurations. It is not part of the main Safe Synthesizer CLI
workflow.

Run it through `uv` with the full engine environment:

```bash
uv run --frozen --extra cu129 --extra engine --group dev \
    python tools/vllm_benchmark.py list
```

## Corpus Format

The input corpus is a JSONL file with one header record followed by prompt
records:

```json
{"kind": "header", "run_id": "run-1", "pretrained_model": "model-ref", "dataset_schema": {}, "engine_parameters": {}}
{"kind": "record", "row_index": 0, "prompt": "...", "sampling_params": {"temperature": 0.0}}
```

The header supplies the model, optional LoRA path, dataset schema, and captured
engine parameters. Each record supplies the exact prompt and sampling parameters
to replay.

## Run a Matrix

Use a preset matrix:

```bash
uv run --frozen --extra cu129 --extra engine --group dev \
    python tools/vllm_benchmark.py run \
    /path/to/trace.jsonl \
    --output /path/to/benchmark.json \
    --candidates default_matrix
```

Use `list` to see available presets. The `bracketed_ab_*` presets emit repeated
baseline and candidate runs for noisier comparisons.

Use a custom candidate file when a preset is too broad:

```json
{
  "candidates": [
    {
      "name": "baseline",
      "engine_config": {},
      "sampling_overrides": {"seed": 42}
    }
  ]
}
```

Then run:

```bash
uv run --frozen --extra cu129 --extra engine --group dev \
    python tools/vllm_benchmark.py run \
    /path/to/trace.jsonl \
    --output /path/to/benchmark.json \
    --candidates-file candidates.json
```

## Compare And Analyze

Render one benchmark JSON:

```bash
uv run --frozen --extra cu129 --extra engine --group dev \
    python tools/vllm_benchmark.py compare /path/to/benchmark.json
```

Analyze every `*.json` result in a directory:

```bash
uv run --frozen --extra cu129 --extra engine --group dev \
    python tools/vllm_benchmark.py analyze /path/to/results-dir \
    --cluster-signal auto \
    --json-out /path/to/analysis.json
```

The analyzer reports candidate-run aggregates by condition. It keeps the JSON
field name `n_cells` for compatibility; in this context a "cell" means one
candidate run in the benchmark matrix.

Use `--min-runs-per-condition` to raise or lower the refusal threshold. The
older `--min-cells-per-condition` spelling remains accepted for compatibility.

## WandB Sink

The harness can use WandB as a metrics sink. Each benchmark candidate run becomes
one WandB run in a shared group. WandB failures do not fail the benchmark; the
benchmark JSON is still written.

WandB mode defaults to disabled. Use `WANDB_MODE`, `NSS_WANDB_PROJECT`, or
`WANDB_PROJECT` consistently with the rest of Safe Synthesizer.
