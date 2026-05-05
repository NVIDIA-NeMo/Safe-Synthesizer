# Smoke Tests

Quick tests that verify training, generation, evaluation, and PII replacement code paths don't crash.
They use tiny or small models and run in seconds (CPU) or a few minutes (GPU).

```bash
make test-smoke             # CPU only, no GPU needed
make test-smoke-gpu          # All staged GPU smoke tests (requires CUDA)
make test-smoke-gpu-train-only
make test-smoke-gpu-generation
make test-smoke-gpu-resume
make test-smoke-gpu-structured-generation
make test-smoke-gpu-timeseries
make test-smoke-gpu-smollm2
```

## When should I add a smoke test?

If you're adding a new training backend, generation backend, evaluation
component, or model family, add a smoke test for it. Same if you're changing
how the SDK orchestrates train/generate/evaluate -- those paths are easy to
break silently.

Smoke tests don't check output quality. They just make sure the code runs
end-to-end without throwing. Use the smallest model that exercises the path
(the local `tiny_llama` stub for most things, SmolLM2-135M when you need
a real tokenizer/model).

## GPU Test Process Isolation

GPU smoke tests use staged Make targets for process isolation and CI visibility:

1. `test-smoke-gpu-train-only`: `requires_gpu` without `vllm`/`smollm2`, auto-discovered via marker algebra.
2. `test-smoke-gpu-generation`: generation vLLM tests.
3. `test-smoke-gpu-resume`: resume + generation vLLM tests.
4. `test-smoke-gpu-structured-generation`: structured generation vLLM tests.
5. `test-smoke-gpu-timeseries`: timeseries generation vLLM tests.
6. `test-smoke-gpu-smollm2`: SmolLM2 Hub download tests, auto-discovered via markers.

`make test-smoke-gpu` runs all GPU smoke stages in order. vLLM stages are
split by file because vLLM pre-allocates all GPU memory and never releases it
within a process. CI runs the same stage targets as separate workflow steps so
the failing smoke lane is visible in the GitHub Actions UI.

When adding a new GPU smoke test, add the appropriate markers to `pytestmark`:

```python
pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.vllm,  # if the test calls .generate() (uses vLLM)
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]
```

If the new file uses vLLM, add a dedicated `test-smoke-gpu-*` Makefile target and include it in `test-smoke-gpu` so CI shows it as its own stage.

## Things that will bite you

- LoRA rank must be 8 (not 4). vLLM silently rejects rank 4. Use `lora_r=8`.
- Preflight requires at least 200 training rows. GPU tests that train should use
  `fixture_gpu_smoke_df` or `fixture_preflight_timeseries_df`.
- Attention implementation: HuggingFaceBackend defaults to `flashinfer`, which HF doesn't recognize. The `_patch_attn_eager` fixture overrides it to `"sdpa"`.
- Stub tokenizer vocab is 32000. If you change the tiny model config, keep `vocab_size=32000` or you'll get shape mismatches.
- CPU tests need `optim="adamw_torch"`. The production default (`paged_adamw_32bit`) requires bitsandbytes CUDA kernels.

## What's in `conftest.py`?

The shared fixtures cover both CPU and GPU smoke tests. Session-scoped fixtures are created once per pytest process; function-scoped fixtures are recreated per test.

Session-scoped (immutable / read-only):

- `fixture_base_smoke_config` -- default `SafeSynthesizerParameters` pointing at the local tiny model (Pydantic frozen model)
- `_patch_attn_eager` -- the attention implementation workaround mentioned above
- `fixture_stub_tokenizer`, `fixture_tiny_llama_config`, `fixture_local_tinyllama_dir` -- tokenizer and tiny model on disk
- `fixture_iris_df`, `fixture_timeseries_df` -- small DataFrames for CPU smoke paths
- `fixture_gpu_smoke_df`, `fixture_preflight_timeseries_df` -- larger DataFrames for GPU paths
  that run preflight

Function-scoped (fresh per test):

- `fixture_tiny_model` -- randomly initialized `LlamaForCausalLM` (mutated by training)

Helpers (plain functions, not fixtures):

- `train_with_sdk(config, data_df, save_path)` -- convenience wrapper around the SDK train flow
- `assert_adapter_saved(workdir)` -- checks that adapter files landed on disk

See [CONTRIBUTING.md](../../CONTRIBUTING.md#testing) for the full list of test commands.
