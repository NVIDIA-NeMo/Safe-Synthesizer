# Smoke Tests

Quick tests that verify training, generation, evaluation, and PII replacement code paths don't crash.
They use tiny or small models and run in seconds (CPU) or a few minutes (GPU).

```bash
make test-smoke             # CPU only, no GPU needed
make test-smoke-gpu          # GPU tests (requires CUDA)
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

GPU smoke tests use three marker-based isolation groups:

1. Train-only (`requires_gpu` without `vllm`/`smollm2`/`unsloth`): share a single process, auto-discovered via marker algebra.
2. vLLM generation (`vllm` marker): each file gets its own process because vLLM pre-allocates all GPU memory and never releases it.
3. SmolLM2 / Unsloth (`smollm2`, `unsloth` markers): each gets its own process, auto-discovered via markers.

When adding a new GPU smoke test, add the appropriate markers to `pytestmark`:

```python
pytestmark = [
    pytest.mark.requires_gpu,
    pytest.mark.vllm,  # if the test calls .generate() (uses vLLM)
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(sys.platform == "darwin", reason="Not applicable on macOS"),
]
```

If the new file uses vLLM, also add it to the explicit file list in the `test-smoke-gpu` Makefile target (vLLM files need per-file isolation).

## Things that will bite you

- LoRA rank must be 8 (not 4). vLLM silently rejects rank 4. Use `lora_r=8`.
- Iris only has 151 rows, but holdout needs >=200. Set `holdout=0, max_holdout=0` to skip it.
- Attention implementation: HuggingFaceBackend defaults to `flashinfer`, which HF doesn't recognize. The `_patch_attn_eager` fixture overrides it to `"sdpa"`.
- Stub tokenizer vocab is 32000. If you change the tiny model config, keep `vocab_size=32000` or you'll get shape mismatches.
- Always set `use_unsloth=False` unless you're specifically testing Unsloth. The `auto` default can pull it in and it monkey-patches transformers globally.
- CPU tests need `optim="adamw_torch"`. The production default (`paged_adamw_32bit`) requires bitsandbytes CUDA kernels.

## What's in `conftest.py`?

The shared fixtures cover both CPU and GPU smoke tests. Session-scoped fixtures are created once per pytest process; function-scoped fixtures are recreated per test.

Session-scoped (immutable / read-only):

- `base_smoke_config` -- default `SafeSynthesizerParameters` pointing at the local tiny model (Pydantic frozen model)
- `_patch_attn_eager` -- the attention implementation workaround mentioned above
- `stub_tokenizer`, `tiny_llama_config`, `local_tinyllama_dir` -- tokenizer and tiny model on disk
- `iris_df`, `timeseries_df` -- small DataFrames for training input

Function-scoped (fresh per test):

- `tiny_model` -- randomly initialized `LlamaForCausalLM` (mutated by training)

Helpers (plain functions, not fixtures):

- `train_with_sdk(config, data_df, save_path)` -- convenience wrapper around the SDK train flow
- `assert_adapter_saved(workdir)` -- checks that adapter files landed on disk

See [CONTRIBUTING.md](../../CONTRIBUTING.md#testing) for the full list of test commands.
