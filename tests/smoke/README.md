# Smoke Tests

Quick tests that verify training and generation code paths don't crash.
They use tiny or small models and run in seconds (CPU) or a few minutes (GPU).

```bash
make test-smoke             # CPU only, no GPU needed
make test-gpu-integration   # GPU tests (requires CUDA)
```

## When should I add a smoke test?

If you're adding a new training backend, generation backend, or model family,
add a smoke test for it. Same if you're changing how the SDK orchestrates
train/generate -- those paths are easy to break silently.

Smoke tests don't check output quality. They just make sure the code runs
end-to-end without throwing. Use the smallest model that exercises the path
(the local `tiny_llama` stub for most things, SmolLM2-135M when you need
a real tokenizer/model).

## Things that will bite you

- **LoRA rank must be 8** (not 4). vLLM silently rejects rank 4. Use `lora_r=8`.
- **Iris only has 151 rows**, but holdout needs >=200. Set `holdout=0, max_holdout=0` to skip it.
- **Attention implementation**: HuggingFaceBackend defaults to `flashinfer`, which HF doesn't recognize. The `_patch_attn_eager` fixture overrides it to `"sdpa"`.
- **Stub tokenizer vocab is 32000**. If you change the tiny model config, keep `vocab_size=32000` or you'll get shape mismatches.
- **Always set `use_unsloth=False`** unless you're specifically testing Unsloth. The `auto` default can pull it in and it monkey-patches transformers globally.
- **CPU tests need `optim="adamw_torch"`**. The production default (`paged_adamw_32bit`) requires bitsandbytes CUDA kernels.
- **Unsloth tests run in a separate process**. Unsloth patches transformers at import time, which breaks Opacus/DP if they share a process. The Makefile handles this automatically.

## What's in `conftest.py`?

The shared fixtures cover both CPU and GPU smoke tests. The most important ones:

- `base_smoke_config` -- default `SafeSynthesizerParameters` pointing at the local tiny model
- `train_with_sdk(config, data_df, save_path)` -- convenience wrapper around the SDK train flow
- `assert_adapter_saved(workdir)` -- checks that adapter files landed on disk
- `_patch_attn_eager` -- the attention implementation workaround mentioned above
- `tiny_model`, `stub_tokenizer`, `tiny_training_dataset` -- CPU test building blocks
- `local_tinyllama_dir` -- saves the tiny model to a temp dir so GPU tests don't need internet
- `iris_df`, `timeseries_df` -- small DataFrames for training input

See [CONTRIBUTING.md](../../CONTRIBUTING.md#testing) for the full list of test commands.
