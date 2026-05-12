# ___PROJECT___ 0.0.0 (DD Mon YYYY)

## New Features

- Add `training.quantization_scheme` for first-class selection of v5
  quantization backends: `bnb-4bit`, `bnb-8bit`, `fp8`, `nvfp4`, `mxfp4`.
  The legacy `training.quantization_bits` field continues to work and maps
  to the matching `bnb-*` scheme. See
  [Quantization schemes](docs/user-guide/configuration.md#quantization-schemes).
- Centralize tokenizer loads through `load_fast_tokenizer()` which surfaces
  a warning when transformers v5 falls back to the slow (Python) backend
  for models without a Rust tokenizer.

## Improvements

- Upgrade `transformers` to v5 (`>=5.0,<6`). Bumps `accelerate>=1.1.0`,
  `peft>=0.18.0`, `bitsandbytes>=0.46.1`, `huggingface-hub>=1.3,<2` to meet
  v5 floors. Override `transformers<5` from `vllm 0.20.0` metadata until
  vLLM publishes a v5-aware wheel.
- Remove the v4.48-era `try/except TypeError` shim in
  `OpacusDPTrainer.training_step` — `num_items_in_batch` is part of the
  stable v5 `Trainer.compute_loss` signature.

## Bug Fixes

- ...
