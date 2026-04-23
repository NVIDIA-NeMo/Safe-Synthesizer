# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier-3 A/B comparison: JSON baseline vs positional serialization.

Runs the full SmolLM3-3B pipeline end-to-end twice on the same dataset and
prints a side-by-side summary of valid-record yield, wall-clock, and
generation stats.

Not a pytest; intended for manual validation on a real GPU.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
from datasets import load_dataset

from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

os.environ.setdefault("NSS_INFERENCE_KEY", "")


def _run_one(df: pd.DataFrame, serialization_format: str, save_path: str) -> dict:
    t0 = time.monotonic()
    builder = (
        SafeSynthesizer(save_path=save_path)
        .with_data_source(df)
        .with_data(serialization_format=serialization_format)
        .with_replace_pii(enable=False)
    )
    builder.run()
    elapsed = time.monotonic() - t0

    gen = builder.generator.gen_results
    return {
        "serialization_format": serialization_format,
        "elapsed_sec": round(elapsed, 1),
        "num_valid_records": gen.num_valid_records,
        "num_invalid_records": gen.num_invalid_records,
        "valid_record_fraction": round(gen.valid_record_fraction, 4),
        "num_prompts": gen.num_prompts,
        "tokens_per_second": gen.tokens_per_second,
        "valid_tokens_per_second": gen.valid_tokens_per_second,
    }


def main() -> None:
    dataset = load_dataset("clinc/clinc_oos", "small")
    df = dataset["train"].to_pandas()

    results = []
    for fmt in ("json", "positional"):
        save_path = f"./tier3-artifacts-{fmt}"
        print(f"\n========== Running {fmt!r} ==========")
        r = _run_one(df, fmt, save_path)
        results.append(r)
        print(f"Result: {r}")

    print("\n========== Summary ==========")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
