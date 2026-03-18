# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Manually run pii replacer as a basic test
import pytest

# Skip all tests in this module if torch is not available
pytest.importorskip("torch", reason="torch is required for these tests (install with: uv sync --extra cpu)")

import pandas as pd

from nemo_safe_synthesizer.pii_replacer.nemo_pii import NemoPII

# Currently use env variables to configure various pieces
# Inferenc endpoint for classify:
# export NVIDIA_API_KEY=<...>
# export NIM_ENDPOINT_URL=https://integrate.api.nvidia.com/v1
# export NIM_MODEL_ID=qwen/qwen2.5-coder-32b-instruct


def main():
    nemo_pii = NemoPII()

    df = pd.read_csv(
        "https://raw.githubusercontent.com/gretelai/gretel-blueprints/refs/heads/main/sample_data/financial_transactions.csv"
    )
    df = df.head(100)

    print("Original df:")
    print(df.to_csv(index=False))

    column_classifications = nemo_pii.classify_df(df.head(10))
    print(column_classifications)
    result = nemo_pii.transform_df(df)

    print("Result df:")
    print(result.transformed_df.to_csv(index=False))
    print("Column statistics:")
    print(result.column_statistics)


if __name__ == "__main__":
    main()
