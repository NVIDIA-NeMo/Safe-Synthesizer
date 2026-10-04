# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import torch

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from ...artifacts.analyzers.field_features import describe_field
from ...data_processing.dataset_profile import DatasetProfile


def find_text_fields(df: pd.DataFrame, dataset_profile: DatasetProfile | None = None) -> list[str]:
    """Identify columns in ``df`` whose content is free-form text.

    Each column is passed through ``describe_field``; those classified
    as ``"text"`` are returned.

    Args:
        df: DataFrame whose columns are inspected.
        dataset_profile: Training-time profile to use instead of re-inference.

    Returns:
        Column names classified as free-form text.
    """
    if dataset_profile is not None:
        return [column for column in dataset_profile.text_columns() if column in df.columns]
    text_fields: list[str] = []
    for col in df.columns:
        field_info = describe_field(col, df[col])
        if field_info.type.value == "text":
            text_fields.append(col)
    return text_fields


def divide_tabular_text(df: pd.DataFrame, text_fields: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into a tabular-only and a text-only DataFrame.

    Columns present in ``text_fields`` go into the text DataFrame; the
    remaining columns go into the tabular DataFrame.

    Args:
        df: Source DataFrame to split.
        text_fields: Column names to treat as text.

    Returns:
        A ``(tabular_df, text_df)`` tuple where ``tabular_df`` contains only
        the non-text columns and ``text_df`` contains only the text columns.
    """
    tabular_fields = [col for col in df.columns if col not in text_fields]
    return df.filter(tabular_fields), df.filter(text_fields)


def embed_text(df: pd.DataFrame, embedder: SentenceTransformer) -> pd.DataFrame:
    """Embed every text column in ``df`` and return a single averaged embedding per row.

    For each column the ``embedder`` produces a ``(n_rows, embed_dim)`` matrix.
    The per-column matrices are stacked and averaged across columns so that
    every column contributes equally to the final embedding.

    Args:
        df: DataFrame whose columns are all text to be embedded.
        embedder: Sentence-transformer model used to produce embeddings.

    Returns:
        Single-column DataFrame with column ``"embedding"`` whose values are
        1-D tensors of shape ``(embed_dim,)``.
    """
    embeddings = {}
    for col in df.columns:
        data = [str(r) for r in df[col].to_list()]
        embeddings[col] = torch.as_tensor(embedder.encode(data, show_progress_bar=False, convert_to_tensor=True))

    stacked = torch.stack([embeddings[col] for col in df.columns], dim=0)  # shape: (n_cols, n_rows, embed_dim)
    avg_embeddings = torch.mean(stacked, dim=0)  # shape: (n_rows, embed_dim)

    return pd.DataFrame({"embedding": list(avg_embeddings)})
