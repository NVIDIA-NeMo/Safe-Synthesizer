# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nearest neighbor search abstraction with GPU acceleration support.

This module provides a unified interface for nearest neighbor search that:
- Uses PyTorch (CUDA) when available for GPU-accelerated search
- Falls back to sklearn NearestNeighbors for CPU-only environments

Usage:
    from nemo_safe_synthesizer.evaluation.nearest_neighbors import NearestNeighborSearch

    # Create search index
    nn = NearestNeighborSearch(n_neighbors=5)
    nn.fit(data)  # numpy array of shape (n_samples, n_features)

    # Query
    distances, indices = nn.kneighbors(queries)  # input is numpy array of shape (n_queries, n_features)
"""

from __future__ import annotations

import logging
import os

# Must be set BEFORE importing numpy/sklearn (which load OpenBLAS)
# Cap threads to avoid OpenBLAS crashes on high-core machines (>128 cores)
# while still allowing some parallelism for large operations
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class NearestNeighborSearch:
    """Unified nearest neighbor search with GPU acceleration support.

    Uses PyTorch (CUDA) when available, falls back to sklearn otherwise.
    Both backends compute exact brute-force L2 distance for consistency.

    Attributes:
        n_neighbors: Number of neighbors to return in queries.
        use_gpu: Whether GPU acceleration is being used.
    """

    # Cache device detection across all instances
    _device_checked = False
    _torch_device: torch.device | None = None

    def __init__(self, n_neighbors: int = 5):
        """Initialize the nearest neighbor search.

        Args:
            n_neighbors: Number of neighbors to find in queries.
        """
        self.n_neighbors = n_neighbors
        self._torch_device = self._detect_device()
        self.use_gpu = self._torch_device is not None
        self._index = None
        self._data_t: torch.Tensor | None = None

    @classmethod
    def _detect_device(cls) -> torch.device | None:
        """Detect the best available torch device for NN search.

        Cached across all instances. Returns None if no GPU is available,
        indicating the sklearn CPU fallback should be used.

        Returns:
            torch.device for CUDA if available, None otherwise.
        """
        if cls._device_checked:
            return cls._torch_device

        if torch.cuda.is_available():
            cls._torch_device = torch.device("cuda")
            logger.debug("NearestNeighborSearch using CUDA")
        else:
            cls._torch_device = None
            logger.debug("NearestNeighborSearch using sklearn (CPU)")

        cls._device_checked = True
        return cls._torch_device

    def fit(self, data: np.ndarray) -> NearestNeighborSearch:
        """Build the search index from data.

        Args:
            data: Array of shape (n_samples, n_features) with float32 values.

        Returns:
            Self for method chaining.
        """
        data = np.ascontiguousarray(data, dtype=np.float32)

        if self.use_gpu:
            self._data_t = torch.from_numpy(data).to(self._torch_device)
        else:
            self._index = NearestNeighbors(
                n_neighbors=self.n_neighbors,
                algorithm="brute",
                metric="euclidean",
            )
            self._index.fit(data)

        return self

    def kneighbors(self, queries: np.ndarray, n_neighbors: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for query points.

        Args:
            queries: Array of shape (n_queries, n_features) with query points.
            n_neighbors: Number of neighbors to return. If None, uses self.n_neighbors.

        Returns:
            Tuple of (distances, indices) where:
            - distances: Array of shape (n_queries, k) with L2 distances
            - indices: Array of shape (n_queries, k) with indices into the fitted data
        """
        if self.use_gpu and self._data_t is None:
            raise RuntimeError("Must call fit() before kneighbors()")
        if not self.use_gpu and self._index is None:
            raise RuntimeError("Must call fit() before kneighbors()")

        k = n_neighbors if n_neighbors is not None else self.n_neighbors
        queries = np.ascontiguousarray(queries, dtype=np.float32)

        if queries.shape[0] == 0:
            empty = np.empty((0, k), dtype=np.float32)
            return empty, np.empty((0, k), dtype=np.int64)

        if self.use_gpu:
            assert self._data_t is not None
            queries_t = torch.from_numpy(queries).to(self._torch_device)
            # torch.cdist computes L2 (p=2) pairwise distances: (n_queries, n_samples)
            dists_t = torch.cdist(queries_t, self._data_t)
            topk_dists, topk_idx = torch.topk(dists_t, k, largest=False)
            distances = topk_dists.cpu().numpy()
            indices = topk_idx.cpu().numpy()
        else:
            assert self._index is not None
            distances, indices = self._index.kneighbors(queries, n_neighbors=k)

        return distances, indices

    @property
    def backend_name(self) -> str:
        """Return the name of the backend being used."""
        if self._torch_device is None:
            return "sklearn (CPU)"
        return f"torch ({self._torch_device.type.upper()})"
