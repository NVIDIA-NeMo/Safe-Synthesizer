# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nearest neighbor search abstraction with GPU acceleration support.

This module provides a unified interface for nearest neighbor search that:
- Uses cuVS (NVIDIA RAPIDS) when available for GPU-accelerated search
- Falls back to sklearn NearestNeighbors for CPU-only environments

Usage:
    from nemo_safe_synthesizer.evaluation.nearest_neighbors import NearestNeighborSearch

    # Create search index
    nn = NearestNeighborSearch(n_neighbors=5)
    nn.fit(data)  # numpy array of shape (n_samples, n_features)

    # Query
    distances, indices = nn.kneighbors(queries)  # numpy array of shape (n_queries, n_features)
"""

from __future__ import annotations

import os

# Must be set BEFORE importing numpy/sklearn (which load OpenBLAS)
# Cap threads to avoid OpenBLAS crashes on high-core machines (>128 cores)
# while still allowing some parallelism for large operations
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np
from nemo_safe_synthesizer.evaluation._cuda_preload import ensure_nvidia_libraries_preloaded
from sklearn.neighbors import NearestNeighbors


class NearestNeighborSearch:
    """Unified nearest neighbor search with GPU acceleration support.

    Uses cuVS (NVIDIA RAPIDS) when available, falls back to sklearn otherwise.
    Both backends use exact brute-force L2 distance for consistency.

    Attributes:
        n_neighbors: Number of neighbors to return in queries.
        use_gpu: Whether GPU acceleration is being used.
    """

    # Cache GPU availability check across all instances
    _gpu_checked = False
    _gpu_available = False
    _cuvs_brute_force = None
    _cp = None

    def __init__(self, n_neighbors: int = 5):
        """Initialize the nearest neighbor search.

        Args:
            n_neighbors: Number of neighbors to find in queries.
        """
        self.n_neighbors = n_neighbors
        self.use_gpu = self._check_gpu_availability()
        self._index = None
        self._data = None
        self._data_gpu = None

    @classmethod
    def _check_gpu_availability(cls) -> bool:
        """Check if GPU acceleration is available and functional.

        This is cached across all instances to avoid repeated GPU initialization attempts.

        Returns:
            True if cuVS is available and functional, False otherwise.
        """
        if cls._gpu_checked:
            return cls._gpu_available

        ensure_nvidia_libraries_preloaded()

        # Try to import cuVS for GPU acceleration
        # We need to verify both import AND actual GPU functionality
        try:
            import cupy as cp_module  # ty: ignore[unresolved-import]
            from cuvs.neighbors import brute_force as cuvs_module  # ty: ignore[unresolved-import]

            # Test that cuVS is actually usable (not just that imports work)
            # This catches cases where CUDA driver/runtime version mismatch
            test_data = cp_module.array([[1.0, 2.0], [3.0, 4.0]], dtype=cp_module.float32)
            test_query = cp_module.array([[1.0, 2.0]], dtype=cp_module.float32)
            test_index = cuvs_module.build(test_data, metric="sqeuclidean")
            test_dist, test_idx = cuvs_module.search(test_index, test_query, 1)
            _ = cp_module.asnumpy(test_dist)  # Force transfer back to CPU
            del test_data, test_query, test_index, test_dist, test_idx

            # If we got here, cuVS is fully functional
            cls._cuvs_brute_force = cuvs_module
            cls._cp = cp_module
            cls._gpu_available = True
        except (ImportError, RuntimeError, OSError, Exception):
            # ImportError: cupy/cuvs not installed
            # RuntimeError: CUDA libraries not available
            # OSError: Library loading failed
            # Exception: Any other GPU-related failure
            cls._gpu_available = False

        cls._gpu_checked = True
        return cls._gpu_available

    def fit(self, data: np.ndarray) -> NearestNeighborSearch:
        """Build the search index from data.

        Args:
            data: Array of shape (n_samples, n_features) with float32 values.

        Returns:
            Self for method chaining.
        """
        # Ensure float32 and contiguous
        data = np.ascontiguousarray(data, dtype=np.float32)
        self._data = data

        if self.use_gpu:
            # Transfer to GPU and build cuVS index
            self._data_gpu = self._cp.asarray(data)
            self._index = self._cuvs_brute_force.build(self._data_gpu, metric="sqeuclidean")
        else:
            # Build sklearn index
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
        if self._index is None:
            raise RuntimeError("Must call fit() before kneighbors()")

        k = n_neighbors if n_neighbors is not None else self.n_neighbors

        # Ensure float32 and contiguous
        queries = np.ascontiguousarray(queries, dtype=np.float32)

        if self.use_gpu:
            # Transfer queries to GPU
            queries_gpu = self._cp.asarray(queries)

            # Search using cuVS
            distances_gpu, indices_gpu = self._cuvs_brute_force.search(self._index, queries_gpu, k)

            # Transfer results back to CPU
            distances = self._cp.asnumpy(distances_gpu)
            indices = self._cp.asnumpy(indices_gpu)

            # cuVS returns squared distances, convert to euclidean
            distances = np.sqrt(distances)
        else:
            # Use sklearn
            distances, indices = self._index.kneighbors(queries, n_neighbors=k)

        return distances, indices

    @property
    def backend_name(self) -> str:
        """Return the name of the backend being used."""
        return "cuVS (GPU)" if self.use_gpu else "sklearn (CPU)"


def get_nearest_neighbors(data: np.ndarray, queries: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Convenience function for one-shot nearest neighbor search.

    Args:
        data: Array of shape (n_samples, n_features) to search in.
        queries: Array of shape (n_queries, n_features) to search for.
        k: Number of neighbors to return.

    Returns:
        Tuple of (distances, indices).
    """
    nn = NearestNeighborSearch(n_neighbors=k)
    nn.fit(data)
    return nn.kneighbors(queries)
