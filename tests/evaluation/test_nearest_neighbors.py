# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NearestNeighborSearch to verify L2 distance semantics.

Both backends (torch GPU and sklearn CPU) must return true L2 (euclidean) distances,
not squared euclidean distances. torch.cdist uses p=2 (euclidean) directly;
sklearn uses euclidean directly. These tests guard that contract.
"""

import numpy as np
import pytest
import torch

from nemo_safe_synthesizer.evaluation.nearest_neighbors import NearestNeighborSearch


def _make_known_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a small dataset with analytically known L2 distances.

    Returns:
        (data, queries, expected_distances) where expected_distances[i][j]
        is the L2 distance from queries[i] to its j-th nearest neighbor.
    """
    data = np.array(
        [
            [0.0, 0.0],  # idx 0
            [3.0, 4.0],  # idx 1 — distance 5.0 from origin
            [1.0, 0.0],  # idx 2 — distance 1.0 from origin
            [0.0, 1.0],  # idx 3 — distance 1.0 from origin
        ],
        dtype=np.float32,
    )
    queries = np.array([[0.0, 0.0]], dtype=np.float32)

    # Sorted nearest → farthest: idx0 (0.0), idx2 (1.0), idx3 (1.0), idx1 (5.0)
    expected_distances = np.array([[0.0, 1.0, 1.0, 5.0]], dtype=np.float32)
    return data, queries, expected_distances


class TestNearestNeighborSearchReturnsL2Distances:
    """Ensure kneighbors returns L2 distances (not squared) on every backend."""

    def test_cpu_backend_returns_l2_distances(self):
        """Sklearn CPU path must return euclidean distances, not squared."""
        data, queries, expected = _make_known_data()

        nn = NearestNeighborSearch(n_neighbors=4)
        # Force CPU regardless of GPU availability
        nn.use_gpu = False
        nn.fit(data)
        distances, _ = nn.kneighbors(queries)

        np.testing.assert_allclose(distances, expected, atol=1e-6)

    def test_torch_path_returns_l2_distances(self):
        """Torch GPU path must return L2 distances from torch.cdist (p=2, not squared)."""
        data, queries, expected = _make_known_data()

        # Force the torch path on CPU so no real GPU is needed
        nn = NearestNeighborSearch.__new__(NearestNeighborSearch)
        nn.n_neighbors = 4
        nn.use_gpu = True
        nn._torch_device = torch.device("cpu")
        nn._index = None
        nn._data_t = None

        nn.fit(data)
        distances, _ = nn.kneighbors(queries)

        np.testing.assert_allclose(distances, expected, atol=1e-6)

    def test_distances_are_not_squared(self):
        """Explicitly verify returned values differ from squared euclidean.

        The 3-4-5 triangle gives L2=5.0 vs squared=25.0 — an unmistakable gap.
        """
        data = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)
        queries = np.array([[0.0, 0.0]], dtype=np.float32)

        nn = NearestNeighborSearch(n_neighbors=2)
        nn.use_gpu = False
        nn.fit(data)
        distances, _ = nn.kneighbors(queries)

        farthest_distance = distances[0, 1]
        assert farthest_distance == pytest.approx(5.0, abs=1e-6), (
            f"Expected L2 distance 5.0, got {farthest_distance}. If 25.0, the backend is returning squared distances."
        )
        assert farthest_distance != pytest.approx(25.0, abs=1e-3)

    @pytest.mark.requires_gpu
    def test_real_gpu_backend_returns_l2_distances(self):
        """When a CUDA GPU is available, verify the real torch CUDA path returns L2 distances."""
        nn = NearestNeighborSearch(n_neighbors=4)
        if not nn.use_gpu:
            pytest.skip("CUDA GPU not available")

        data, queries, expected = _make_known_data()
        nn.fit(data)
        distances, _ = nn.kneighbors(queries)

        np.testing.assert_allclose(distances, expected, atol=1e-6)

    def test_higher_dimensional_l2_distances(self):
        """Verify L2 distances in higher dimensions (not just 2D)."""
        rng = np.random.RandomState(42)
        data = rng.randn(50, 10).astype(np.float32)
        queries = rng.randn(5, 10).astype(np.float32)

        nn = NearestNeighborSearch(n_neighbors=3)
        nn.use_gpu = False
        nn.fit(data)
        distances, indices = nn.kneighbors(queries)

        for i in range(queries.shape[0]):
            for j in range(3):
                neighbor_idx = indices[i, j]
                expected_l2 = np.sqrt(np.sum((queries[i] - data[neighbor_idx]) ** 2))
                assert distances[i, j] == pytest.approx(expected_l2, rel=1e-5), (
                    f"Query {i}, neighbor {j}: got {distances[i, j]}, expected L2={expected_l2}"
                )
