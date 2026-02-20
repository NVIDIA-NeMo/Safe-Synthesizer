# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark test comparing nearest neighbor implementations.

This benchmark evaluates the performance difference between:
- sklearn NearestNeighbors (CPU) - the fallback implementation
- cuVS (NVIDIA RAPIDS GPU) - the accelerated implementation
- FAISS GPU - Facebook's similarity search library with GPU support

The NearestNeighborSearch abstraction in nemo_safe_synthesizer.evaluation.nearest_neighbors
automatically selects the best available backend.

Run with:
    uv run --frozen pytest packages/nemo_safe_synthesizer/tests/evaluation/components/benchmark_nearest_neighbor.py -v -s -m benchmark

Or standalone:
    uv run --frozen --extra cu128 python packages/nemo_safe_synthesizer/tests/evaluation/components/benchmark_nearest_neighbor.py
"""

import logging
import os

# Must be set BEFORE importing numpy/sklearn (which load OpenBLAS)
# Cap threads to avoid OpenBLAS crashes on high-core machines (>128 cores)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import time
from dataclasses import dataclass

import numpy as np
import pytest
from nemo_safe_synthesizer.evaluation._cuda_preload import ensure_nvidia_libraries_preloaded
from sklearn.neighbors import NearestNeighbors

ensure_nvidia_libraries_preloaded()

# Optional cuVS/RAPIDS import
# We need to verify both import AND actual GPU functionality
CUVS_AVAILABLE = False
cuvs_brute_force = None
cp = None

try:
    import cupy as _cp
    from cuvs.neighbors import brute_force as _cuvs_brute_force

    # Test that cuVS is actually usable (not just that imports work)
    _test_data = _cp.array([[1.0, 2.0], [3.0, 4.0]], dtype=_cp.float32)
    _test_query = _cp.array([[1.0, 2.0]], dtype=_cp.float32)
    _test_index = _cuvs_brute_force.build(_test_data, metric="sqeuclidean")
    _test_dist, _test_idx = _cuvs_brute_force.search(_test_index, _test_query, 1)
    _ = _cp.asnumpy(_test_dist)
    del _test_data, _test_query, _test_index, _test_dist, _test_idx

    cuvs_brute_force = _cuvs_brute_force
    cp = _cp
    CUVS_AVAILABLE = True
except (ImportError, RuntimeError, OSError, Exception):
    logging.exception("cuVS not available")
    CUVS_AVAILABLE = False

# Optional FAISS GPU import, install with uv pip install faiss-gpu-cu12
FAISS_GPU_AVAILABLE = False
faiss = None

try:
    import faiss as _faiss

    # Check if GPU support is available
    if _faiss.get_num_gpus() > 0:
        faiss = _faiss
        FAISS_GPU_AVAILABLE = True
    else:
        logging.info("FAISS installed but no GPU available")
except ImportError:
    logging.info("FAISS not available")
except Exception:
    logging.exception("FAISS GPU check failed")


@dataclass
class BenchmarkResult:
    """Results from a nearest neighbor benchmark run."""

    name: str
    fit_time_sec: float
    query_time_sec: float
    total_time_sec: float
    n_samples: int
    n_features: int
    n_queries: int
    k: int

    def __str__(self) -> str:
        return (
            f"{self.name}: fit={self.fit_time_sec:.3f}s, "
            f"query={self.query_time_sec:.3f}s, total={self.total_time_sec:.3f}s"
        )


def generate_data(n_samples: int, n_features: int, seed: int = 42) -> np.ndarray:
    """Generate random float32 data for benchmarking."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_samples, n_features)).astype(np.float32)


def benchmark_sklearn(data: np.ndarray, queries: np.ndarray, k: int = 5) -> BenchmarkResult:
    """Benchmark sklearn NearestNeighbors with brute force algorithm."""
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")

    t0 = time.perf_counter()
    nn.fit(data)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    distances, indices = nn.kneighbors(queries)
    query_time = time.perf_counter() - t0

    return BenchmarkResult(
        name="sklearn_cpu",
        fit_time_sec=fit_time,
        query_time_sec=query_time,
        total_time_sec=fit_time + query_time,
        n_samples=data.shape[0],
        n_features=data.shape[1],
        n_queries=queries.shape[0],
        k=k,
    )


def benchmark_cuvs(data: np.ndarray, queries: np.ndarray, k: int = 5) -> BenchmarkResult | None:
    """Benchmark cuVS brute force nearest neighbors on GPU."""
    if not CUVS_AVAILABLE:
        return None

    t0 = time.perf_counter()
    # Convert to cupy arrays (GPU)
    data_gpu = cp.asarray(data, dtype=cp.float32)
    queries_gpu = cp.asarray(queries, dtype=cp.float32)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Build index and search
    index = cuvs_brute_force.build(data_gpu, metric="sqeuclidean")
    distances, indices = cuvs_brute_force.search(index, queries_gpu, k)
    # Sync GPU
    cp.cuda.Stream.null.synchronize()
    query_time = time.perf_counter() - t0

    return BenchmarkResult(
        name="cuvs_gpu",
        fit_time_sec=fit_time,
        query_time_sec=query_time,
        total_time_sec=fit_time + query_time,
        n_samples=data.shape[0],
        n_features=data.shape[1],
        n_queries=queries.shape[0],
        k=k,
    )


def benchmark_faiss_gpu(data: np.ndarray, queries: np.ndarray, k: int = 5) -> BenchmarkResult | None:
    """Benchmark FAISS GPU for exact L2 nearest neighbors."""
    if not FAISS_GPU_AVAILABLE:
        return None

    t0 = time.perf_counter()
    # Create CPU index first
    d = data.shape[1]
    cpu_index = faiss.IndexFlatL2(d)
    # Transfer to GPU
    gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, cpu_index)
    gpu_index.add(data)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    # Search
    distances, indices = gpu_index.search(queries, k)
    query_time = time.perf_counter() - t0

    return BenchmarkResult(
        name="faiss_gpu",
        fit_time_sec=fit_time,
        query_time_sec=query_time,
        total_time_sec=fit_time + query_time,
        n_samples=data.shape[0],
        n_features=data.shape[1],
        n_queries=queries.shape[0],
        k=k,
    )


def benchmark_abstraction(data: np.ndarray, queries: np.ndarray, k: int = 5) -> BenchmarkResult:
    """Benchmark the NearestNeighborSearch abstraction (auto-selects best backend)."""
    from nemo_safe_synthesizer.evaluation.nearest_neighbors import NearestNeighborSearch

    nn = NearestNeighborSearch(n_neighbors=k)

    t0 = time.perf_counter()
    nn.fit(data)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    distances, indices = nn.kneighbors(queries)
    query_time = time.perf_counter() - t0

    return BenchmarkResult(
        name=f"abstraction ({nn.backend_name})",
        fit_time_sec=fit_time,
        query_time_sec=query_time,
        total_time_sec=fit_time + query_time,
        n_samples=data.shape[0],
        n_features=data.shape[1],
        n_queries=queries.shape[0],
        k=k,
    )


def run_benchmark(
    n_samples: int = 100_000,
    n_features: int = 100,
    n_queries: int = 1_000,
    k: int = 5,
) -> list[BenchmarkResult]:
    """Run all benchmarks and return results."""
    print(f"\n{'=' * 60}")
    print(f"Benchmark: {n_samples:,} samples × {n_features} features")
    print(f"Queries: {n_queries:,}, k={k}")
    print("=" * 60)

    data = generate_data(n_samples, n_features, seed=420)
    queries = generate_data(n_queries, n_features, seed=123)

    results = []

    # sklearn (baseline)
    result = benchmark_sklearn(data, queries, k)
    results.append(result)
    print(f"  {result}")

    # cuVS if available
    result = benchmark_cuvs(data, queries, k)
    if result:
        results.append(result)
        print(f"  {result}")
    else:
        print("  cuvs_gpu: NOT AVAILABLE")

    # FAISS GPU if available
    result = benchmark_faiss_gpu(data, queries, k)
    if result:
        results.append(result)
        print(f"  {result}")
    else:
        print("  faiss_gpu: NOT AVAILABLE")

    # Test the abstraction layer
    result = benchmark_abstraction(data, queries, k)
    results.append(result)
    print(f"  {result}")

    # Print speedup comparison
    sklearn_time = results[0].total_time_sec
    print("\n  Speedups vs sklearn:")
    for r in results[1:]:
        speedup = sklearn_time / r.total_time_sec if r.total_time_sec > 0 else 0
        print(f"    {r.name}: {speedup:.1f}×")

    return results


# ============================================================================
# Pytest tests
# ============================================================================


@pytest.mark.benchmark
def test_benchmark_small_dataset():
    """Benchmark on small dataset (10k rows) - baseline for comparison."""
    results = run_benchmark(n_samples=10_000, n_features=100, n_queries=500, k=5)

    # sklearn should complete in reasonable time for small datasets
    sklearn_result = results[0]
    assert sklearn_result.total_time_sec < 30, "sklearn too slow on small dataset"


@pytest.mark.benchmark
def test_benchmark_medium_dataset():
    """Benchmark on medium dataset (50k rows)."""
    results = run_benchmark(n_samples=50_000, n_features=100, n_queries=500, k=5)

    sklearn_result = results[0]
    print(f"\nMedium dataset sklearn time: {sklearn_result.total_time_sec:.2f}s")


@pytest.mark.benchmark
def test_benchmark_large_dataset():
    """Benchmark on large dataset (100k rows) - target size for evaluation."""
    results = run_benchmark(n_samples=100_000, n_features=100, n_queries=1_000, k=5)

    sklearn_result = results[0]
    print(f"\nLarge dataset sklearn time: {sklearn_result.total_time_sec:.2f}s")

    # Report all speedups
    for r in results[1:]:
        speedup = sklearn_result.total_time_sec / r.total_time_sec if r.total_time_sec > 0 else 0
        print(f"{r.name} speedup: {speedup:.1f}×")


@pytest.mark.benchmark
@pytest.mark.skipif(not CUVS_AVAILABLE, reason="cuVS not available")
def test_benchmark_gpu_stress():
    """Benchmark on very large dataset to show GPU benefit."""
    results = run_benchmark(n_samples=500_000, n_features=100, n_queries=10_000, k=5)

    sklearn_result = results[0]
    print(f"\nGPU stress test sklearn time: {sklearn_result.total_time_sec:.2f}s")

    # GPU backends should show significant speedup
    for r in results:
        if "gpu" in r.name.lower():
            speedup = sklearn_result.total_time_sec / r.total_time_sec if r.total_time_sec > 0 else 0
            print(f"{r.name} speedup: {speedup:.1f}×")
            # GPU should be at least 5x faster on large data
            assert speedup > 5, f"Expected GPU to be >5x faster, got {speedup:.1f}x"


@pytest.mark.benchmark
def test_benchmark_typical_aia_mia_workload():
    """Benchmark typical AIA/MIA evaluation workload.

    Based on actual usage patterns:
    - Training data: ~5k-50k rows
    - Synthetic data: ~5k-50k rows
    - Features: typically 10-50 after normalization
    - k: 1-10 nearest neighbors
    """
    print("\n" + "=" * 60)
    print("Typical AIA/MIA workload simulation")
    print("=" * 60)

    # Typical small dataset
    run_benchmark(n_samples=5_000, n_features=30, n_queries=200, k=5)

    # Typical medium dataset
    run_benchmark(n_samples=20_000, n_features=30, n_queries=200, k=5)

    # Large dataset (stress test)
    run_benchmark(n_samples=50_000, n_features=50, n_queries=500, k=5)


# ============================================================================
# Standalone execution
# ============================================================================

if __name__ == "__main__":
    print("Nearest Neighbor Benchmark: sklearn vs cuVS vs FAISS GPU")
    print(f"cuVS available: {CUVS_AVAILABLE}")
    print(f"FAISS GPU available: {FAISS_GPU_AVAILABLE}")

    # Run comprehensive benchmarks
    print("\n" + "=" * 70)
    print("COMPREHENSIVE BENCHMARK SUITE")
    print("=" * 70)

    # Typical AIA/MIA workloads
    print("\n--- Typical AIA/MIA Workloads ---")
    results = []
    results.extend(run_benchmark(n_samples=5_000, n_features=30, n_queries=200, k=5))
    results.extend(run_benchmark(n_samples=20_000, n_features=30, n_queries=200, k=5))

    # Stress tests
    print("\n--- Stress Tests (Large Data) ---")
    results.extend(run_benchmark(n_samples=50_000, n_features=100, n_queries=500, k=5))
    results.extend(run_benchmark(n_samples=100_000, n_features=100, n_queries=1_000, k=5))

    # GPU stress test (larger for GPU to show benefit)
    results.extend(run_benchmark(n_samples=500_000, n_features=100, n_queries=10_000, k=5))
    results.extend(run_benchmark(n_samples=1_000_000, n_features=200, n_queries=10_000, k=5))

    # Dimension scaling
    print("\n--- Dimension Scaling (50k samples) ---")
    for dim in [20, 50, 100, 200]:
        results.extend(run_benchmark(n_samples=50_000, n_features=dim, n_queries=500, k=5))
