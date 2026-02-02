# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_safe_synthesizer.data_processing.stats import RunningStatistics, Statistics


def test_statistics():
    stats_with_count = Statistics(count=5, sum_sq_diffs=16.0)
    assert stats_with_count.count == 5
    assert stats_with_count.mean == 0
    assert stats_with_count.sum_sq_diffs == 16.0
    assert stats_with_count.min == float("inf")
    assert stats_with_count.max == float("-inf")
    assert stats_with_count.stddev == 2

    stats_with_no_count = Statistics()
    assert stats_with_no_count.count == 0
    assert stats_with_no_count.sum_sq_diffs == 0
    assert stats_with_no_count.stddev == 0


def test_running_statistics():
    run_stats = RunningStatistics()
    run_stats.update(1)
    run_stats.update(3)
    run_stats.update(-0.1)

    assert run_stats.count == 3
    assert run_stats.min == -0.1
    assert run_stats.max == 3
    assert round(run_stats.mean, 4) == 1.3
    assert round(run_stats.sum_sq_diffs, 4) == 4.94
    assert round(run_stats.stddev, 4) == 1.5716

    run_stats.reset()
    assert run_stats.count == 0
    assert run_stats.mean == 0
    assert run_stats.sum_sq_diffs == 0
    assert run_stats.min == float("inf")
    assert run_stats.max == float("-inf")
    assert run_stats.stddev == 0
