<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Autocorrelation Similarity

!!! note "Standalone metric"

    This metric is available for standalone evaluation only. It is not yet included in an aggregate score or the default report.

Autocorrelation Similarity measures whether synthetic values depend on their recent history in the same way as real values. It can reveal lost persistence, incorrect oscillation, over-smoothed dynamics, and synthetic sequences whose order has effectively been shuffled.

## Reading the score

A higher score means the real and synthetic autocorrelation profiles are more alike.

| Band | Score | Interpretation |
| --- | --- | --- |
| Low | 0.0–4.9 | The synthetic lag structure is substantially different. |
| Medium | 5.0–6.9 | Some temporal dependence is preserved, but important lags differ. |
| High | 7.0–10.0 | The synthetic series preserves the real short-range dependence well. |

![Three autocorrelation plots comparing real and synthetic lag profiles. The low example scores 4.0 and has a slowly decaying synthetic profile, the medium example scores 6.7 and partially follows the real oscillation, and the high example scores 10.0 with overlapping profiles.](../../assets/time-series-metrics/autocorrelation-similarity/score-examples.png)

The examples are computed by the metric with `max_lag: 5`: low 4.0, medium 6.7, and high 10.0. The illustration uses periodic signals with increasingly similar lag structure.

## Data and grouping requirements

- Real and synthetic data need a shared timestamp column and at least one shared numeric value column. Set `value_columns` to restrict the comparison; otherwise all shared numeric columns except the timestamp and group columns are used.
- Rows are sorted by the configured timestamp. With `group_column`, profiles are computed independently for each shared group and value column, then averaged. A sequence never crosses a group boundary.
- A test set is not required.
- Each group/column comparison needs at least `min_points` finite values, with a default of 4. Constant or near-constant series cannot produce a usable autocorrelation profile.
- The effective maximum lag is the smaller of `max_lag` and half the available sequence length.

The result is `UNAVAILABLE` when the metric is disabled, required columns or shared groups are absent, or every group/column comparison is too short, constant, or otherwise has no stable lag. Usable comparisons still contribute when only some comparisons are skipped.

## Calculation

For every usable group and value column, the metric computes autocorrelation at lags 1 through the effective maximum lag. It takes the mean absolute difference between the real and synthetic profiles, divides by 2 to map the maximum possible difference to 1, and calculates:

`atomic similarity = 1 - mean_absolute_profile_difference / 2`

The final 0–10 score is ten times the mean atomic similarity. Therefore, 10 means matching profiles and lower scores mean larger lag-by-lag disagreement.

## Configuration

```yaml
time_series:
  is_timeseries: true
  timestamp_column: time
evaluation:
  time_series:
    autocorrelation:
      enabled: true
      value_columns: [value]
      group_column: entity_id
      max_lag: 20
      min_points: 4
      max_groups: 128
```

## Diagnosing and improving a low score

Inspect the per-column and per-group details to identify whether the mismatch is global or concentrated in particular entities. In the plot, check for missing peaks, incorrect sign changes, or decay that is too fast or too slow. Confirm timestamp ordering and group boundaries first. Then review whether sequence length, context length, and training examples expose the generator to the relevant temporal span. A larger `max_lag` tests longer memory but also requires longer sequences.

## Limitations and complementary metrics

Autocorrelation describes linear dependence within one channel. It does not identify cross-channel lead/lag relationships, local regime changes, frequency composition, or privacy leakage. Pair it with Lagged Dependency Fidelity for channel interactions, Rolling Statistics Similarity for local level and volatility, Spectral Similarity for periodic behavior, and Window Membership Inference Protection for privacy.
