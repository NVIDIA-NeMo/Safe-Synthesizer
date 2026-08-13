<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Autocorrelation Similarity

Autocorrelation Similarity measures whether synthetic values depend on their recent history in the same way as training values. It can reveal lost persistence, incorrect oscillation, overly smooth or repetitive behavior, and synthetic sequences whose temporal order has been disrupted.

## Reading the score

A higher score means the training and synthetic autocorrelation profiles are more alike.

| Band | Score | Interpretation |
| --- | --- | --- |
| Low | 0.0–4.9 | The synthetic lag structure is substantially different. |
| Medium | 5.0–6.9 | Some temporal dependence is preserved, but important lags differ. |
| High | 7.0–10.0 | The synthetic series preserves the training short-range dependence well. |

![Three examples arranged from low to high score. Each example shows the original training and synthetic sequences above their autocorrelation profiles. The low example has a much slower synthetic oscillation, the medium example has a related but longer synthetic cycle, and the high example has overlapping sequences and profiles.](../../assets/time-series-metrics/autocorrelation-similarity/score-examples.png)

The scores are computed by the metric with `max_lag: 5`: low 4.0, medium 6.7, and high 10.0. The sequence panels show the first 80 of the 240 points used to calculate each score.

- Low (4.0) -- The synthetic sequence oscillates much more slowly, and its autocorrelation remains strongly positive.
- Medium (6.7) -- The synthetic sequence has a related but longer cycle.
- High (10.0) -- The sequences and autocorrelation profiles overlap.

## Calculation

For every usable group and value column, the metric computes autocorrelation at lags 1 through the effective maximum lag. It takes the mean absolute difference between the training and synthetic profiles, divides by 2 to map the maximum possible difference to 1, and calculates:

`atomic similarity = 1 - mean_absolute_profile_difference / 2`

The final 0–10 score is ten times the mean atomic similarity. Therefore, 10 means matching profiles and lower scores mean larger lag-by-lag disagreement.

Non-finite observations remain as gaps in their original temporal positions. Each lag uses only pairs whose two endpoints are finite, so removing an invalid observation cannot collapse the time axis. A comparison is skipped when the training series is constant because its autocorrelation is undefined. If the training series varies but the synthetic series is constant, that comparison receives zero similarity to represent complete loss of temporal variation.

## Configuration

The following block shows the default autocorrelation settings.

```yaml
time_series:
  is_timeseries: true
  timestamp_column: time
evaluation:
  time_series:
    autocorrelation:
      enabled: null
      value_columns: null
      group_column: null
      max_lag: 20
      min_points: 4
      max_groups: 128
```

With the defaults, `enabled: null` automatically enables the metric for time-series data, `value_columns: null` evaluates all shared numeric value columns, timestamp ordering uses the top-level `time_series.timestamp_column`, and `group_column: null` uses the top-level data grouping setting.

If more than `max_groups` groups are shared, the metric evaluates a reproducible hash-selected subset instead of favoring labels that sort first. The result notes and details report the total, evaluated, and omitted group counts.

## Diagnosing and improving a low score

Start by checking whether the low score is widespread or concentrated in particular columns or groups. Then compare the sequence and autocorrelation plots:

- If synthetic autocorrelation decays too quickly, the generated values are losing persistence. Preserve row order, then lengthen training examples and generated sequences until they cover the dependency span before retraining and regenerating.
- If synthetic autocorrelation stays high for too long, the generated sequences may be overly smooth or repetitive. Remove smoothing postprocessing, check for duplicated generated patterns, and retrain with examples that preserve the expected short-term variation.
- If peaks or sign changes occur at the wrong lags, verify the timestamp interval and make sure training examples contain multiple complete cycles. Increase `max_lag` only when the expected cycle lies beyond the current evaluation horizon.
- If only some groups score poorly, use the per-group details to identify them, then add representative training coverage or use a separate synthesis configuration for groups with distinct temporal behavior.

Correct timestamp ordering, group boundaries, and sampling intervals before changing the generator. After data preparation is verified, improve the synthetic data by preserving longer temporal context, representing slow and fast patterns in the training examples, and avoiding generation or postprocessing steps that break sequence order. Change `max_lag` only when the evaluation horizon is wrong for the use case; changing it does not improve the synthetic data itself.

## Limitations

Autocorrelation summarizes average linear dependence within one value channel. Different sequences can have similar autocorrelation profiles, so a high score does not mean that individual events, amplitudes, or phases match. A single profile can also hide local regime changes, and irregular sampling intervals can make lag comparisons misleading. The metric does not establish causality or show whether the synthetic data memorizes training sequences. Interpret the score at a lag range and sequence granularity that match the behavior you need to preserve.
