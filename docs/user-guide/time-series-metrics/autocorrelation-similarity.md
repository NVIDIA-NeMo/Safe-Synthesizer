<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Autocorrelation Similarity

Autocorrelation Similarity measures whether synthetic values depend on their
recent history in the same way as training values. It can reveal lost
persistence, incorrect oscillation, overly smooth or repetitive behavior, and
synthetic sequences whose temporal order has been disrupted.

## Reading the score

A higher score means the training and synthetic autocorrelation profiles are more alike.

- A score near 10 can occur when synthetic and training profiles overlap across the evaluated lags.
- A middle score can occur when the general profile shape is preserved but a
  cycle has shifted or persistence decays at a different rate.
- A score near 0 can occur when synthetic values are constant or the lag relationships oppose the training profile.

These examples are not calibrated quality bands. Compare scores only when the
selected columns, groups, and `max_lag` are the same. Whether a difference
matters depends on the downstream use case. For example, preserving short-term
dependence may be critical for sensor simulation but unimportant for a workload
that uses only long-term totals.

## Calculation

For every usable group and numeric value column, the metric computes the Pearson
correlation between values at positions `t` and `t + lag`. Each lag uses only
endpoint pairs that are finite in that sequence and requires at least three
pairs. It then takes the mean absolute difference between the training and
synthetic profiles, divides by 2 to map the maximum possible difference to 1,
and calculates:

`profile similarity = 1 - (mean absolute profile difference / 2)`

The final 0–10 score is ten times the mean profile similarity. Therefore, 10
means matching profiles and lower scores mean larger lag-by-lag disagreement.

Non-finite observations remain as gaps in their original temporal positions, so
an invalid observation cannot collapse the time axis. A comparison is skipped
when the training series is constant because its autocorrelation is undefined.
If the training series varies but the synthetic series is constant, that
comparison receives zero similarity to represent complete loss of temporal
variation.

## Configuration

Time-series evaluation is off by default. Enable it explicitly to compute the
metric and add the time-series section to the HTML evaluation report.

```yaml
time_series:
  is_timeseries: true
  timestamp_column: time
evaluation:
  time_series:
    enabled: true
    autocorrelation:
      value_columns: null
      max_lag: 20
      min_points: 4
      max_groups: 128
```

`value_columns: null` evaluates all shared columns inferred as numeric. Set an
explicit list to evaluate only selected numeric channels. Timestamp ordering
uses `time_series.timestamp_column`, and sequence grouping uses
`data.group_training_examples_by`.

If more than `max_groups` groups are shared, the metric evaluates a reproducible
seeded sample instead of favoring labels that sort first. The result reports the
total, evaluated, and omitted shared-group counts. Changing `value_columns`,
`max_lag`, or the grouping configuration changes what the aggregate score
measures.

## Diagnosing and improving a low score

Start by checking whether the low score is widespread or concentrated in particular columns or groups. Then compare the sequence and autocorrelation plots:

- If synthetic autocorrelation decays more quickly than training
  autocorrelation, verify `time_series.timestamp_column`,
  `time_series.timestamp_interval_seconds`, and
  `data.group_training_examples_by`. If ordering is correct, modestly lower
  `generation.temperature` and rerun the evaluation.
- If synthetic autocorrelation remains higher than training autocorrelation for
  the same lags, confirm that the training data contains the expected
  short-term variation. For overly smooth or repetitive output, modestly
  increase `generation.temperature` or set `generation.repetition_penalty`
  slightly above 1, changing one parameter at a time.
- If peaks or sign changes occur at different lags than in training, verify
  `time_series.timestamp_interval_seconds` and confirm that the training data
  contains multiple examples of the expected cycle. Increase `max_lag` only
  when the behavior that matters lies beyond the current evaluation horizon.
- If only some groups score poorly, use the per-group details to identify them
  and improve their training coverage. If groups require materially different
  modeling behavior, split them into separate datasets and synthesis runs.

## Limitations

Autocorrelation summarizes average linear dependence within one value channel.
Different sequences can have similar autocorrelation profiles, so a high score
does not mean that individual events, amplitudes, or phases match. A single
profile can also hide local regime changes, and irregular sampling intervals
can make lag comparisons misleading. Binary and categorical channels require
different lag-association measures and are not selected automatically. The
metric does not establish causality or show whether the synthetic data
memorizes training sequences. Interpret the score at a lag range and sequence
granularity that match the behavior you need to preserve.
