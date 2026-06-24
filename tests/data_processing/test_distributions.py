# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import datetime, timedelta

from typing_extensions import override

from nemo_safe_synthesizer.data_processing.actions.distributions import DatetimeDistribution


class FixedDatetimeDistribution(DatetimeDistribution):
    @override
    def sample_datetimes(self, num_records: int) -> list[datetime]:
        start = datetime(2024, 1, 1, 12, 20)
        return [start + timedelta(minutes=20 * offset) for offset in range(num_records)]


def test_datetime_distribution_applies_precision_and_format_without_mutating_samples():
    distribution = FixedDatetimeDistribution(precision=timedelta(hours=1), format="%H:%M")

    assert distribution.sample(2) == ["12:00", "13:00"]


def test_datetime_distribution_applies_precision_without_format():
    distribution = FixedDatetimeDistribution(precision=timedelta(hours=1))

    assert distribution.sample(2) == [
        datetime(2024, 1, 1, 12),
        datetime(2024, 1, 1, 13),
    ]
