# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from functools import partial
from typing import Annotated, Any, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field


class Distribution(BaseModel, ABC):
    """
    Abstract base class representing a distribution.
    Child classes should specify whichever arguments are needed
    to properly parametrize their distribution.
    """

    @abstractmethod
    def sample(self, num_records: int) -> list[Any]: ...


class DatetimeDistribution(BaseModel, ABC):
    """
    This class is separate from the `Distribution` ABC above
    because datetimes need slightly different handling than floats.
    Providing this separate class hierarchy also makes it easier
    in pydantic to specify what datatypes we expect in the distribution
    parameters (float vs datetime), as well as dt-specific arguments.

    In practice, this means creating a "copy" `DatetimeDistribution`
    for each regular `Distribution` where it makes sense. We could probably
    automate some of this with generics, but IMO that'd just make it confusing
    to read. We're still able to reuse the original `Distribution` class most
    of the time in `DatetimeDistribution`, making the only business logic
    really be about how we want to translate dates --> floats.
    """

    precision: Optional[timedelta] = None
    """
    `precision` controls how we precisely want to round the dates.

    Most `DatetimeDistribution`s sample from their distribution by first
    sampling from an underlying float-based distribution. When this is then converted
    back into datetimes, these datetimes always have specific hours/minutes/seconds.

    `precision` is technically a `timedelta` field, so the `timedelta` you pass in
    is the unit to which the datetimes will be rounded to. If invoking a distribution
    via json/yaml, look at pydantic's timedelta parsing:
        https://docs.pydantic.dev/2.2/usage/types/datetime/

    Examples:
        precision: 00:10  # rounded to nearest 10 minutes
        precision: 01:00  # rounded to nearest hour
        precision: 1      # rounded to whole day
    """

    format: Optional[str] = None
    """
    `format` will determine the string output format of the generated datetimes.
    The expected strings match inputs to the `strftime` function.
    """

    @abstractmethod
    def sample_datetimes(self, num_records: int) -> list[datetime]: ...

    def sample(self, num_records: int) -> list[datetime] | list[str]:
        samples = self.sample_datetimes(num_records)
        return self._apply_universal_params(samples)

    def _round_datetime(self, dt: datetime, precision: timedelta) -> datetime:
        rounded_ts = round(dt.timestamp() / precision.total_seconds()) * precision.total_seconds()
        return datetime.fromtimestamp(rounded_ts)

    def _apply_universal_params(self, samples: list[datetime]) -> list[datetime]:
        ret: list[str] | list[datetime] = samples

        ops = []
        if self.precision is not None:
            ops.append(partial(self._round_datetime, self.precision))
        if self.format is not None:
            ops.append(lambda x: x.strftime(self.format))

        for sample in samples:
            n = sample
            for op in ops:
                n = op(n)
            ret.append(n)

        return ret


class GaussianDistribution(Distribution):
    distribution_type: Literal["gaussian"] = "gaussian"
    mean: float
    std_dev: float = Field(gt=0)

    def sample(self, num_records: int) -> list[float]:
        return np.random.normal(loc=self.mean, scale=self.std_dev, size=num_records).tolist()


class DatetimeGaussianDistribution(DatetimeDistribution):
    distribution_type: Literal["gaussian"] = "gaussian"
    mean: datetime
    std_dev: timedelta

    def sample_datetimes(self, num_records: int) -> list[datetime]:
        float_samples = GaussianDistribution(mean=self.mean.timestamp(), std_dev=self.std_dev.total_seconds()).sample(
            num_records=num_records
        )
        return [datetime.fromtimestamp(sample) for sample in float_samples]


class UniformDistribution(Distribution):
    distribution_type: Literal["uniform"] = "uniform"
    low: float
    high: float

    def sample(self, num_records: int) -> list[float]:
        return np.random.uniform(low=self.low, high=self.high, size=num_records).tolist()


class DatetimeUniformDistribution(DatetimeDistribution):
    distribution_type: Literal["uniform"] = "uniform"
    low: datetime
    high: datetime

    def sample_datetimes(self, num_records: int) -> list[datetime]:
        float_samples = UniformDistribution(low=self.low.timestamp(), high=self.high.timestamp()).sample(
            num_records=num_records
        )
        return [datetime.fromtimestamp(sample) for sample in float_samples]


DistributionT = Annotated[Distribution, Field(discriminator="distribution_type")]
DistributionT.__origin__ = Union[tuple(Distribution.__subclasses__())]  # type: ignore  # noqa: UP007 -- runtime Union needed for dynamic tuple()

DatetimeDistributionT = Annotated[DatetimeDistribution, Field(discriminator="distribution_type")]
DatetimeDistributionT.__origin__ = Union[tuple(DatetimeDistribution.__subclasses__())]  # type: ignore  # noqa: UP007 -- runtime Union needed for dynamic tuple()
