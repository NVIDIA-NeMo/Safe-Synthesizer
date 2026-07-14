# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ...data_processing.records.value_path import ValuePath


@dataclass(eq=True, frozen=True, order=True)
class NERPrediction:
    text: str
    start: int
    end: int
    label: str
    source: str
    score: Optional[float]
    field: Optional[str] = None
    value_path: Optional[ValuePath] = None
    substring_match: Optional[bool] = None

    @property
    def as_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, source: dict):
        return cls(**source)
