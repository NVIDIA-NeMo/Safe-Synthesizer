# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, cast

from nemo_safe_synthesizer.data_processing.stats import Statistics
from nemo_safe_synthesizer.utils import _is_statistics_list


def test_is_statistics_list_validates_all_items():
    assert _is_statistics_list([Statistics()])
    assert not _is_statistics_list(Statistics())
    assert not _is_statistics_list(cast(Any, [object()]))
