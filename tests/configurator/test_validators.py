# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for DependsOnValidator getsource fallback."""

from __future__ import annotations

from typing import Annotated
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from nemo_safe_synthesizer.configurator.validators import DependsOnValidator


class MyModel(BaseModel):
    group_by: str | None = None
    order_by: Annotated[
        str | None,
        DependsOnValidator(
            depends_on="group_by",
            depends_on_func=lambda v: v is not None,
            value_func=lambda v: v is not None,
        ),
    ] = None


class TestDependsOnValidatorGetsourceFallback:
    """The error message must not crash when inspect.getsource() is unavailable."""

    def test_validation_error_when_source_available(self):
        with pytest.raises(ValidationError, match="order_by is only allowed when group_by pass condition"):
            MyModel(order_by="time")

    def test_validation_error_when_source_unavailable(self):
        with patch("nemo_safe_synthesizer.configurator.validators.inspect.getsource", side_effect=OSError):
            with pytest.raises(ValidationError, match="order_by is only allowed when group_by pass condition"):
                MyModel(order_by="time")
