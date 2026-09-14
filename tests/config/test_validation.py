# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from pydantic import BaseModel, ValidationError

from nemo_safe_synthesizer.config.validation import format_pydantic_validation_error


class _ItemConfig(BaseModel):
    count: int


class _ContainerConfig(BaseModel):
    items: list[_ItemConfig]


@pytest.mark.unit
def test_format_pydantic_validation_error_uses_nested_field_paths() -> None:
    with pytest.raises(ValidationError) as error:
        _ContainerConfig.model_validate({"items": [{"count": "invalid"}]})

    details = format_pydantic_validation_error(error.value)

    assert details.startswith("items.0.count: Input should be a valid integer")
