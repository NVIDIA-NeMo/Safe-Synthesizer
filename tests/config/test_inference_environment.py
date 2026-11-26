# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.config.inference_environment import InferenceEnvironment, InvalidRequestSettingsError


@pytest.mark.unit
class TestAllowLocal:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            (None, True),
            ("", True),
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            (" False ", False),
            ("0", False),
            ("no", False),
            ("off", False),
        ],
    )
    def test_parses_boolean_values(self, raw: str | None, expected: bool) -> None:
        environ = {} if raw is None else {"NSS_INFERENCE_ALLOW_LOCAL": raw}

        assert InferenceEnvironment.from_environ(environ).allow_local is expected

    @pytest.mark.parametrize("raw", ["disabled", "2", "nope"])
    def test_rejects_other_values(self, raw: str) -> None:
        with pytest.raises(InvalidRequestSettingsError, match=f"NSS_INFERENCE_ALLOW_LOCAL must be true or false, got '{raw}'"):
            InferenceEnvironment.from_environ({"NSS_INFERENCE_ALLOW_LOCAL": raw})
