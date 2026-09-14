# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.pii_replacer.replacement.canonicalization import canonicalize_scalar


@pytest.mark.unit
class TestCanonicalizeScalar:
    @pytest.mark.parametrize(
        ("python_value", "library_value"),
        [
            (1, np.int64(1)),
            (True, np.bool_(True)),
            (1.5, np.float64(1.5)),
            (" Ada ", np.str_(" Ada ")),
            (date(2020, 1, 2), pd.Timestamp("2020-01-02").date()),
            (timedelta(days=2), np.timedelta64(2, "D")),
        ],
    )
    def test_equivalent_python_numpy_and_pandas_scalars_share_identity(
        self,
        python_value: object,
        library_value: object,
    ) -> None:
        assert canonicalize_scalar(python_value) == canonicalize_scalar(library_value)

    def test_type_tag_separates_boolean_integer_float_and_string(self) -> None:
        identities = {
            canonicalize_scalar(True),
            canonicalize_scalar(1),
            canonicalize_scalar(1.0),
            canonicalize_scalar("1"),
        }

        assert len(identities) == 4

    def test_string_content_is_not_cleaned(self) -> None:
        assert canonicalize_scalar(" Ada ").normalized_value == " Ada "
        assert canonicalize_scalar("Ada") != canonicalize_scalar("ada")

    @pytest.mark.parametrize("missing", [None, pd.NA, pd.NaT, np.nan])
    def test_missing_values_are_never_canonicalized(self, missing: object) -> None:
        with pytest.raises(ParameterError, match="missing values cannot be canonicalized"):
            canonicalize_scalar(missing)
