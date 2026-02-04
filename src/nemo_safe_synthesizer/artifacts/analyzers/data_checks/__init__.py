# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Make all checks available to import from `data_checks` module.
from .base import (
    DataCheck,
)
from .dataset_size import (
    DatasetSizeCheck,
)
from .high_float_precision import (
    HighFloatPrecisionCheck,
)
from .missing_data import (
    MissingDataCheck,
)
from .sparse_data import (
    SparseDataCheck,
)
from .surrounding_whitespaces import (
    SurroundingWhitespacesCheck,
)
from .text_data import (
    TextDataCheck,
)


def create_all_checks() -> list[DataCheck]:
    return [
        HighFloatPrecisionCheck(),
        MissingDataCheck(),
        SparseDataCheck(),
        SurroundingWhitespacesCheck(),
        TextDataCheck(),
        DatasetSizeCheck(),
    ]
