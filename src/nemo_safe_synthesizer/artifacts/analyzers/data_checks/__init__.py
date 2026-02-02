# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Make all checks available to import from `data_checks` module.
from nemo_safe_synthesizer.artifacts.analyzers.data_checks.base import (
    DataCheck,
)
from nemo_safe_synthesizer.artifacts.analyzers.data_checks.dataset_size import (
    DatasetSizeCheck,
)
from nemo_safe_synthesizer.artifacts.analyzers.data_checks.high_float_precision import (
    HighFloatPrecisionCheck,
)
from nemo_safe_synthesizer.artifacts.analyzers.data_checks.missing_data import (
    MissingDataCheck,
)
from nemo_safe_synthesizer.artifacts.analyzers.data_checks.sparse_data import (
    SparseDataCheck,
)
from nemo_safe_synthesizer.artifacts.analyzers.data_checks.surrounding_whitespaces import (
    SurroundingWhitespacesCheck,
)
from nemo_safe_synthesizer.artifacts.analyzers.data_checks.text_data import (
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
