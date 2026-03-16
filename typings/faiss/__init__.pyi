# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal type stubs for faiss (SWIG-generated, no upstream stubs)."""

import numpy as np
import numpy.typing as npt

class IndexFlatL2:
    def __init__(self, d: int) -> None: ...
    def add(self, x: npt.ArrayLike) -> None: ...
    def search(self, x: npt.ArrayLike, k: int) -> tuple[np.ndarray, np.ndarray]: ...
