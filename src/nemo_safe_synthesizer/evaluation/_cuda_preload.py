# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preload NVIDIA CUDA libraries so cuVS/cuPy can resolve symbols.

Uses the official cuda.pathfinder API (NVIDIA CUDA Python) to load nvrtc and
cudart with RTLD_GLOBAL before importing cuVS/cuPy. See:
https://nvidia.github.io/cuda-python/cuda-pathfinder/latest/

Call ensure_nvidia_libraries_preloaded() before any cupy/cuvs import.
"""

from __future__ import annotations

_preload_done = False

# Lib names used by cuda.pathfinder.load_nvidia_dynamic_lib (see SUPPORTED_NVIDIA_LIBNAMES).
_PRELOAD_LIBS = ("nvrtc", "cudart")


def ensure_nvidia_libraries_preloaded() -> None:
    """Preload NVIDIA CUDA libraries so cuVS/cuPy can resolve symbols.

    Uses cuda.pathfinder.load_nvidia_dynamic_lib (official API); loads with
    RTLD_GLOBAL so subsequently loaded cupy/cuvs can find the symbols.
    Idempotent; safe to call multiple times. No-op if cuda.pathfinder or
    the libraries are not present (e.g. CPU-only install).
    """
    global _preload_done
    if _preload_done:
        return
    try:
        from cuda.pathfinder import (  # noqa: PLC0415
            DynamicLibNotFoundError,
            load_nvidia_dynamic_lib,
        )
    except ImportError:
        _preload_done = True
        return
    for libname in _PRELOAD_LIBS:
        try:
            load_nvidia_dynamic_lib(libname)
        except DynamicLibNotFoundError:
            pass
    _preload_done = True
