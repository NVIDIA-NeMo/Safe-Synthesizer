# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Additional filters for Transforms."""

from __future__ import annotations

from typing import Any


def partial_mask(value: Any, left: int, mask: str, right: int, tolerance: int = 1) -> str:
    """Mask the middle of a string, keeping ``left`` chars at the start and ``right`` at the end (MSSQL partial()-style).

    If ``left + right`` would exceed the effective length (``len(value) - tolerance``),
    prefix/suffix sizes are reduced until they fit. Higher ``tolerance`` shrinks the
    effective length, so more of the value is replaced by ``mask``. Roughly matches
    MSSQL partial() DDM; see Microsoft docs on dynamic data masking (Custom String).
    https://learn.microsoft.com/en-us/sql/relational-databases/security/dynamic-data-masking?view=sql-server-ver16#define-a-dynamic-data-mask

    Args:
        value: String to mask (stringified if not str).
        left: Number of leading characters to keep.
        mask: String to use for the middle (replaces the rest).
        right: Number of trailing characters to keep.
        tolerance: Subtracted from value length to get effective length; higher = more masking.

    Returns:
        ``prefix + mask + suffix``, or just ``mask`` if effective length <= 0 or both left and right are < 1.

    Raises:
        ValueError: If ``mask`` is empty or ``tolerance`` < 0.
    """
    value = str(value)
    if not mask:
        raise ValueError("When using partial_mask, the mask value must be greater than len of 0.")

    if tolerance < 0:
        raise ValueError("The partial_mask tolerance must be greater than or equal to 0.")

    # adjust the perceived value len by
    # the tolerance, such that a higher
    # tolerance is more aggresive on masking
    #
    # if the updated len is less than zero
    # then we have to mask the whole value
    val_len = len(value) - tolerance
    if val_len <= 0:
        return mask

    if left < 1 and right < 1:
        return mask

    prefix = ""
    suffix = ""

    while left + right >= val_len:
        if left > 0:
            left -= 1

        if left + right < val_len:
            break

        if right > 0:
            right -= 1

    if left:
        prefix = value[:left]

    if right:
        suffix = value[-right:]

    return f"{prefix}{mask}{suffix}"
