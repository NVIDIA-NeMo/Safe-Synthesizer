# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Additional filters for Transforms."""

from typing import Any


def partial_mask(value: Any, left: int, mask: str, right: int, tolerance: int = 1) -> str:
    """
    Roughly implements the MSSQL partial() DDM function

    DDDM docs, see the "Custom String" entry in the tabls
    https://learn.microsoft.com/en-us/sql/relational-databases/security/dynamic-data-masking?view=sql-server-ver16#define-a-dynamic-data-mask

    If the strlen of the prefix and suffix is larger than the
    original value, the prefix and suffix sizes will be reduced
    until they are less than the strlen of the original value.

    The tolerance controls the perceived size of the original value, so
    the higher the tolerance, the prefix and suffix sizes will be reduced
    more aggressively.
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
