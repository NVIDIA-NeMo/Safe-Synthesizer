# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FK topological processing order for multi-table replacement."""

from __future__ import annotations

from collections import defaultdict, deque

from ...observability import get_logger
from .schema import DatabaseSchema

logger = get_logger(__name__)

__all__ = ["processing_order"]


def processing_order(schema: DatabaseSchema) -> list[str]:
    """Return table names in FK topological order.

    Referenced / parent tables come before referencing / child tables. Schema
    listing order is the tiebreaker among independent tables. If the FK graph
    has a cycle, fall back to schema listing order for the cyclic component and
    warn.
    """
    listing = schema.table_order_names()
    listing_index = {name: i for i, name in enumerate(listing)}

    # Edge: parent -> child (process parent first). Also track indegree on children.
    children: dict[str, set[str]] = defaultdict(set)
    indegree: dict[str, int] = {name: 0 for name in listing}

    for child, _child_cols, parent, _parent_cols in schema.fk_links():
        if parent not in indegree or child not in indegree:
            continue
        if child == parent:
            continue
        if child not in children[parent]:
            children[parent].add(child)
            indegree[child] += 1

    ready = sorted([n for n, d in indegree.items() if d == 0], key=lambda n: listing_index[n])
    queue: deque[str] = deque(ready)
    ordered: list[str] = []

    while queue:
        node = queue.popleft()
        ordered.append(node)
        for child in sorted(children[node], key=lambda n: listing_index[n]):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    if len(ordered) < len(listing):
        remaining = [n for n in listing if n not in set(ordered)]
        logger.user.warning(
            "[PII Replacement] Schema FK graph has a cycle involving "
            f"{', '.join(remaining)}; falling back to schema listing order for those tables."
        )
        ordered.extend(remaining)

    return ordered
