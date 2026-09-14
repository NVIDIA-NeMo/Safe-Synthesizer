# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile replacement-plan dependencies into a stable execution order."""

from __future__ import annotations

import heapq

from ...config.replace_pii import PiiColumnPlan, PiiReplacementPlan
from ...errors import ParameterError

__all__ = ["compile_plan"]


def compile_plan(plan: PiiReplacementPlan) -> tuple[PiiColumnPlan, ...]:
    """Return plan targets in stable topological order.

    Independent targets retain their declaration order. Dependencies that are
    read-only dataframe columns do not create graph nodes.
    """
    specs = plan.columns_to_replace
    by_name = {spec.column_name: spec for spec in specs}
    position = {spec.column_name: index for index, spec in enumerate(specs)}
    outgoing: dict[str, list[str]] = {column: [] for column in by_name}
    indegree = dict.fromkeys(by_name, 0)

    for spec in specs:
        for dependency in spec.depends_on:
            source = dependency.column_name
            if source not in by_name:
                continue
            outgoing[source].append(spec.column_name)
            indegree[spec.column_name] += 1

    ready = [(position[column], column) for column, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[PiiColumnPlan] = []
    while ready:
        _, source = heapq.heappop(ready)
        ordered.append(by_name[source])
        for target in sorted(outgoing[source], key=position.__getitem__):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, (position[target], target))

    if len(ordered) != len(specs):
        cycle_columns = sorted(column for column, degree in indegree.items() if degree > 0)
        raise ParameterError(
            "replacement dependencies contain a cycle involving: "
            + ", ".join(repr(column) for column in cycle_columns)
        )
    return tuple(ordered)
