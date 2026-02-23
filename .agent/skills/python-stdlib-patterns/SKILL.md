---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: python-stdlib-patterns
description: "Modern Python 3.11+ standard library patterns. Triggers on: pathlib, dataclass, functools, itertools, collections, contextlib, enum, context manager, cached_property, singledispatch."
compatibility: "Python 3.11+ (no backward-compat wrappers needed)."
---

# Python Standard Library Patterns

Repo conventions for standard library usage. Python 3.11+ -- no backward-compat wrappers needed.

## Constraints

### MUST DO

- Use `pathlib.Path` for all file path operations, not `os.path`
- Use `@dataclass` for data-carrying classes, not manual `__init__`
- Use `field(default_factory=list)` for mutable defaults in dataclasses, never `= []`
- Use `@wraps(func)` on all decorator wrappers to preserve function metadata
- Use `Enum` (preferably `str, Enum`) for fixed sets of named values
- Use `suppress()` instead of empty `try/except` blocks for expected exceptions

### MUST NOT DO

- Use `os.path.join()` -- use `Path(...) / "subdir"` operator
- Use mutable defaults in function signatures (`def f(x=[])`)
- Use bare `dict` for config with known keys -- prefer `TypedDict` or `@dataclass`
- Use magic strings where an `Enum` would be clearer
- Use `os.makedirs()` -- use `Path.mkdir(parents=True, exist_ok=True)`
