---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: python-typing
description: "Python type hints and type safety patterns. Triggers on: type hints, typing, TypeVar, Generic, Protocol, ty, type annotation, overload, TypedDict, type check."
compatibility: "Python 3.11+ (uses union syntax X | Y, Self, TypeVarTuple)."
---

# Python Typing Patterns

## Type Checker

This repo uses [ty](https://github.com/astral-sh/ty) (Astral's type checker), **not** mypy.

```bash
# Full project (respects [tool.ty.src] in pyproject.toml)
ty check

# Specific path
ty check src/nemo_safe_synthesizer/config/

# Changed files only (diff against main)
bash tools/lint/run-ty-check.sh

# Changed files against a specific base
bash tools/lint/run-ty-check.sh <merge-base-sha>

# If ty is not installed locally
uvx ty check
```

ty is configured in `pyproject.toml` under `[tool.ty.src]`. See `references/ty-config.md` for details.

## Repo Scripts

- `tools/lint/run-ty-check.sh` -- run ty on changed files (diff against main or a given SHA)
- `tools/lint/filter_ty_exclusions.py` -- filters files against `[tool.ty.src].exclude`

## Constraints

### MUST DO

- Type hints on all function signatures and class attributes
- `X | Y` union syntax, not `Optional[X]` or `Union[X, Y]`
- `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- `Self` type for fluent interfaces and factory classmethods (3.11+)
- Collection ABCs for parameters (`Sequence`, `Mapping`, `Iterable`); concrete types for return values
- `Protocol` for structural typing over inheritance
- `TypedDict` for dict-like objects with known keys
- Run `ty check` or `bash tools/lint/run-ty-check.sh` before committing

### MUST NOT DO

- Use `Optional[X]` -- use `X | None`
- Use `Union[X, Y]` -- use `X | Y`
- Import from `typing` what's available as a builtin (`List`, `Dict`, `Tuple`, `Set`)
- Use mypy -- this repo uses ty
- Use `Any` as a lazy escape hatch -- prefer `object`, generics, or `Protocol`
- Ignore ty errors -- fix them or add targeted `# ty: ignore[code]` with justification
- Skip type annotations on public APIs

## Common Pitfalls

1. **Mutable default arguments**: `def f(items: list[str] = [])` -- use `None` sentinel instead
2. **Overusing `Any`**: silences the type checker -- use `object` or define a `Protocol`
3. **Not narrowing `None`**: access `.attr` on `X | None` without a guard -- use `if x is not None:`
4. **Mixing `TypeVar` bounds vs constraints**: `T = TypeVar("T", int, str)` means "exactly int or str"; use `bound=` for upper bounds
