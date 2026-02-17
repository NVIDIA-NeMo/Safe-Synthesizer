---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: python-typing
description: "Python type hints and type safety patterns. Triggers on: type hints, typing, TypeVar, Generic, Protocol, ty, type annotation, overload, TypedDict, type check."
compatibility: "Python 3.11+ (uses union syntax X | Y, Self, TypeVarTuple)."
allowed-tools: "Read Write"
depends-on: []
related-skills: [python-testing-patterns, python-observability, python-stdlib-patterns]
---

# Python Typing Patterns

Modern type hints for safe, documented Python code.

## Basic Annotations

```python
# Variables
name: str = "Alice"
count: int = 42
items: list[str] = ["a", "b"]
mapping: dict[str, int] = {"key": 1}

# Function signatures
def greet(name: str, times: int = 1) -> str:
    return f"Hello, {name}!" * times

# None handling
def find(id: int) -> str | None:
    return db.get(id)  # May return None
```

## Collections

```python
from collections.abc import Sequence, Mapping, Iterable

# Use collection ABCs for flexibility
def process(items: Sequence[str]) -> list[str]:
    """Accepts list, tuple, or any sequence."""
    return [item.upper() for item in items]

def lookup(data: Mapping[str, int], key: str) -> int:
    """Accepts dict or any mapping."""
    return data.get(key, 0)

# Nested types
Matrix = list[list[float]]
Config = dict[str, str | int | bool]
```

## Optional and Union

```python
# Modern syntax (3.10+)
def find(id: int) -> User | None:
    pass

def parse(value: str | int | float) -> str:
    pass

# With default None
def fetch(url: str, timeout: float | None = None) -> bytes:
    pass
```

## TypedDict

```python
from typing import TypedDict, Required, NotRequired

class UserDict(TypedDict):
    id: int
    name: str
    email: str | None

class ConfigDict(TypedDict, total=False):  # All optional
    debug: bool
    log_level: str

class APIResponse(TypedDict):
    data: Required[list[dict]]
    error: NotRequired[str]

def process_user(user: UserDict) -> str:
    return user["name"]  # Type-safe key access
```

## Callable

```python
from collections.abc import Callable

# Function type
Handler = Callable[[str, int], bool]

def register(callback: Callable[[str], None]) -> None:
    pass

# With keyword args (use Protocol instead)
from typing import Protocol

class Processor(Protocol):
    def __call__(self, data: str, *, verbose: bool = False) -> int:
        ...
```

## Generics

```python
from typing import TypeVar

T = TypeVar("T")

def first(items: list[T]) -> T | None:
    return items[0] if items else None

# Bounded TypeVar
from typing import SupportsFloat

N = TypeVar("N", bound=SupportsFloat)

def average(values: list[N]) -> float:
    return sum(float(v) for v in values) / len(values)
```

## Protocol (Structural Typing)

```python
from typing import Protocol

class Readable(Protocol):
    def read(self, n: int = -1) -> bytes:
        ...

def load(source: Readable) -> dict:
    """Accepts any object with read() method."""
    data = source.read()
    return json.loads(data)

# Works with file, BytesIO, custom classes
load(open("data.json", "rb"))
load(io.BytesIO(b"{}"))
```

## Type Guards

```python
from typing import TypeGuard

def is_string_list(val: list[object]) -> TypeGuard[list[str]]:
    return all(isinstance(x, str) for x in val)

def process(items: list[object]) -> None:
    if is_string_list(items):
        # items is now list[str]
        print(", ".join(items))
```

## Literal and Final

```python
from typing import Literal, Final

Mode = Literal["read", "write", "append"]

def open_file(path: str, mode: Mode) -> None:
    pass

# Constants
MAX_SIZE: Final = 1024
API_VERSION: Final[str] = "v2"
```

## Quick Reference

| Type | Use Case |
|------|----------|
| `X \| None` | Optional value |
| `list[T]` | Homogeneous list |
| `dict[K, V]` | Dictionary |
| `Callable[[Args], Ret]` | Function type |
| `TypeVar("T")` | Generic parameter |
| `Protocol` | Structural typing |
| `TypedDict` | Dict with fixed keys |
| `Literal["a", "b"]` | Specific values only |
| `Final` | Cannot be reassigned |

## Constraints

### MUST DO

- Type hints on all function signatures and class attributes
- `X | Y` union syntax, not `Optional[X]` or `Union[X, Y]`
- `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- `Self` type for fluent interfaces and factory classmethods (3.11+)
- Collection ABCs for function parameters (`Sequence`, `Mapping`, `Iterable`)
- Concrete types for return values (`list`, `dict`)
- `Protocol` for structural typing over inheritance
- `TypedDict` for dict-like objects with known keys
- Use ty for type checking (`ty check` or `bash tools/lint/run-ty-check.sh`)

### MUST NOT DO

- Use `Optional[X]` -- use `X | None` instead
- Use `Union[X, Y]` -- use `X | Y` instead
- Import from `typing` what's available as a builtin (`List`, `Dict`, `Tuple`, `Set`)
- Use mypy -- this repo uses ty (Astral's type checker)
- Skip type annotations on public APIs
- Use `Any` as a lazy escape hatch -- prefer `object`, generics, or `Protocol`
- Ignore ty errors -- fix them or add targeted `# ty: ignore[code]` with justification

## Common Pitfalls

1. **Mutable default arguments**: `def f(items: list[str] = [])` mutates shared state
   - Fix: Use `None` sentinel: `def f(items: list[str] | None = None)`

2. **Forgetting covariance in return types**: Returning `list[Dog]` where `list[Animal]` is expected
   - Fix: Use `Sequence[Animal]` in signatures that need covariance

3. **Overusing `Any`**: Silences the type checker entirely
   - Fix: Use `object` for truly unknown types, or define a `Protocol`

4. **Mixing up `TypeVar` bounds and constraints**: `T = TypeVar("T", int, str)` means "exactly int or str", not "subclass of"
   - Fix: Use `bound=` for upper bounds: `T = TypeVar("T", bound=SupportsFloat)`

5. **Not narrowing `None`**: Accessing `.attr` on `X | None` without a guard
   - Fix: Use `if x is not None:` or `assert x is not None` before access

6. **Using `dict` when `TypedDict` is appropriate**: Losing type safety on known key structures
   - Fix: Define a `TypedDict` for dicts with a fixed schema

## Type Checker Commands

This repo uses [ty](https://github.com/astral-sh/ty) (Astral's type checker) for static type checking.

```bash
# Run ty on the full project (respects [tool.ty.src] in pyproject.toml)
ty check

# Run ty on specific files
ty check src/nemo_safe_synthesizer/config/

# Run ty on changed files only (repo script, compares against main)
bash tools/lint/run-ty-check.sh

# Run ty on changed files against a specific base
bash tools/lint/run-ty-check.sh <merge-base-sha>

# If ty is not installed locally, use uvx
uvx ty check
```

ty is configured in `pyproject.toml` under `[tool.ty.src]` (exclusions, etc).

## Additional Resources

- `./references/generics-advanced.md` - TypeVar, ParamSpec, TypeVarTuple
- `./references/protocols-patterns.md` - Structural typing, runtime protocols
- `./references/type-narrowing.md` - Guards, isinstance, assert
- `./references/ty-config.md` - ty configuration and usage
- `./references/runtime-validation.md` - Pydantic v2, typeguard, beartype
- `./references/overloads.md` - @overload decorator patterns

## Repo Scripts

- `tools/lint/run-ty-check.sh` - Run ty on changed files (diff against main or a given SHA)
- `tools/lint/filter_ty_exclusions.py` - Filters files against `[tool.ty.src].exclude`

---

## See Also

This is a **foundation skill** with no prerequisites.

**Related Skills:**

- `python-testing-patterns` - Type-safe fixtures, mocking, property-based testing
- `python-observability` - Structured logging, traced decorators, log categories
- `python-stdlib-patterns` - pathlib, dataclasses, functools, itertools, enum patterns
