---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: python-stdlib-patterns
description: "Modern Python 3.11+ standard library patterns. Triggers on: pathlib, dataclass, functools, itertools, collections, contextlib, enum, context manager, cached_property, singledispatch."
compatibility: "Python 3.11+ (no backward-compat wrappers needed)."
allowed-tools: "Read Write"
depends-on: []
related-skills: [python-typing, python-testing-patterns, python-observability]
---

# Python Standard Library Patterns

Idiomatic patterns for the Python 3.11+ standard library. Prefer these over third-party libraries or manual implementations where possible.

## When to Use This Skill

- Working with file paths and file I/O
- Defining data-carrying classes (dataclasses over manual `__init__`)
- Caching function results or expensive computations
- Iterating, grouping, or combining sequences
- Managing resources with context managers
- Defining enumerated constants

## Pathlib for File Operations

Always use `pathlib.Path` instead of `os.path` for file operations.

```python
from pathlib import Path

# Path construction
project_root = Path(__file__).parent.parent
config_file = project_root / "config" / "settings.toml"
data_dir = Path.home() / "data"

# Reading and writing
content = config_file.read_text(encoding="utf-8")
config_file.write_text("new content", encoding="utf-8")
binary = config_file.read_bytes()

# Traversal and globbing
python_files = list(project_root.rglob("*.py"))
configs = list(project_root.glob("config/*.toml"))

# Path properties
config_file.exists()
config_file.is_file()
config_file.is_dir()
config_file.suffix      # ".toml"
config_file.stem        # "settings"
config_file.name        # "settings.toml"
config_file.parent      # Path("config")

# Directory creation
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

# Temporary files (combine with tempfile)
from tempfile import TemporaryDirectory

with TemporaryDirectory() as tmpdir:
    temp_path = Path(tmpdir) / "output.txt"
    temp_path.write_text("data")
```

## Dataclasses

Use `@dataclass` for data-carrying classes. Avoid writing manual `__init__` methods.

```python
from dataclasses import dataclass, field, asdict, replace
from typing import ClassVar

# Basic dataclass (auto-generates __init__, __repr__, __eq__)
@dataclass
class User:
    id: int
    name: str
    email: str
    active: bool = True

# Post-init validation
@dataclass
class Product:
    name: str
    price: float
    discount: float = 0.0

    def __post_init__(self) -> None:
        if self.discount > 1.0:
            raise ValueError("Discount must be <= 1.0")

    @property
    def final_price(self) -> float:
        return self.price * (1 - self.discount)

# Mutable default fields -- use field(default_factory=...)
@dataclass
class ShoppingCart:
    user_id: int
    items: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

# Frozen (immutable) dataclass
@dataclass(frozen=True)
class Point:
    x: float
    y: float

# Ordered dataclass (enables <, >, <=, >= comparisons)
@dataclass(order=True)
class Priority:
    level: int
    name: str = field(compare=False)  # excluded from comparison

# Class variables (not included in __init__)
@dataclass
class Config:
    API_VERSION: ClassVar[str] = "v1"
    timeout: int = 30

# Conversion utilities
user = User(1, "Alice", "alice@example.com")
user_dict = asdict(user)                    # -> dict
updated = replace(user, name="Alice Smith") # -> new User with one field changed
```

## Functools

Caching, partial application, and dispatch patterns.

```python
from functools import cache, lru_cache, cached_property, partial, wraps, reduce, singledispatch

# Unlimited cache (Python 3.9+)
@cache
def fibonacci(n: int) -> int:
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# LRU cache with size limit
@lru_cache(maxsize=128)
def fetch_user(user_id: int) -> dict[str, str]:
    return {"id": str(user_id), "name": "User"}

# Cached property -- computed once, then stored
class DataProcessor:
    def __init__(self, data: list[int]) -> None:
        self._data = data

    @cached_property
    def mean(self) -> float:
        return sum(self._data) / len(self._data)

# Partial application
from operator import mul

double = partial(mul, 2)
triple = partial(mul, 3)

# Decorator that preserves function metadata
from typing import ParamSpec, TypeVar
from collections.abc import Callable

P = ParamSpec("P")
R = TypeVar("R")

def timing(func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        import time
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    return wrapper

# Single dispatch -- polymorphism without classes
@singledispatch
def process(arg: object) -> str:
    return f"Unknown type: {type(arg)}"

@process.register
def _(arg: int) -> str:
    return f"Integer: {arg * 2}"

@process.register
def _(arg: str) -> str:
    return f"String: {arg.upper()}"

@process.register(list)
def _(arg: list) -> str:
    return f"List with {len(arg)} items"

# Reduce for aggregation
from operator import add

total = reduce(add, [1, 2, 3, 4, 5])  # 15
```

## Itertools

Memory-efficient iteration, grouping, and combination patterns.

```python
from itertools import (
    chain, islice, groupby, accumulate,
    combinations, permutations, product,
    zip_longest, filterfalse, count, cycle, repeat
)

# Chain multiple iterables into one
combined = list(chain([1, 2], [3, 4], [5, 6]))  # [1, 2, 3, 4, 5, 6]

# Slice an iterator (memory-efficient)
first_10 = list(islice(range(1_000_000), 10))

# Group consecutive elements by key (data must be sorted by key first)
data = [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
grouped = {k: list(v) for k, v in groupby(data, key=lambda x: x[0])}
# {"A": [("A", 1), ("A", 2)], "B": [("B", 1), ("B", 2)]}

# Running totals
cumsum = list(accumulate([1, 2, 3, 4, 5]))  # [1, 3, 6, 10, 15]

# Combinations and permutations
combos = list(combinations([1, 2, 3], 2))     # [(1,2), (1,3), (2,3)]
perms = list(permutations([1, 2, 3], 2))      # [(1,2), (1,3), (2,1), ...]

# Cartesian product
pairs = list(product([1, 2], ["a", "b"]))     # [(1,'a'), (1,'b'), (2,'a'), (2,'b')]

# Zip iterables of different lengths
paired = list(zip_longest([1, 2], ["a", "b", "c"], fillvalue=0))
# [(1, 'a'), (2, 'b'), (0, 'c')]

# Filter by negating a predicate
odds = list(filterfalse(lambda x: x % 2 == 0, range(10)))  # [1, 3, 5, 7, 9]

# Infinite iterators (use with islice)
counter = count(start=1, step=2)   # 1, 3, 5, 7, ...
cycled = cycle(["a", "b", "c"])    # a, b, c, a, b, c, ...
```

## Collections

Specialized container types.

```python
from collections import defaultdict, Counter, deque, ChainMap

# defaultdict -- auto-creates missing keys
word_index: defaultdict[str, list[int]] = defaultdict(list)
for i, word in enumerate(["hello", "world", "hello"]):
    word_index[word].append(i)
# {"hello": [0, 2], "world": [1]}

# Counter -- counting and frequency
word_counts = Counter(["apple", "banana", "apple", "cherry", "banana", "apple"])
word_counts.most_common(2)  # [("apple", 3), ("banana", 2)]

# Counter arithmetic
c1 = Counter(a=3, b=1)
c2 = Counter(a=1, b=2)
print(c1 + c2)  # Counter({"a": 4, "b": 3})

# deque -- efficient append/pop from both ends
from collections import deque

queue: deque[str] = deque()
queue.append("first")
queue.appendleft("priority")
item = queue.popleft()  # "priority"

# Ring buffer with maxlen
recent: deque[int] = deque(maxlen=3)
for i in range(5):
    recent.append(i)  # Only keeps last 3: deque([2, 3, 4])

# ChainMap -- layered lookups (first match wins)
defaults = {"color": "red", "user": "guest"}
environment = {"user": "admin"}
combined = ChainMap(environment, defaults)
combined["user"]   # "admin" (from environment)
combined["color"]  # "red" (from defaults)
```

## Contextlib

Context manager utilities.

```python
from contextlib import contextmanager, suppress, ExitStack
from collections.abc import Iterator

# Custom context manager via decorator
@contextmanager
def managed_resource(resource_id: str) -> Iterator[Resource]:
    resource = acquire_resource(resource_id)
    try:
        yield resource
    finally:
        release_resource(resource)

# Suppress specific exceptions
from pathlib import Path
with suppress(FileNotFoundError):
    Path("nonexistent.txt").unlink()

# ExitStack -- manage a dynamic number of context managers
def process_files(filenames: list[str]) -> None:
    with ExitStack() as stack:
        files = [stack.enter_context(open(fn)) for fn in filenames]
        for f in files:
            process(f.read())
        # All files auto-closed on exit
```

## Enum

Use `Enum` for fixed sets of named constants instead of magic strings or ints.

```python
from enum import Enum, auto, IntEnum, Flag

# String enum (common for config values, API modes)
class Status(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

# Auto-numbered enum
class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()

# IntEnum for numeric values that need int comparisons
class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

# Flag for combinable bit flags
class Permission(Flag):
    READ = auto()
    WRITE = auto()
    EXECUTE = auto()

user_perms = Permission.READ | Permission.WRITE
if Permission.READ in user_perms:
    print("Can read")
```

## Constraints

### MUST DO

- Use `pathlib.Path` for all file path operations, not `os.path`
- Use `@dataclass` for data-carrying classes, not manual `__init__`
- Use `field(default_factory=list)` for mutable defaults in dataclasses, never `= []`
- Use `@wraps(func)` on all decorator wrappers to preserve function metadata
- Use `Enum` (preferably `str, Enum`) for fixed sets of named values
- Use `suppress()` instead of empty `try/except` blocks for expected exceptions
- Use `ExitStack` when managing a variable number of context managers

### MUST NOT DO

- Use `os.path.join()` -- use `Path(...) / "subdir"` operator instead
- Use mutable defaults in function signatures (`def f(x=[])`)
- Use bare `dict` for configuration objects with known keys -- prefer `TypedDict` or `@dataclass`
- Use magic strings where an `Enum` would be clearer
- Use `os.makedirs()` -- use `Path.mkdir(parents=True, exist_ok=True)`

---

## See Also

**Related Skills:**

- `python-typing` - Type hints, generics, protocols, ty checker
- `python-testing-patterns` - pytest patterns, fixtures, mocking, property-based testing
- `python-observability` - Structured logging, traced decorators, log categories
