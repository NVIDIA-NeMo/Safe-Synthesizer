<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ty Fix Patterns -- Repo-Specific Reference

Detailed examples for the most common ty error patterns in this codebase.

## Layered Approach for Bulk Fixes

When facing many diagnostics across multiple modules:

- L1 (parallel, module-local): mechanical fixes within each module -- stale ignores, redundant casts, None-narrowing. If a fix requires changing another module, append to a shared `.cursor/ty-fix-notes.md` (see format below) and leave a temporary `# ty: ignore[rule]  # TODO(ty-fix)` placeholder.
- L2 (sequential, cross-module): one agent works through the notes stack -- base class signature alignment, parameter type widening, annotation corrections. Remove all `TODO(ty-fix)` placeholders.
- L3 (readonly analysis): examine for systemic improvements -- None elimination, union narrowing, descriptor typing, stub gaps. Produce findings, not code changes.

## BoundDir Chain Wrapping

`BoundDir.__getattr__` returns `Path | BoundDir`. When chaining through descriptors (e.g., `workdir.train.config`), the type checker can't prove the result is `Path`. Wrap in `Path()` since `BoundDir` implements `os.PathLike[str]`:

```python
# Before (3 errors: call-non-callable x2, invalid-return-type)
train_config = source_workdir.train.config
if train_config.exists():
    return train_config

# After (clean)
train_config = Path(source_workdir.train.config)
if train_config.exists():
    return train_config
```

Same pattern applies to `.training`, `.test`, `.adapter`, `.metadata` and any other `FileNode` accessed through a `BoundDir`.

## Generic Return via TypeVar

When a method returns different types based on a `cls` parameter, make it generic so the return type tracks the argument:

```python
# Before (_resolve_config returns broad union)
def _resolve_config(self, values, cls, **kwargs) -> NSSParameters:
    ...

# After (return type narrows to whatever cls is)
ParamT = TypeVar("ParamT", bound=NSSParameters)

def _resolve_config(self, values: NSSParameters | ParamDict | None, cls: type[ParamT], **kwargs) -> ParamT:
    ...
```

Keep `values` using the broad type -- only `cls` drives the generic. If `values` also uses `ParamT`, the type checker may fail to unify the constraints.

## Match-Branch Inlining

When a match statement assigns different callables to a variable, the post-match type is a union that breaks downstream calls:

```python
# Before (5 invalid-argument-type errors from read_parquet stubs)
match Path(url).suffix.lstrip("."):
    case "csv" | "txt":
        reader = pd.read_csv
        default_load_args = {}
    case "jsonl":
        reader = pd.read_json
        default_load_args = {"lines": True}
    case "parquet":
        reader = pd.read_parquet
        default_load_args = {}
final_load_args = {**default_load_args, **(self.load_args or {})}
return reader(url, **final_load_args)

# After (clean -- each branch calls directly)
extra: dict[str, Any] = self.load_args or {}
match Path(url).suffix.lstrip("."):
    case "csv" | "txt":
        return pd.read_csv(url, **extra)
    case "jsonl":
        return pd.read_json(url, lines=True, **extra)
    case "parquet":
        return pd.read_parquet(url, **extra)
    case _:
        raise ValueError(f"Unsupported extension")
```

## Self Return for Chainable Methods

When a method returns `self` for chaining but is annotated `-> None`, downstream calls see `None`:

```python
# Before (4 unresolved-attribute errors on chained calls)
def save_results(self, output_file) -> None:
    ...
    return self

# After
from typing import Self

def save_results(self, output_file) -> Self:
    ...
    return self
```

The `@traced` decorator preserves return types via `Callable[P, R] -> Callable[P, R]`, so `Self` propagates correctly through it.

## None Guards

ty (v0.0.18+) fully supports `assert x is not None` as a narrowing construct, along with `if x is None: raise` and `isinstance`. All three correctly narrow the type.

Three patterns, choose based on context:

```python
# 1. Assert (internal invariant -- field is always set by this point)
assert self._workdir is not None
self._workdir.ensure_directories()

# 2. Raise with message (user-facing -- explain what's missing)
if df is None:
    raise click.UsageError("--url is required for this command")

# 3. Early return (optional operation -- skip if not configured)
if self.training_bar is None:
    return
self.training_bar.update()
```

Caveat: `assert` statements are stripped when Python runs with `-O`. Use pattern 2 for cases where the guard provides actual runtime safety, not just type narrowing.

Don't blindly assert non-None. If a field can legitimately be `None` in some code paths, a guard that raises will break those paths:

```python
# Wrong -- validation_dataset is None when there's no holdout split
assert self.validation_dataset is not None
training = self._prepare(self.validation_dataset, ...)

# Right -- only assert what's actually required, leave optional fields alone
assert self.train_dataset is not None
training = self._prepare(self.train_dataset, ...)
validation = (
    self._prepare(self.validation_dataset, ...)
    if self.validation_dataset is not None
    else None
)
```

## Match Exhaustiveness

ty flags implicit `None` returns when a match statement doesn't cover all cases:

```python
# Before (implicit None return flagged)
match values:
    case BaseModel():
        return cls.model_validate(values.model_dump())
    case dict():
        return cls(**values)
    case None:
        return cls(**kwargs)

# After (explicit exhaustiveness)
match values:
    case BaseModel():
        return cls.model_validate(values.model_dump())
    case dict():
        return cls(**values)
    case None:
        return cls(**kwargs)
    case _:
        raise TypeError(f"Expected BaseModel, dict, or None; got {type(values)}")
```

## Constructor Initialization Pattern

When `__init__` accepts `T | None` but always resolves to `T` before storing, annotate the attribute as `T` to avoid cascading `unresolved-attribute` errors in every method:

```python
# Before (18 cascading errors -- every self._workdir.foo access needs a None guard)
def __init__(self, config, workdir: Workdir | None = None):
    self._workdir = workdir  # inferred as Workdir | None
    if self._workdir is None:
        self._workdir = Workdir.create_default(config)

# After (clean -- _workdir is always Workdir after __init__)
def __init__(self, config, workdir: Workdir | None = None):
    self._workdir: Workdir = workdir if workdir is not None else Workdir.create_default(config)
```

## Protocol for Interface Requirements

When a function needs `len()` support but the concrete type only inherits from an abstract base without `__len__`, define a `Protocol`:

```python
from typing import Iterator, Protocol

class _SizedSampler(Protocol):
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator: ...

class _EntitySampler(Sampler):
    def __init__(self, entity_sampler: _SizedSampler, ...):
        ...
        total = len(entity_sampler)  # clean -- _SizedSampler guarantees __len__
```

This avoids `ty: ignore` on `len()` calls when the declared parameter type (`Sampler`) lacks `__len__`.

## getattr for call-top-callable

When an attribute's type is a union that includes non-callable types, ty flags direct calls with `call-top-callable`. Use `getattr` + `callable` to narrow:

```python
# Before (call-top-callable -- self.optimizer could be Optimizer | None)
self.optimizer.train()

# After
optimizer_train = getattr(self.optimizer, "train", None)
if optimizer_train is not None and callable(optimizer_train):
    optimizer_train()
```

## isinstance Narrowing for Union Variants

When a method exists on only one arm of a non-None union, narrow with `isinstance` before calling:

```python
# Before (unresolved-attribute -- .step() only exists on RDPAccountant)
if not self.accountant.use_prv:
    self.accountant.accountant.step(noise_multiplier=nm, sample_rate=sr)

# After
if not self.accountant.use_prv:
    assert isinstance(self.accountant.accountant, RDPAccountant)
    self.accountant.accountant.step(noise_multiplier=nm, sample_rate=sr)
```

Boolean guards (`if not use_prv:`) don't narrow the union -- ty needs an explicit type check.

## Base Class Override Matching

`invalid-method-override` means the subclass signature is incompatible with the base. Common mismatches:

- Missing parameters the base class defines (even with defaults)
- Renamed parameters (`examples` vs `features`)
- Narrower parameter types (violates Liskov -- parameters should be same or wider)

```python
# Base class
class DataCollatorMixin:
    def __call__(self, features, return_tensors: str | None = None): ...

# Before (invalid-method-override -- missing return_tensors, renamed features)
class PrivateCollator(DataCollatorMixin):
    def __call__(self, examples: list[dict]) -> dict: ...

# After (matches base signature)
class PrivateCollator(DataCollatorMixin):
    def __call__(self, features, return_tensors: str | None = None) -> dict:
        batch = super().__call__(features, return_tensors=return_tensors)
        ...
```

## Removing Suppressions Safely

Removing a `type: ignore` can unmask errors the suppression was hiding beyond the one ty reported as unused. Always verify each removal individually:

```python
# ty reports this as unused:
self.train.cache.path.mkdir(...)  # type: ignore[union-attr]

# But removing it reveals a DIFFERENT error:
# error[unresolved-attribute]: `Path` has no attribute `path`
# because self.train.cache goes through BoundDir.__getattr__ → Path | BoundDir,
# and .path only exists on BoundDir, not Path.
```

Run `ty check <file>` after each batch of removals, not just at the end.

## Platform-Specific Suppression

Some imports only resolve on specific platforms. Keep the suppression even if ty doesn't flag it on your current OS:

```python
from unsloth import FastLanguageModel  # ty: ignore[unresolved-import]
```

This shows as an `unused-type-ignore-comment` warning on Linux (where unsloth is installed) but is needed on macOS. Leave it.

## RootModel Constructor

ty doesn't fully understand Pydantic's `RootModel.__init__`. Pass the `root=` keyword explicitly:

```python
# Before (missing-argument + invalid-argument-type)
LockfileDiff(changes)

# After
LockfileDiff(root=changes)
```

## Working-Notes Stack Format

Format for `.cursor/ty-fix-notes.md` used during L1/L2 layered fixes:

```markdown
# ty Fix Working Notes

## Stack

- [ ] (training/backend.py) widen `trainer_type` annotation to include `partial`
- [ ] (evaluation/component.py) align `from_components` signatures across subclasses
```
