<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# ty Fix Patterns -- Repo-Specific Reference

Detailed examples for the most common ty error patterns in this codebase.

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

When fixing diagnostics across many modules, create `.cursor/ty-fix-notes.md`:

```markdown
# ty Fix Working Notes

Items appended by L1 agents when a fix requires touching files outside their module.
L2 agent works through them top-to-bottom.

## Stack

- [ ] (training/backend.py) widen `trainer_type` annotation to include `partial`
- [ ] (evaluation/component.py) align `from_components` signatures across subclasses
```
