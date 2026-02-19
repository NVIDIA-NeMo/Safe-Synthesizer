---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: configurator
description: "The repo's Pydantic-to-Click configurator pattern for config models, CLI options, and parameter validation. Triggers on: configurator, pydantic_options, Parameter, parse_overrides, CLI options, config fields, DependsOnValidator, ValueValidator, AutoParam."
---

# Configurator (Pydantic -> Click CLI)

This repo uses a custom system that automatically generates Click CLI options from nested Pydantic config models. Located in `src/nemo_safe_synthesizer/configurator/`.

## How It Works

```
Pydantic model fields  -->  pydantic_options() decorator  -->  Click CLI options
     (nested)                    (flattens)                   (--data__holdout)
```

1. **`pydantic_options(model_class, field_separator="__")`** walks the Pydantic model tree recursively
2. Each leaf field becomes a `click.option()` with name `--{prefix}{sep}{field_name}`
3. At runtime, **`parse_overrides(kwargs, field_sep="__")`** converts flat Click kwargs back into a nested dict
4. The nested dict is used with `model_copy(update=overrides)` or `model_validate()`

## Adding a New Config Field

1. **Add the field** to the relevant Pydantic model (e.g., `DataParameters`, `TrainingHyperparams`):

```python
class DataParameters(Parameters):
    my_new_field: int = Field(default=10, description="Description for CLI help")
```

2. **That's it** -- `pydantic_options` will auto-generate `--data__my_new_field` (or whatever the nesting path is)

3. For nested models, add as a sub-model field:

```python
class MySubConfig(Parameters):
    threshold: float = Field(default=0.5, description="Threshold value")

class DataParameters(Parameters):
    sub: MySubConfig = Field(default_factory=MySubConfig)
    # Generates: --data__sub__threshold
```

## Parameter Types

The `Parameter[T]` class wraps config values with extra semantics:

| Type | Meaning | Usage |
|------|---------|-------|
| `Parameter[T]` | A wrapped value | Base class |
| `AutoParam[T]` | Value or `"auto"` (computed at runtime) | `AutoParam(value="auto")` |
| `UnsetParam[T]` | Field was never set | Sentinel for optional fields |
| `ValidNoneParam[T]` | `None` is a valid, intentional value | Distinct from "not set" |

`Parameter[T]` integrates with Pydantic via `__get_pydantic_core_schema__` (accepts both raw values and `Parameter` instances) and `model_serializer` (serializes to `.value`).

Common type aliases:

```python
OptionalAutoInt = AutoParam[int] | None
OptionalAutoFloat = AutoParam[float] | None
```

## Conditional Validation

### DependsOnValidator

Field is only valid when another field meets a condition:

```python
from typing import Annotated
from nemo_safe_synthesizer.configurator.validators import DependsOnValidator

class DataParameters(Parameters):
    group_by: str | None = None
    order_by: Annotated[
        str | None,
        DependsOnValidator(
            depends_on="group_by",
            depends_on_func=lambda v: v is not None,  # only valid when group_by is set
        ),
    ] = None
```

### ValueValidator

Validates the effective value of a field:

```python
from nemo_safe_synthesizer.configurator.validators import ValueValidator

class DataParameters(Parameters):
    holdout: Annotated[
        float,
        ValueValidator(value_func=lambda v: v >= 0),
    ] = 0.1
```

### AutoParamRangeValidator

Convenience for non-negative values that support `"auto"`:

```python
from nemo_safe_synthesizer.configurator.validators import AutoParamRangeValidator

class TrainingHyperparams(Parameters):
    batch_size: Annotated[AutoParam[int], AutoParamRangeValidator] = AutoParam(value="auto")
```

## Key Files

| File | Purpose |
|------|---------|
| `configurator/pydantic_click_options.py` | `pydantic_options()` decorator and `parse_overrides()` |
| `configurator/parameter.py` | `Parameter[T]`, `AutoParam`, `UnsetParam`, `ValidNoneParam` |
| `configurator/validators.py` | `DependsOnValidator`, `ValueValidator`, `AutoParamRangeValidator` |
| `config/base.py` | `NSSBaseModel`, `pydantic_model_config` |

## Conventions

1. **Always add `description`** to `Field()` -- it becomes the CLI `--help` text
2. **Use `Parameter[T]`** for config fields that support `"auto"` or unset semantics
3. **Use `DependsOnValidator`** for fields that are only valid when another field is set
4. **Use `ValueValidator`** for simple range/predicate checks
5. **Field separator is `"__"`** (`CLI_NESTED_FIELD_SEPARATOR`) -- don't use `.` unless matching existing code
6. **Max nesting depth is 2** -- `parse_overrides` raises `ValueError` for 3+ levels
