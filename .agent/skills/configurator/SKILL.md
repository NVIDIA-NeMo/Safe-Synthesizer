---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: configurator
description: "Pydantic models and the Pydantic-to-Click configurator for config, CLI, and SDK validation. Triggers on: configurator, pydantic_options, Parameter, parse_overrides, CLI options, config fields, DependsOnValidator, ValueValidator, AutoParam, Pydantic, BaseModel, NSSBaseModel, BaseSettings, field_validator, model_validator, ConfigDict, TypeAdapter, pydantic-settings, validation."
---

# Configurator and Pydantic Patterns

Pydantic v2 is the backbone of config, settings, CLI, and SDK validation. This repo has a custom system that automatically generates Click CLI options from nested Pydantic config models. Located in `src/nemo_safe_synthesizer/configurator/`.

## How It Works

```
Pydantic model fields  -->  pydantic_options() decorator  -->  Click CLI options
     (nested)                    (flattens)                   (--data__holdout)
```

1. `pydantic_options(model_class, field_separator="__")` walks the Pydantic model tree recursively
2. Each leaf field becomes a `click.option()` with name `--{prefix}{sep}{field_name}`
3. At runtime, `parse_overrides(kwargs, field_sep="__")` converts flat Click kwargs back into a nested dict
4. The nested dict is used with `model_copy(update=overrides)` or `model_validate()`

## Base Models

### NSSBaseModel (config and result models)

All config/result models that don't use `Parameter` fields inherit from `NSSBaseModel`:

```python
from nemo_safe_synthesizer.config.base import NSSBaseModel

class MyConfig(NSSBaseModel):
    name: str
    count: int = 10
```

Shared `model_config` (`config/base.py`):
- `arbitrary_types_allowed=True`
- `validation_error_cause=True`
- `from_attributes=True`
- `validate_default=True`
- `protected_namespaces=()`

### BaseSettings (env/CLI settings)

Settings classes use `pydantic-settings`, not `NSSBaseModel`:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AliasChoices, Field

class MySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",
        extra="ignore",
        env_file=".env",
    )

    log_level: str = Field(
        default="INFO",
        validation_alias=AliasChoices("log_level", "NSS_LOG_LEVEL"),
    )
```

Conventions:
- Use `AliasChoices("field_name", "NSS_ENV_VAR")` for env var mapping
- Nested settings via `Field(default_factory=SubSettings)`
- No global `env_prefix`; each field declares its own aliases
- Filter `None` values from CLI kwargs before constructing: `{k: v for k, v in kwargs.items() if v is not None}`

## Adding a New Config Field

1. Add the field to the relevant Pydantic model (e.g., `DataParameters`, `TrainingHyperparams`):

```python
class DataParameters(Parameters):
    my_new_field: int = Field(default=10, description="Description for CLI help")
```

2. That's it -- `pydantic_options` will auto-generate `--data__my_new_field` (or whatever the nesting path is)

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

## Validators

### field_validator

```python
from pydantic import field_validator

class MyModel(NSSBaseModel):
    mode: str

    @field_validator("mode", mode="before")
    @classmethod
    def coerce_mode(cls, v: str) -> str:
        return v.lower()
```

- `mode="before"` for coercion (str -> enum, normalization)
- `mode="after"` for value checks and clamping

### model_validator

```python
from pydantic import model_validator

class MyModel(NSSBaseModel):
    start: int
    end: int

    @model_validator(mode="after")
    def check_range(self) -> "MyModel":
        if self.end <= self.start:
            raise ValueError("end must be > start")
        return self
```

- `mode="before"` + `@classmethod` for preprocessing raw dicts
- `mode="after"` as instance method for cross-field checks

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
            depends_on_func=lambda v: v is not None,
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

## ConfigDict Variants

| Use Case | Config |
|----------|--------|
| Standard config models | `pydantic_model_config` from `config/base.py` |
| Forbid extra fields | `ConfigDict(extra="forbid")` |
| YAML key alias (`type_` -> `type`) | `ConfigDict(alias_generator=...)` |
| Hide sensitive input in errors | `ConfigDict(hide_input_in_errors=True)` |
| Settings from env | `SettingsConfigDict(env_nested_delimiter="__", env_file=".env")` |

## TypeAdapter (validation without a model)

```python
from pydantic import TypeAdapter, ConfigDict

ta = TypeAdapter(
    dict[str, str | None],
    config=ConfigDict(hide_input_in_errors=True),
)
result = ta.validate_python(raw_data)
```

## Key Files

| File | Purpose |
|------|---------|
| `configurator/pydantic_click_options.py` | `pydantic_options()` decorator and `parse_overrides()` |
| `configurator/parameter.py` | `Parameter[T]`, `AutoParam`, `UnsetParam`, `ValidNoneParam` |
| `configurator/validators.py` | `DependsOnValidator`, `ValueValidator`, `AutoParamRangeValidator` |
| `config/base.py` | `NSSBaseModel`, `pydantic_model_config` |

## Conventions

1. Inherit from `NSSBaseModel` for config/result models (not bare `BaseModel`)
2. Use `BaseSettings` for env/CLI settings (not `NSSBaseModel`)
3. Use `AliasChoices` for env var mapping, not `env_prefix`
4. Always add `description` to `Field()` -- it becomes the CLI `--help` text
5. Use `Parameter[T]` for config fields that support `"auto"` or unset semantics
6. Use `mode="before"` validators for coercion, `mode="after"` for checks
7. Use `model_validator(mode="after")` for cross-field validation
8. Use `DependsOnValidator` for fields that are only valid when another field is set
9. Field separator is `"__"` (`CLI_NESTED_FIELD_SEPARATOR`) -- don't use `.` unless matching existing code
10. Max nesting depth is 2 -- `parse_overrides` raises `ValueError` for 3+ levels
