---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
name: pydantic
description: "Pydantic v2 patterns for this repo's config, CLI, and SDK models. Triggers on: Pydantic, BaseModel, NSSBaseModel, BaseSettings, field_validator, model_validator, ConfigDict, TypeAdapter, pydantic-settings, validation."
---

# Pydantic in Safe-Synthesizer

Pydantic v2 is the backbone of config, settings, CLI, and SDK validation. This repo has specific base classes and conventions.

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

## Validators

### field_validator

```python
from pydantic import field_validator

class MyModel(NSSBaseModel):
    mode: str
    
    @field_validator("mode", mode="before")
    @classmethod
    def coerce_mode(cls, v: str) -> str:
        """Coerce string to expected format."""
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

## Custom Pydantic Core Schema

For types that need special Pydantic integration (like `Parameter[T]`):

```python
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

class MyType:
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        return core_schema.union_schema([
            core_schema.is_instance_schema(cls),
            core_schema.no_info_before_validator_function(cls, handler(str)),
        ])
```

See `configurator/parameter.py` for the full `Parameter[T]` implementation.

## Conventions

1. Inherit from `NSSBaseModel` for config/result models (not bare `BaseModel`)
2. Use `BaseSettings` for env/CLI settings (not `NSSBaseModel`)
3. Use `AliasChoices` for env var mapping, not `env_prefix`
4. Always add `description` to `Field()` -- it becomes CLI help text
5. Use `mode="before"` validators for coercion, `mode="after"` for checks
6. Use `model_validator(mode="after")` for cross-field validation
7. Never use `model_config = {"validate_assignment": True}` unless explicitly needed (performance cost)
