<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Configuration Management

The configuration-management layer maps nested Pydantic parameter models to
Click options, then maps Click's flat keyword arguments back to the nested
structure Pydantic expects. It powers the CLI surfaces that expose
`SafeSynthesizerParameters`.

For general Pydantic style rules, see [STYLE_GUIDE.md](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/STYLE_GUIDE.md#data-modeling).
This page covers repository-specific mechanics.

The type system is still growing. The long-term goal is that every setting
that changes runtime behavior is defined in the main configuration system,
validated through typed models, and overridable from the CLI.

## Click Option Generation

Source: `src/nemo_safe_synthesizer/configurator/pydantic_click_options.py`.

```text
Pydantic model fields -> _collect_params() -> LeafParam / FlagParam -> click.option(...)
```

`pydantic_options(model_class, field_separator="__")` walks the model tree:

- Nested Pydantic models become flattened option paths, such as
  `--data__holdout`.
- Scalar fields become `LeafParam` options and preserve underscores in field
  names, such as `--training__learning_rate`.
- Nullable sub-model fields (`SomeModel | None`) become nested leaf options
  plus a disabling `FlagParam`.

The generated Click options use `Field(description=...)` as CLI help text.

## Nullable Sub-Config Flags

Optional sub-configs use `None` as the disabled signal. Do not add a second
boolean flag for the same feature.

For a field such as `replace_pii: PiiReplacerConfig | None`,
`pydantic_options()` emits:

```bash
safe-synthesizer run --no-replace-pii
```

Click stores that flag as `no_replace_pii=True`, and `parse_overrides()`
converts it to:

```python
{"replace_pii": None}
```

Leaf options and disabling flags intentionally use different spellings:

- Leaf options preserve generated field paths: `--training__learning_rate`.
- Disable flags use standard CLI dashes: `--no-replace-pii`.

## Override Parsing

`parse_overrides(values, field_sep="__")` reverses the flattening:

```python
parse_overrides({"data__holdout": 0.1})
# {"data": {"holdout": 0.1}}
```

It drops unset values:

- `None` values from regular Click options are ignored.
- `no_<field>=False` is ignored.
- `no_<field>=True` becomes `{field: None}`.

The `field_sep` passed to `parse_overrides()` must match the
`field_separator` used by `pydantic_options()`.

## Parameter Models

Most top-level user configuration groups subclass `Parameters`, not bare
`BaseModel`. `Parameters` lives in
`src/nemo_safe_synthesizer/configurator/parameters.py` and provides recursive
iteration, name-based lookup, and YAML / JSON helpers.

Use `NSSBaseModel` from `src/nemo_safe_synthesizer/config/base.py` for config
or result models that do not need `Parameters` tree behavior.

The shared `pydantic_model_config` sets `arbitrary_types_allowed`,
`validation_error_cause`, `from_attributes`, `validate_default`,
`protected_namespaces=()`, and `json_schema_mode_override="validation"`.

`SafeSynthesizerParameters.unknown_fields` controls Pydantic's recursive
unknown-field policy at raw-input boundaries:

- `"reject"` is the default and maps to `extra="forbid"`. Normalization
  preserves unknown keys until Pydantic reports them.
- `"ignore"` maps to `extra="ignore"` for compatibility with stale
  configurations and notebooks or mismatched client and service versions.
  Normalization filters unknown keys.
- `extra="allow"` is unsupported because Safe Synthesizer cannot act on
  undeclared fields.

Sparse patch compilation and SDK raw-mapping builders use the same effective
setting. Complete inputs use the field default when it is omitted; partial
patches inherit the existing configuration's policy.

## Adding a Config Field

Add the field to the relevant parameter model. The CLI option is generated
automatically:

```python
class DataParameters(Parameters):
    my_new_field: int = Field(default=10, description="Description for CLI help.")
```

```bash
safe-synthesizer run --data__my_new_field 20
```

Nested models generate nested option paths:

```python
class DataParameters(Parameters):
    sub: MySubConfig = Field(default_factory=MySubConfig)
```

```bash
safe-synthesizer run --data__sub__threshold 0.75
```

## Parameter Wrappers

`src/nemo_safe_synthesizer/configurator/parameter.py` defines wrappers for
configuration values:

| Type | Meaning |
| ---- | ------- |
| `Parameter[T]` | Wrapped value accepted by Pydantic from raw `T` or a `Parameter` instance. |
| `AutoParam[T]` | Value can be the `"auto"` sentinel and resolved later. |
| `UnsetParam[T]` | Field was not set. |
| `ValidNoneParam[T]` | `None` is intentional, not unset. |

`Parameter[T]` serializes to `.value` and compares against raw values of the
wrapped type.

`AutoParamType` lets Click accept either `"auto"` or the wrapped scalar type.
It is used only when the string literal side is exactly `"auto"`.

## Validators and Settings

General Pydantic validator style lives in [STYLE_GUIDE.md](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/STYLE_GUIDE.md#data-modeling). Repository-specific validators live in
`src/nemo_safe_synthesizer/configurator/validators.py`:

- `DependsOnValidator`: field is valid only when another field is set.
- `ValueValidator`: validates the effective field value.
- `AutoParamRangeValidator`: validates non-negative fields that support
  `"auto"`.

Settings classes use `pydantic-settings` `BaseSettings`, not `NSSBaseModel`.
Prefer `AliasChoices` on individual fields when a setting must respond to both
its Python field name and an environment variable. `env_prefix` is acceptable
when every field shares one prefix.

Filter unset CLI values before constructing settings objects:

```python
settings_kwargs = {key: value for key, value in kwargs.items() if value is not None}
```

Use `TypeAdapter` for validation when a full model would add noise.

## Key Files

| File | Purpose |
| ---- | ------- |
| `src/nemo_safe_synthesizer/configurator/pydantic_click_options.py` | `pydantic_options()`, `_collect_params()`, `LeafParam`, `FlagParam`, and `parse_overrides()`. |
| `src/nemo_safe_synthesizer/configurator/parameter.py` | `Parameter[T]` and wrapper types. |
| `src/nemo_safe_synthesizer/configurator/validators.py` | Repository-specific Pydantic validator helpers. |
| `src/nemo_safe_synthesizer/configurator/parameters.py` | `Parameters` base class for user-facing configuration groups. |
| `src/nemo_safe_synthesizer/config/base.py` | `NSSBaseModel` and `pydantic_model_config`. |
