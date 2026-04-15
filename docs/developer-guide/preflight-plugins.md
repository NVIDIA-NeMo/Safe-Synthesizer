<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Preflight Plugins

The preflight layer (`nemo_safe_synthesizer.preflight`) exposes a public
extension point for third-party validation checks. Plugins slot into the
same pipeline that ships with Safe Synthesizer and share its reporting,
gating, and failure-isolation machinery.

The preflight package is split by concern:

- `preflight.types` -- result and context dataclasses (no rendering deps).
- `preflight.base` -- the `PreflightCheck` ABC + stage subclasses.
- `preflight.registry` -- plugin registration and registry validation.
- `preflight.orchestrator` -- `run_preflight` and per-check gating.
- `preflight.checks` -- bundled core checks, grouped by stage.

Rendering a `PreflightReport` lives in
`nemo_safe_synthesizer.tooling.preflight.render_preflight_report`. The
preflight layer itself is rendering-free so that alternative output
modes (agent-friendly markdown, JSON, plain text) can be added in
`tooling` without touching check code. A `PreflightReport` is a list of
`PreflightCheckResult` entries (`name`, `status`, `issues`) -- one per
check the orchestrator considered. Display metadata (`label`,
`category`) is intentionally _not_ on the result; it lives on the
`PreflightCheck` class and the renderer looks it up from the registry
by check name.

## Writing a check

```python
import os

from nemo_safe_synthesizer.preflight import (
    ConfigCheck,
    IssueCollector,
    PreflightContext,
    register_preflight_check,
)


class LicenseCheck(ConfigCheck):
    name = "acme.license"           # non-core namespace prefix
    label = "License key"
    category = "environment"

    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
        if not os.environ.get("ACME_LICENSE"):
            collector.error("acme_missing_license", "ACME_LICENSE is required.")


register_preflight_check(LicenseCheck())
```

Subclass the stage ABC that matches what your check needs:

| ABC | Use when |
| --- | --- |
| `ConfigCheck` | only the resolved config is required |
| `DataFrameCheck` | the training DataFrame is required |
| `MetadataCheck` | model metadata (tokenizer, context length) is required |
| `AdvisoryCheck` | data-quality advisories that never gate downstream checks |

Picking the right stage matters beyond ergonomics. `ConfigCheck` /
`DataFrameCheck` / `MetadataCheck` errors are hard gates: any downstream
check listing the errored check in its `requires` is skipped. `AdvisoryCheck`
errors appear in the report but never gate dependents. If your check is
reporting a data-quality *finding* rather than a *blocker*, use
`AdvisoryCheck`.

The severity you hand to the collector follows the same logic: use
`collector.error` only when the pipeline genuinely cannot proceed.
"Looks suspicious; please review" is a `warning`, even in the advisory
stage.

## Worked example: an advisory data-quality check

A plugin that warns about exact-duplicate rows. Duplicates don't block
training, so this is `AdvisoryCheck` + `warning` -- never `error`.

```python title="acme_nss_plugins/duplicates.py"
from __future__ import annotations

from nemo_safe_synthesizer.preflight import (
    AdvisoryCheck,
    IssueCollector,
    PreflightContext,
    register_preflight_check,
)


class DuplicateRowsCheck(AdvisoryCheck):
    """Surface exact-duplicate rows in the training split."""

    name = "acme.duplicate_rows"    # namespace prefix; avoids core collisions
    label = "Duplicate rows"
    warn_ratio: float = 0.01        # tunable via subclass override
    loud_ratio: float = 0.25

    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
        data = ctx.data
        if len(data) == 0:          # runtime-state skip lives in `check`
            return

        n_dupes = int(data.duplicated().sum())
        if n_dupes == 0:
            return

        ratio = n_dupes / len(data)
        msg = f"{n_dupes} of {len(data)} rows ({ratio:.0%}) are exact duplicates"
        if ratio >= self.loud_ratio:
            collector.warning("acme_many_duplicates", f"{msg}; review your upstream pipeline.")
        elif ratio >= self.warn_ratio:
            collector.warning("acme_some_duplicates", f"{msg}.")


register_preflight_check(DuplicateRowsCheck())
```

A few non-obvious design choices:

- No `requires`: there is no upstream check whose failure would make this
  one meaningless. Don't invent dependencies; reserve `requires` for
  genuine preconditions (see the next example).
- No `enabled()` override: the empty-data guard belongs in `check()`
  because it is runtime state, not config state. `enabled()` is the
  home for declarative, config-driven skips.
- Thresholds as class attributes: a subclass can override `warn_ratio`
  without forking the check body. If you need user-facing tuning, expose
  them via your own config schema and read from `ctx.config`.
- Code prefix: both issue codes are prefixed `acme_` to avoid collision
  with core checks that filter on `code` alone.

## Worked example: a check with a real dependency

A plugin that warns about very small groups. This one legitimately uses
both `requires` and `enabled()` because it crashes if the group-by
column isn't valid and produces no signal when grouping isn't
configured.

```python title="acme_nss_plugins/tiny_groups.py"
from __future__ import annotations

from nemo_safe_synthesizer.preflight import (
    AdvisoryCheck,
    IssueCollector,
    PreflightContext,
    register_preflight_check,
)


class TinyGroupsCheck(AdvisoryCheck):
    """Warn about groups with very few rows."""

    name = "acme.tiny_groups"
    label = "Tiny groups"
    requires = ("columns.groupby",)   # real dependency: bail if group col invalid
    min_group_size: int = 3

    def enabled(self, ctx: PreflightContext) -> bool:
        # Declarative skip on config state -- `enabled` is the right home.
        if not super().enabled(ctx):
            return False
        return ctx.config.data.group_training_examples_by is not None

    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
        data = ctx.data
        group_col = ctx.config.data.group_training_examples_by
        if group_col not in data.columns:
            return

        group_sizes = data.groupby(group_col).size()
        tiny_groups = int((group_sizes < self.min_group_size).sum())
        if tiny_groups > 0:
            collector.warning(
                "acme_tiny_groups",
                f"{tiny_groups} group(s) have fewer than {self.min_group_size} rows.",
            )


register_preflight_check(TinyGroupsCheck())
```

Compare the two: `DuplicateRowsCheck` stands alone, `TinyGroupsCheck`
encodes both "don't run me until the group column has been validated"
(`requires`) and "don't run me at all when grouping is off" (`enabled`).

This check previously shipped as a core check
(`dataset.tiny_groups`) before being moved out to demonstrate the
plugin API. The threshold (`min_group_size`) and the framing of "few
rows" are judgement calls that belong in a plugin rather than the core
pipeline.

## Helpers

`nemo_safe_synthesizer.preflight.helpers` exposes small translators
between common failure modes and `IssueCollector` calls. Callers own
the issue code, the helper owns the try/except convention.

```python
from nemo_safe_synthesizer.preflight import (
    DataFrameCheck,
    IssueCollector,
    PreflightContext,
    helpers,
)
from nemo_safe_synthesizer.data_processing.validation import check_column_present
from nemo_safe_synthesizer.errors import ParameterError


class MyColumnCheck(DataFrameCheck):
    name = "acme.my_column"
    label = "My column"

    def check(self, ctx: PreflightContext, collector: IssueCollector) -> None:
        helpers.emit_on_raise(
            collector,
            lambda: check_column_present(ctx.data, "amount", role="Amount"),
            expect=ParameterError,
            code="acme_amount_missing",
        )
```

Available helpers:

- `emit_on_raise(collector, action, *, expect, code, severity="error")`
  -- run `action`; if it raises `expect`, emit an issue with `code` and
  the exception message and return `False`. Unexpected exceptions
  propagate to the orchestrator's crash handler.
- `require_import(collector, module, *, code, message, severity="error")`
  -- import `module` or emit an issue and return `None`. Use this for
  checks that depend on an optional runtime like `torch`.
- `resolved_record_count(ctx)` -- return
  `config.training.num_input_records_to_sample` once it has been
  normalized to a concrete `int`; returns `None` if it still carries a
  sentinel like `"auto"`.

## Testing a check

Unit-test the check logic in isolation -- no need to touch the global
registry. Instantiate the check, build a minimal `PreflightContext`,
and invoke `check()` with your own `IssueCollector`:

```python title="tests/test_duplicate_rows.py"
import pandas as pd
from unittest.mock import MagicMock

from acme_nss_plugins.duplicates import DuplicateRowsCheck
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.preflight import IssueCollector, PreflightContext


def _ctx(df: pd.DataFrame) -> PreflightContext:
    return PreflightContext(data=df, config=SafeSynthesizerParameters(), metadata=MagicMock())


def test_many_duplicates_warns_loudly():
    df = pd.concat([pd.DataFrame({"x": range(10)})] * 4, ignore_index=True)   # 75% duplicates
    collector = IssueCollector(check_name="acme.duplicate_rows")
    DuplicateRowsCheck().check(_ctx(df), collector)
    assert [i.code for i in collector.issues] == ["acme_many_duplicates"]


def test_no_duplicates_is_silent():
    df = pd.DataFrame({"x": range(100)})
    collector = IssueCollector(check_name="acme.duplicate_rows")
    DuplicateRowsCheck().check(_ctx(df), collector)
    assert collector.issues == []
```

Integration tests that exercise `register_preflight_check` should call
`reset_preflight_plugins()` in teardown (or fixture scope) so a failing
test doesn't leak plugin instances into the rest of the suite:

```python
import pytest
from nemo_safe_synthesizer.preflight import reset_preflight_plugins


@pytest.fixture(autouse=True)
def _clean_preflight_plugins():
    yield
    reset_preflight_plugins()
```

## Rules

### At class-definition time

- `name`, `label`, and `stage` must be set (`stage` comes from the stage
  ABC you subclass).
- `__preflight_api_version__` must be a value in the set supported by
  the installed runtime (currently includes `1`). Bump this on your
  subclass only once a new major preflight API ships.

### At registration time

- The first dotted segment of `name` must not match a reserved core
  namespace (e.g. `gpu.vram` is reserved; `gpux.foo` is fine). The
  authoritative list of reserved namespaces lives in `_CORE_NAMESPACES`
  in `src/nemo_safe_synthesizer/preflight/base.py`. You don't need to
  memorise it: the `ValueError` raised by `register_preflight_check`
  includes the current set, so a collision tells you what to avoid.

## Runtime behavior

- Plugin checks are appended after all core checks and stably sorted
  into the correct stage block by `build_registry`.
- Users can disable any check by name via config:

  ```yaml
  preflight:
    disabled_checks:
      - acme.license
      - gpu.vram
  ```

- Disabling a check via `disabled_checks` also skips any downstream
  check that lists it in `requires`, exactly as if the upstream check
  had errored.
- Uncaught exceptions raised by `check()`, `run()`, or `enabled()` are
  captured and reported as a `PreflightIssue` with code
  `preflight.check_crash`; the registry continues to execute.
- `PreflightIssue.namespace` returns the prefix before the first `.`
  in the check name, or `None` if the check name has no dotted prefix.
- `reset_preflight_plugins()` clears every registered plugin and
  rebuilds the global `PREFLIGHT_REGISTRY` from core checks only.
  Useful for a clean slate between test cases or notebook re-runs:

  ```python
  from nemo_safe_synthesizer.preflight import reset_preflight_plugins

  reset_preflight_plugins()
  ```

- `run_preflight` accepts an optional keyword-only `registry=` argument.
  Callers who want to run against a bespoke registry (for example,
  a vendored subset of core checks plus one or two plugins) build a
  `PreflightRegistry` via `build_registry(...)` and pass it explicitly,
  without mutating the global registry:

  ```python
  from nemo_safe_synthesizer.preflight import build_registry, run_preflight

  report = run_preflight(
      df, config, metadata,
      registry=build_registry((MyCheck(), AnotherCheck())),
  )
  ```

  A `PreflightRegistry` is a frozen, name-keyed view: iterate with
  `for check in registry`, look up with `registry[name]`, test with
  `name in registry`.

## Categories

Each `PreflightCheck` declares a `category` string (on the class) that
determines which rendered panel it appears under in the Rich report.
The renderer reads this off the registry at render time -- it is not
carried on individual `PreflightIssue` values. Core checks use three
values:

- `data quality` (the default)
- `environment`
- `configuration`

Every distinct `category` string produces its own panel, so plugins
should reuse one of these three values unless there is a strong reason
to introduce a new category.

## Issue codes

Issue `code` strings are not namespaced by the framework. Downstream
consumers that filter only on `code` may collide across checks. Prefer
filtering on the `(check, code)` pair, or prefix your codes (for
example, `"acme_missing_license"`) to reduce collision risk.

## Config access stability

`ctx.config` exposes the full `SafeSynthesizerParameters` schema.
Dotted attribute paths on that schema are *not* part of the preflight
plugin API contract and may change between minor releases. Design
plugins to degrade gracefully (e.g. guard with `getattr` or a
`try/except AttributeError`) if a field moves.

See the package docstring of
[`src/nemo_safe_synthesizer/preflight/__init__.py`](../../src/nemo_safe_synthesizer/preflight/__init__.py)
for the authoritative reference.
