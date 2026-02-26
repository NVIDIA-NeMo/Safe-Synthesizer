<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Style Guide

Style conventions for Safe Synthesizer -- for humans and AI agents alike.

This guide covers taste: how to write code, documentation, configs, and tests.
For contribution process (PRs, branches, DCO, CI), see [CONTRIBUTING.md](CONTRIBUTING.md).
For architecture and module map, see [AGENTS.md](AGENTS.md) and [docs/developer-guide/architecture.md](docs/developer-guide/architecture.md).
For the full test matrix, markers, and fixture catalog, see [tests/TESTING.md](tests/TESTING.md).

## Contents

- [Principles](#principles)
- [Python: library conventions](#python-library-conventions) -- what to use
  - [Data modeling](#data-modeling)
  - [Logging and observability](#logging-and-observability)
  - [Error hierarchy](#error-hierarchy)
- [Python: code style](#python-code-style) -- how to write
  - [Type hints](#type-hints)
  - [Control flow](#control-flow)
  - [Keeping functions flat](#keeping-functions-flat)
  - [Error messages](#error-messages)
  - [Resource cleanup](#resource-cleanup)
  - [Naming](#naming)
  - [Imports](#imports)
  - [Docstrings](#docstrings)
  - [Patterns to avoid](#patterns-to-avoid)
- [Testing](#testing)
- [Markdown](#markdown)
- [Dockerfiles](#dockerfiles)
- [Shell scripts](#shell-scripts)
- [Configuration files](#configuration-files)
- [General conventions](#general-conventions)

---

## Principles

- Use American English spelling: "initialize" not "initialise", "recognize" not "recognise", "color" not "colour".
- Tools enforce what they can (`ruff`, `ty`, `pre-commit`). This guide covers what tools can't enforce.
- Some rules below are aspirational -- legacy code is being migrated. New code must follow these conventions; existing deviations are tolerated during migration.
- The current ruff rule set ([ruff.toml](ruff.toml)) does not enforce `UP006`/`UP007`/`B006`/`T201`. These rules are review-enforced until the ruff config is expanded.
- `__all__` defines the public API surface. Identifiers with a leading `_` are private and can change without notice.

---

## Python: library conventions

These are the building blocks of this codebase -- the specific types, base classes, and patterns you reach for when writing Safe Synthesizer code.

### Data modeling

- Pydantic: `NSSBaseModel` for config/parameter models in `config/` which define the user-facing configuration of NSS. Raw `BaseModel` or module-specific bases (e.g., `ReportBaseModel`) for data transfer objects and internal structures.
- `BaseSettings` for env/CLI settings. Prefer `AliasChoices` on individual fields when you need a field to respond to both its Python name and an env var name (e.g., `validation_alias=AliasChoices("config_path", "NSS_CONFIG")`). `env_prefix` is acceptable for simple settings classes where all fields share a common prefix and no per-field aliasing is needed.
- Always add `description` to `Field()` -- it becomes CLI help text.
- `@dataclass(frozen=True)` preferred for immutable value objects and validators. Mutable `@dataclass` acceptable for builders, accumulators, and pipeline state.
- `field(default_factory=list)` for mutable defaults, never `= []`.
- `StrEnum` for string-valued enums used in configs/serialization. Plain `Enum` for internal-only named constants.

```python
class TrainingHyperparams(NSSBaseModel):
    learning_rate: float = Field(default=2e-4, description="Learning rate for the optimizer.")
    num_epochs: int = Field(default=3, description="Number of training epochs.")
    lora_rank: int = Field(default=16, description="Rank of the LoRA adapter.")
```

### Logging and observability

- `observability.get_logger(__name__)` -- never `logging.getLogger()` or `structlog.get_logger()` directly
- Category loggers: `.runtime` for internals, `.user` for progress/results, `.system` for system events
- `@traced` decorators are an optional enhancement for entry-point functions, not a universal requirement
- Never `print()` for operational output. Approved alternatives: `click.echo()` for CLI output, `sys.stdout.write()` for raw output in tools.
- Use `extra={}` for data that downstream tools should query or aggregate (metrics, counts, durations). f-strings are fine for human-readable context that doesn't need machine parsing.

```python

# inside the package while developing, note the relative import
from ..observability import get_logger

logger = get_logger(__name__)

# Structured data that tools should query -- use extra={}
logger.user.info("Training complete", extra={"epochs": 3, "loss": 0.42})
logger.runtime.debug("Memory usage", extra={"bytes": freed_bytes, "phase": "teardown"})

# Human-readable context -- f-strings are fine
logger.info(f"Loading model from: {model_path}")
logger.warning(f"Column {column!r} not found, skipping")
```

### Error hierarchy

Raise from the custom hierarchy with dual inheritance so callers can catch either the library-specific type or the built-in:

- `SafeSynthesizerError` -- base for all known errors
- `UserError(SafeSynthesizerError)` -- invalid usage (bad inputs, uninitialized state)
- `DataError(UserError, ValueError)` -- problems with training data
- `ParameterError(UserError, ValueError)` -- invalid config or parameter input
- `GenerationError(UserError, RuntimeError)` -- sampling/generation failures
- `InternalError(SafeSynthesizerError, RuntimeError)` -- library bug (equivalent to HTTP 5xx)

---

## Python: code style

How to write clear, testable Python -- independent of which library primitives you use.

### Type hints

The codebase targets Python 3.11+ and uses native typing syntax throughout. Expect this minimum for the foreseeable future.

```python
# After
from typing import Self

def process(data: pd.DataFrame, columns: list[str] | None = None) -> Self:

# Before
def process(data: pd.DataFrame, columns: Optional[List[str]] = None) -> "SafeSynthesizer":
```

- `X | Y` not `Optional[X]` or `Union[X, Y]`
- `list[str]` not `List[str]`, `dict[str, int]` not `Dict[str, int]`
- `Self` for fluent method returns
- Collection ABCs for function arguments (`Sequence`, `Mapping`, `Iterable`) so callers can pass any compatible container; concrete types for return values so callers know exactly what they get
- `Protocol` for structural subtyping when you need duck-typing boundaries
- Avoid `Any` -- prefer `object`, generics, or `Protocol`
- `TYPE_CHECKING` guards for heavy imports (`pandas`, `torch`, `transformers`); not needed for stdlib or lightweight imports

Legacy modules (`pii_replacer/`, `data_processing/`, `privacy/`, `artifacts/`) still use `Optional`/`List`/`Dict` in ~250 places -- migration in progress.

### Control flow

- Prefer `match`/`case` for dispatch on types or tagged values. Not a blanket rule -- `if`/`elif` is fine for simple boolean predicates.
- Comprehensions over imperative loops where intent is clearer. No multiple `for` clauses -- optimize for readability, not conciseness (per [Google Python Style Guide sec 2.7](https://google.github.io/styleguide/pyguide.html#27-comprehensions--generator-expressions)).
- Clamping/saturation over raising when out-of-range inputs shouldn't crash the system -- prefer returning a bounded value with a log warning over raising (e.g., `p = max(0.0, min(p, 1.0))`).

Builder pattern -- `with_*` methods return `Self`:

```python
synthesizer = (
    SafeSynthesizer(config)
    .with_data_source(df)
    .with_train(learning_rate=0.0001)
    .with_generate(num_records=10000)
)
```

Config resolution via `match`/`case`:

```python
match values:
    case BaseModel() as model:
        return model.model_copy(update=overrides)
    case dict() as d:
        return cls.model_validate(d).model_copy(update=overrides)
    case None:
        return cls(**overrides)
```

### Keeping functions flat

Deeply nested code is hard to read, test, and modify. If a function has more than two levels of indentation beyond `def`, it needs work.

Here is a bad example -- arrow-shaped code with nested loops, conditionals, and interleaved logic:

```python
# No -- deeply nested, hard to test individual pieces
def build_training_examples(records, tokenizer, config):
    examples = []
    for group in records:
        if group:
            tokens_used = 0
            for record in group:
                if record.is_valid():
                    encoded = tokenizer.encode(record.text)
                    if tokens_used + len(encoded) <= config.token_budget:
                        tokens_used += len(encoded)
                        if config.include_metadata:
                            encoded = _prepend_metadata(encoded, record)
                        examples.append(encoded)
                    else:
                        break
    return examples
```

Three ways to rewrite it:

#### Option A: guard clauses + extract inner loop

Flatten the nesting with early `continue`/`break`, and pull the inner loop into a named helper with a single job:

```python
def _collect_from_group(group, tokenizer, config):
    collected = []
    tokens_used = 0
    for record in group:
        if not record.is_valid():
            continue
        encoded = tokenizer.encode(record.text)
        if tokens_used + len(encoded) > config.token_budget:
            break
        tokens_used += len(encoded)
        if config.include_metadata:
            encoded = _prepend_metadata(encoded, record)
        collected.append(encoded)
    return collected

def build_training_examples(records, tokenizer, config):
    examples = []
    for group in records:
        if not group:
            continue
        examples.extend(_collect_from_group(group, tokenizer, config))
    return examples
```

#### Option B: generator with `yield`

Use a generator to separate iteration from collection. Each `yield` represents one valid example -- the caller decides how to consume them:

```python
def _iter_examples(records, tokenizer, config):
    for group in records:
        if not group:
            continue
        tokens_used = 0
        for record in group:
            if not record.is_valid():
                continue
            encoded = tokenizer.encode(record.text)
            if tokens_used + len(encoded) > config.token_budget:
                break
            tokens_used += len(encoded)
            if config.include_metadata:
                encoded = _prepend_metadata(encoded, record)
            yield encoded

def build_training_examples(records, tokenizer, config):
    return list(_iter_examples(records, tokenizer, config))
```

#### Option C: functional decomposition with higher-order functions

Break the problem into composable steps. Each function is pure and independently testable:

```python
def _valid_records(group):
    return (r for r in group if r.is_valid())

def _encode_within_budget(records, tokenizer, budget):
    tokens_used = 0
    for record in records:
        encoded = tokenizer.encode(record.text)
        if tokens_used + len(encoded) > budget:
            return
        tokens_used += len(encoded)
        yield record, encoded

def _maybe_add_metadata(record, encoded, include_metadata):
    if include_metadata:
        return _prepend_metadata(encoded, record)
    return encoded

def build_training_examples(records, tokenizer, config):
    examples = []
    for group in records:
        if not group:
            continue
        for record, encoded in _encode_within_budget(
            _valid_records(group), tokenizer, config.token_budget
        ):
            examples.append(_maybe_add_metadata(record, encoded, config.include_metadata))
    return examples
```

#### Named predicates for complex conditions

When a condition is a wall of boolean logic, name each piece:

```python
# Before -- opaque
if (prev_row_idx is not None and row_idx < prev_row_idx) or \
   (current_group is not None and record_group != current_group) or \
   num_sequences >= max_sequences or \
   token_total + record_len > token_budget:
       flush_example()

# After -- each condition tells you what it means
restart_boundary = prev_row_idx is not None and row_idx < prev_row_idx
group_boundary = current_group is not None and record_group != current_group
would_exceed_seq = num_sequences >= max_sequences
would_exceed_tokens = token_total + record_len > token_budget

if restart_boundary or group_boundary or would_exceed_seq or would_exceed_tokens:
    flush_example()
```

Or extract it entirely when the predicate is reused or has domain meaning:

```python
def _should_flush_example(*, prev_row_idx, row_idx, current_group, record_group,
                          num_sequences, max_sequences, token_total, record_len, token_budget) -> bool:
    """Determine if the current training example should be flushed and a new one started."""
    restart_boundary = prev_row_idx is not None and row_idx < prev_row_idx
    group_boundary = current_group is not None and record_group != current_group
    return restart_boundary or group_boundary or num_sequences >= max_sequences or token_total + record_len > token_budget
```

### Error messages

Error messages must precisely match the actual error condition. Interpolated pieces must be clearly identifiable:

```python
# After
raise ValueError(f"Not a probability: {p!r}")
raise DataError(f"Column {column!r} not found in dataframe with columns {list(df.columns)}")

# Before
raise ValueError("Invalid value")
raise DataError(f"The {column} column could not be processed.")
```

### Naming

- PascalCase classes, snake_case functions/variables, UPPER_SNAKE_CASE constants, leading `_` for private
- Type variables: PascalCase with `T` suffix (`DataT`, `ParameterT`)
- Private methods: `_determine_*` for internal resolution, `_resolve_*` for config handling

### Imports

- Order of imports: 1) stdlib, 2) third-party, 3) local (enforced by ruff I001/I002)
- Relative imports in `src/` (`from ..observability import get_logger`), absolute imports in `tests/` (`from nemo_safe_synthesizer.observability import get_logger`)
- `TYPE_CHECKING` blocks for heavy forward references (`pandas`, `torch`, `transformers`)
- `from __future__ import annotations` -- do not add to modules that don't already have it (adding it can surface subtle type issues in legacy code). New modules may omit it since modern syntax (`X | Y`, `list[str]`) works natively.

### Resource cleanup

- `try/finally` for resource cleanup (never rely on `__del__` alone)
- `except Exception:` + `logger.debug(..., exc_info=True)` for non-fatal cleanup at teardown boundaries -- this is distinct from the "patterns to avoid" rule against defensive `try/except Exception` wrapping trusted internal calls that shouldn't fail
- Bare `except Exception: pass` only in `__del__` methods where suppression is intentional
- Use `raise X from e` to chain exceptions when re-raising with a different type (e.g., `raise GenerationError(f"Failed to write: {e}") from e`)

Standard context managers (`with open(...)`, `with lock:`) are fine when they fit. Prefer `try/finally` over `__del__`, custom context managers that obscure when resources are acquired/released, or bare `return` that skips cleanup. Multi-step cleanup should isolate each step so one failure doesn't prevent the next:

```python
try:
    ss.run()
    ss.save_results(workdir)
finally:
    if hasattr(ss, "generator") and ss.generator is not None:
        ss.generator.teardown()
```

For backends with expensive resources, use the `_torn_down` guard pattern:

```python
def teardown(self) -> None:
    if self._torn_down:
        return
    self._torn_down = True

    try:
        cleanup_dist_env_and_memory()
    except Exception:
        logger.debug("cleanup_dist_env_and_memory failed during teardown", exc_info=True)

    self.llm = None

    try:
        cleanup_memory()
    except Exception:
        logger.debug("cleanup_memory failed during teardown", exc_info=True)
```

### Docstrings

Google style is mandatory. Canonical references:
[Google Python Style Guide sec 3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings),
[Napoleon examples](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).

#### When docstrings are required

A docstring is mandatory for every function that has one or more of: being part of the public API, nontrivial size, or non-obvious logic.

Overridden methods decorated with `@override` do not need a docstring unless they materially refine the base contract.

Private helper methods (`_name`) with trivial implementations need no docstring. Private methods with non-obvious logic should use the tier matching their complexity.

#### Tiers

These are three classes of documentation depth, not a preference ranking. Use the tier appropriate to the complexity of the particular function or class.

Tier 1 -- simple/obvious. One-line summary, no `Args:` block if the signature is self-documenting:

```python
def teardown(self) -> None:
    """Release GPU memory and clean up distributed resources."""
    self._clear_llm_state()
```

Tier 2 -- moderate. Summary + `Args:` / `Returns:` / `Raises:` blocks:

```python
def _resolve_config(self, values: ParamDict | NSSParameters | None, cls: type[ParamT], **kwargs) -> ParamT:
    """Resolve configuration from various input types.

    Merges caller-supplied overrides on top of a base config. Accepts Pydantic models
    (copied with updates), plain dicts (validated then updated), or None (built from
    overrides alone).

    Args:
        values: Base configuration -- a Pydantic model, a dict, or None.
        cls: The Pydantic model class to validate against.
        **kwargs: Field-level overrides applied on top of the base.

    Returns:
        An instance of `cls` with all overrides applied.

    Raises:
        TypeError: If `values` is not a BaseModel, dict, or None.
    """
```

Tier 3 -- complex. Summary paragraph explaining WHY and WHEN, full sections, lifecycle documentation:

```python
class GeneratorBackend(metaclass=abc.ABCMeta):
    """Abstract base for generation backends.

    Lifecycle: __init__ -> initialize() -> generate() [-> generate() ...] -> teardown()

    teardown() must be idempotent and safe to call multiple times.
    Callers should use try/finally to guarantee teardown runs even if
    generate() raises. Each cleanup step is isolated so one failure
    doesn't prevent the next from running.

    Subclasses must implement initialize(), generate(), and teardown().
    The base class provides the _torn_down guard flag pattern -- check
    it at the top of teardown() and set it before returning.

    Attributes:
        config: Generation parameters controlling temperature, top_p, etc.
        adapter_path: Path to the trained LoRA adapter directory.
    """
```

#### Before and after

Vague module docstring:

```python
# Before
"""
Custom error classes
"""

# After
"""
Error hierarchy for Safe Synthesizer.

All public exceptions inherit from SafeSynthesizerError. User-facing errors
(bad data, bad config, generation failure) inherit from UserError and a
matching built-in (ValueError, RuntimeError) so callers can catch either.

Classes:
    SafeSynthesizerError: Base for all known errors.
    UserError: Invalid usage (bad inputs, uninitialized state).
    InternalError: Library bug (equivalent to HTTP 5xx).
    DataError: Problems with training data (NaNs, unsupported types).
    ParameterError: Invalid config or parameter input.
    GenerationError: Sampling/generation failures.
"""
```

No docstring on abstract method:

```python
# Before
@abc.abstractmethod
def teardown(self):
    pass

# After
@abc.abstractmethod
def teardown(self) -> None:
    """Release all resources held by this backend.

    Must be idempotent -- safe to call multiple times. Implementations
    should use the ``_torn_down`` guard flag and isolate each cleanup
    step so one failure doesn't prevent subsequent cleanup.

    Callers should wrap generate() in try/finally to guarantee this runs.
    """
```

Class without Attributes section:

```python
# Before
class SafeSynthesizerParameters(Parameters):
    """Main configuration class for the Safe Synthesizer pipeline."""

# After
class SafeSynthesizerParameters(Parameters):
    """Main configuration class for the Safe Synthesizer pipeline.

    Orchestrates all aspects of synthetic data generation including training,
    generation, privacy, evaluation, and data handling. Provides cross-field
    validation to ensure parameter compatibility.

    Attributes:
        data: Data parameters (holdout ratio, column config, etc.).
        replace_pii: PII replacement parameters.
        training: Training hyperparameters (learning rate, epochs, LoRA config).
        generation: Generation parameters (temperature, top_p, num_records).
        privacy: Differential privacy parameters (epsilon, delta).
        evaluation: Evaluation component toggles and settings.
        enable_synthesis: Enable synthesizing new data by training a model.
        enable_replace_pii: Enable replacing PII in the data.

    Example:
        config = SafeSynthesizerParameters.from_yaml("config.yaml")
        synthesizer = SafeSynthesizer(config).with_data_source("data.csv")
        synthesizer.run()
    """
```

Generator with `Yields:` instead of `Returns:`:

```python
# Before (uses Returns: for a generator, vague description)
def generate_batches(self, num_records: int) -> Iterator[pd.DataFrame]:
    """Generate synthetic records in batches.

    Returns:
        Batches of synthetic records.
    """

# After (Yields: tells the reader what each iteration produces)
def generate_batches(self, num_records: int) -> Iterator[pd.DataFrame]:
    """Generate synthetic records in batches.

    Each batch contains up to ``batch_size`` records. Invalid records are
    filtered and retried until ``num_records`` valid records are produced
    or the retry limit is reached.

    Args:
        num_records: Total number of valid records to generate.

    Yields:
        DataFrame of valid synthetic records for the current batch.
    """
```

`@property` -- use attribute style, not method style:

```python
# Before (method style -- wrong)
@property
def adapter_path(self) -> Path:
    """Returns the adapter path."""

# After (attribute style -- correct)
@property
def adapter_path(self) -> Path:
    """The path to the trained LoRA adapter directory."""
```

Redundant docstring that restates the signature:

```python
# Before (restates the obvious)
def get_logger(name: str | None = None) -> CategoryLogger:
    """Get a logger with the given name."""

# After (explains what the caller actually needs to know)
def get_logger(name: str | None = None) -> CategoryLogger:
    """Return a category logger for structured logging.

    Always pass ``__name__`` as the argument. The returned logger has
    sub-loggers (``.runtime``, ``.user``, ``.system``, ``.backend``)
    for categorized output. Logging is NOT initialized on import --
    entry points must call ``initialize_observability()`` first.
    """
```

#### Quick reference

The before/after examples above demonstrate most rules. These additional points are not shown in the examples:

- No decorative `**bold**` in docstrings
- Document side effects, thread safety, and idempotency guarantees where applicable
- Use `Example:` sections with working code for public API methods
- Complex code deserves proportionally detailed explanation -- err on the side of more context
- Cross-references in docstrings: use double backticks (` `` `) for inline code, `:meth:`method_name` `, `:class:`ClassName` `, and `:func:`function_name` ` for API cross-links in `MkDocs`/Sphinx

### Patterns to avoid

- Narrating comments -- comments should explain "why", not "what"
- Redundant docstrings that restate the function signature
- Defensive `try/except Exception` on trusted internal paths
- `# type: ignore` / `# ty: ignore` without attempting a fix
- `cast()` / `Any` to paper over type mismatches
- `os.path` -- use `pathlib.Path` (tolerated only in vendored/tooling scripts)
- Mutable default arguments -- use `None` and initialize inside the function. Acceptable immutable defaults: `None`, `str`, `int`, `float`, `bool`, `tuple`, `frozenset`, `pathlib.Path`
- `print()` statements -- use `get_logger(__name__)` from `observability.py` or `click.echo()` for CLI
- `assert` for validation in library code -- `assert` statements can be stripped by `-O` and must never guard correctness. Use `if/raise` for input validation. `assert` is fine in tests where `pytest` relies on it.

---

## Testing

Testing conventions are substantial enough to warrant their own section. For the full test matrix, markers, and fixture catalog, see [tests/TESTING.md](tests/TESTING.md). This section covers style conventions for writing tests.

- File naming: `test_*.py`; class naming: `Test*`; function naming: `test_<module>_<expected_behavior>`
- Fixtures: `fixture_` prefix convention for grep-ability and to separate fixtures from test functions. Add `# Purpose:` comments describing usage and data.
- Fixture scope: function-scoped by default. Session scope only when empirically justified by test runtime -- not based on assumptions about cost.
- Assertions: bare `assert` is the primary style; `pytest.raises()` with `match=` for exceptions; `pytest.approx()` for floating-point comparisons
- Docstrings: optional for simple tests, recommended for complex/e2e tests explaining purpose
- Markers: auto-assigned by path via `pytest_collection_modifyitems` (`/e2e/` -> `e2e`, `/gpu_integration/` -> `gpu_integration`, default -> `unit`). Explicit markers: `@pytest.mark.slow`, `@pytest.mark.timeout()`.
- `conftest.py`: shared fixtures per directory; root conftest has `load_test_dataset()` and `load_test_dataframe()` helpers
- Use `tmp_path` fixture for file operations, never write to the repo tree
- Mark CUDA-dependent tests with `@pytest.mark.e2e` or `@pytest.mark.gpu_integration`
- Mock only external boundaries, not internal implementation details
- Test isolation: no shared mutable state or execution-order dependencies between tests. If something must be run first before executing a test, include it in the test or a fixture.
- Use `@pytest.mark.parametrize` for testing multiple input combinations rather than copy-pasting similar tests

---

## Markdown

- No decorative `**bold**` in body text, list items, or docstrings. Use headers, list markers, colons, and backticks for structure. Bold acceptable only in table header-like cells.
- Use `--` (em-dash) for asides, not `-` (hyphen).
- Use single backticks for code identifiers, paths, and CLI commands in markdown. In Python docstrings, use double backticks (` `` `) for inline code per reStructuredText convention.
- Documentation pages (in `docs/`): classify as tutorial, how-to, explanation, or reference per the [Diataxis framework](https://diataxis.fr/). Use `MkDocs Material` syntax -- [admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/) (`!!! note`), tabs (`===`), code blocks with titles and highlights.
- `Mermaid` diagrams: no spaces in node IDs, quote labels with special characters, no explicit colors or styles.

---

## Dockerfiles

The repo currently has one CI `Dockerfile` ([containers/Dockerfile.test_ci](containers/Dockerfile.test_ci)). These conventions apply to new `Dockerfile`s; the CI image follows a simpler pattern.

- Multi-stage builds for production images
- Copy uv from `ghcr.io/astral-sh/uv:<version>`
- `--mount=type=cache` for pip/uv caches
- `--no-install-recommends` + `rm -rf /var/lib/apt/lists/*`
- Non-root user (`appuser`)
- `HEALTHCHECK` directives
- Order `COPY` directives for cache efficiency (deps before source)
- Comments explaining cache invalidation points

---

## Shell scripts

Current state is inconsistent; these are the target conventions for new scripts.

- Shebang: `#!/usr/bin/env bash` (not `#!/bin/bash`)
- Safety: minimum floor is `set -eu`. Use `set -euo pipefail` unless `pipefail` breaks piped-grep patterns in the specific script.
- Naming: `snake_case` for functions, `_` prefix for internal helpers
- Variables: always quote (`"$VAR"`, `"${VAR}"`), defaults via `${VAR:-default}`. Use `readonly` for variables that should not change after assignment.
- Repo root detection: `REPO_ROOT=${REPO_ROOT:-$(git rev-parse --show-toplevel)}`
- Source shared utilities from `tools/binaries/defs.sh` and `tools/binaries/common_functions.sh` where applicable
- Use `shellcheck` to lint shell scripts. When disabling a check, add `# shellcheck disable=SCXXXX` with a brief reason.

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
source "${REPO_ROOT}/tools/binaries/defs.sh"

readonly OUTPUT_DIR="${1:?Usage: $0 <output-dir>}"
```

---

## Configuration files

### YAML

- 2-space indentation
- Colon-space for key-value pairs (`: `)
- SPDX copyright headers at top
- Unquoted values unless special characters require them
- Empty line at end of file
- GitHub Actions workflows: `#` with dashes for section dividers

### TOML

- Spaces around `=` for key-value pairs
- Comments: `# comment` with inline comments for dependency pins
- Section ordering in `pyproject.toml`: `[project]`, `[dependency-groups]`, `[project.optional-dependencies]`, `[tool.uv]`, `[build-system]`, `[tool.*]`

### Makefile

- Target help format: `target-name: ## Description` (enables `make help` auto-generation)
- Tab indentation (standard Makefile)
- `.PHONY` declaration directly above each target it applies to
- Variables in `### CONFIGURATION ###` section

---

## General conventions

### Copyright headers

Every source file requires an SPDX copyright header - `make format` handles this automatically. See [tools/lint/copyright_fixer.py](tools/lint/copyright_fixer.py).

E.g., Hash-comments for `.py`, `.sh`, `.yaml`, `.yml`. HTML-comment for `.md`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
```


```markdown
<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
```

Exception: `.md` files that start with YAML frontmatter (`---`) get hash-comment headers inside the frontmatter block instead of HTML comments, since HTML comments are not valid YAML:

```markdown
---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Page title
---
```

### File endings

- Newline at EOF, no trailing whitespace (enforced by `pre-commit`)
- Line length: 120 characters for code, comments, and docstrings (configured in [ruff.toml](ruff.toml))

---

## Parting words

Be consistent. If the code around you follows a convention, follow it too -- even if this guide says otherwise. Local consistency matters more than global rules.

That said, don't use consistency as an excuse to perpetuate old patterns. When touching legacy code, migrate toward these conventions where practical.
