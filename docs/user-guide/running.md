<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Running Safe Synthesizer

Full reference for pipeline execution. For a quick first run, see
[Getting Started](getting-started.md). For parameter tables, see
[Configuration Reference](configuration.md). For environment variables, see
[Environment Variables](environment.md).

---

## Configuration Interfaces

NeMo Safe Synthesizer has two ways to run the pipeline and four-and-a-half ways to configure it.

Two ways to run:

- `safe-synthesizer` CLI -- the command-line application
- Python SDK -- the [`SafeSynthesizer`][nemo_safe_synthesizer.sdk.library_builder.SafeSynthesizer] builder, for use in scripts, notebooks, and services

Four-and-a-half ways to configure:

- YAML config file -- a portable, versionable snapshot of parameters; passed to the CLI via `--config` or loaded in the SDK with [`SafeSynthesizerParameters.from_yaml()`][nemo_safe_synthesizer.config.parameters.SafeSynthesizerParameters]
- CLI flags -- `--generation__num_records 10000`, `--privacy__dp_enabled true`; override the YAML file when both are provided
- Python SDK builder calls -- `.with_generate(num_records=10000)`, `.with_differential_privacy(dp_enabled=True)`; override the YAML file when both are used
- [Dataset registry](#dataset-registry) -- a YAML file (passed via `--dataset-registry`) that defines named datasets and their parameter overrides so you can refer to them by name in the CLI
- Environment variables (the half) -- control infrastructure only: artifact paths, logging, model cache locations, WandB mode. They do not set synthesis parameters like learning rate or record count

The asymmetry matters: YAML and environment variables are *configuration only* -- they don't invoke the pipeline. CLI and SDK are *run and configure* -- they set parameters and execute.

All configuration surfaces share the same underlying [Pydantic](https://docs.pydantic.dev/) parameter models defined in `src/nemo_safe_synthesizer/config/`. The `__` syntax used in CLI flags (e.g. `--privacy__dp_enabled true`) mirrors the nested structure of those models: `privacy` is the config section, `dp_enabled` is the field. Setting a parameter via YAML, CLI flag, or SDK call resolves to the same field in the same model.

Exactly what avenues of configuration are available, and thus how precedence is resolved, depends on how you run the pipeline. Settings are resolved in this order, from highest (first) to lowest priority (last):

- CLI: `CLI flags` > `dataset registry` > `YAML config file` > `model defaults`
- SDK: `Python SDK builder calls` > `YAML config file` > `model defaults`

See [Configuration Precedence](configuration.md#configuration-precedence) for details.

The same run, three ways -- 10,000 records with DP-SGD:

=== "CLI"

    ```bash
    safe-synthesizer run \
      --data-source data.csv \
      --generation__num_records 10000 \
      --privacy__dp_enabled true \
      --privacy__epsilon 8.0
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_generate(num_records=10000)
        .with_differential_privacy(dp_enabled=True, epsilon=8.0)
    )
    synthesizer.run()
    ```

=== "Config reference"

    ```yaml
    # config.yaml
    generation:
      num_records: 10000
    privacy:
      dp_enabled: true
      epsilon: 8.0
    ```

    ```bash
    safe-synthesizer run --config config.yaml --data-source data.csv
    ```

---

## Running the Pipeline

The pipeline runs five stages in sequence. PII replacement is on by default as a pre-processing step; disable it with `--no-replace-pii` (CLI) or `.with_replace_pii(enable=False)` (SDK).

```mermaid
flowchart LR
    data[Data Input] --> pii["PII Replacement<br/>(on by default)"]
    pii --> train["Training<br/>LoRA fine-tune"]
    train --> gen["Generation<br/>vLLM sampling"]
    gen --> eval["Evaluation<br/>SQS + DPS report"]
```

Run the full end-to-end pipeline in one step:

=== "CLI"

    ```bash
    safe-synthesizer run \
      --config config.yaml \
      --data-source data.csv \
      --artifact-path ./artifacts
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
    synthesizer = SafeSynthesizer().with_data_source("data.csv")
    synthesizer.run()

    results = synthesizer.results
    ```

You can also run stages individually:

- `safe-synthesizer run train` -- train only, saves the adapter
- `safe-synthesizer run generate` -- generate only (use `--auto-discover-adapter` or `--run-path`)
- SDK stepwise: `process_data()` → `train()` → `generate()` → `evaluate()`

## Pre-flight Validation

Use `--validate` to check your dataset and configuration before committing to
a full run. Pre-flight catches configuration mistakes, missing columns, token
budget overflows, and GPU issues in seconds -- before the pipeline downloads
models or starts training. See [`run --validate`](#run-validate) in CLI
Commands for the full reference.

---

## Using YAML Config Files

A `config.yaml` file is optional for the CLI and SDK. When omitted, model
defaults apply. When provided, CLI flags and SDK builder calls override the
values from the file.

### CLI

Pass `--config` to load a base config, then override individual fields with
`--key__subkey value` syntax:

```bash
# All defaults, no config file
safe-synthesizer run --data-source data.csv

# Config file as base, override two fields
safe-synthesizer run \
  --config config.yaml \
  --data-source data.csv \
  --training__learning_rate 0.001 \
  --generation__num_records 2000
```

### SDK

Pass a [`SafeSynthesizerParameters`][nemo_safe_synthesizer.config.parameters.SafeSynthesizerParameters] loaded from YAML as the seed, then use
`with_*` calls to override specific sections:

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
from nemo_safe_synthesizer.config import SafeSynthesizerParameters

# Load base config from file, override generation settings
config = SafeSynthesizerParameters.from_yaml("config.yaml")
synthesizer = (
    SafeSynthesizer(config)
    .with_data_source("data.csv")
    .with_generate(num_records=2000, temperature=0.8)
)
synthesizer.run()
```

`with_*` keyword arguments take precedence over whatever is in the YAML file.
Sections not mentioned in the builder call retain their values from `config`.

See [Configuration Reference -- CLI Override Syntax](configuration.md#cli-override-syntax)
for the full override precedence rules.

---

## CLI Commands

Print the installed version and exit:

```bash
safe-synthesizer --version
```

```bash
safe-synthesizer --help
```

### `run` -- Execute the Pipeline

Without a subcommand, `run` executes the full end-to-end pipeline (data processing,
PII replacement, training, generation, evaluation). PII replacement is on by default.

```bash
safe-synthesizer run --config config.yaml --data-source data.csv
```

#### Common Options

These options apply to `run` and `run generate`. Only `--data-source` is required;
all others have defaults or are optional.

| Option | Env var | Default | Description |
|--------|---------|---------|-------------|
| `--config` | `NSS_CONFIG` | (model defaults) | Path to YAML config file; omit to use all model defaults |
| `--data-source` | -- | (required) | Dataset path, URL, or name from `--dataset-registry` |
| `--artifact-path` | `NSS_ARTIFACTS_PATH` | `./safe-synthesizer-artifacts` | Base directory for all runs |
| `--run-path` | -- | -- | Explicit run directory (for `run generate`, must point to an existing trained run) |
| `--output-file` | -- | -- | Path to output CSV file |
| `--log-format` | `NSS_LOG_FORMAT` | `plain` (TTY) / `json` (non-TTY) | Console log format -- auto-detected from TTY; accepts `plain` or `json` |
| `--log-file` | `NSS_LOG_FILE` | -- | Log file path (defaults to run directory) |
| `--log-color` / `--no-log-color` | `NSS_LOG_COLOR` | auto | Colorize console output (auto-detected from TTY) |
| `--wandb-mode` | `NSS_WANDB_MODE` | `disabled` | WandB mode (`online`, `offline`, `disabled`) |
| `--wandb-project` | `NSS_WANDB_PROJECT` | -- | WandB project name |
| `--wandb-upload-evaluation-report` / `--no-wandb-upload-evaluation-report` | `NSS_WANDB_UPLOAD_EVALUATION_REPORT` | `true` | Control evaluation HTML and artifact publishing |
| `--dataset-registry` | `NSS_DATASET_REGISTRY` | -- | Dataset registry YAML path/URL |
| `-v` / `-vv` | -- | -- | Verbose logging (`-v` debug, `-vv` debug + dependencies) |

#### Synthesis Parameter Overrides

Any synthesis parameter can be overridden on the command line using
`--section__field` syntax (e.g., `--training__learning_rate 0.001`).
See [Configuration Reference -- CLI Override Syntax](configuration.md#cli-override-syntax)
for the full syntax, examples, and precedence rules.

### `run train`

Train only -- saves the adapter without generating or evaluating.

```bash
safe-synthesizer run train --config config.yaml --data-source data.csv
```

Accepts the same common options and synthesis parameter overrides as `run`.

### `run generate`

Generate only -- requires a previously trained adapter.

```bash
safe-synthesizer run generate \
  --config config.yaml \
  --data-source data.csv \
  --auto-discover-adapter

# Or specify an explicit run path
safe-synthesizer run generate \
  --config config.yaml \
  --data-source data.csv \
  --run-path ./safe-synthesizer-artifacts/myconfig---mydata/2026-01-15T12:00:00
```

| Option | Description |
|--------|-------------|
| `--auto-discover-adapter` | Find the latest trained adapter in the artifact directory |
| `--run-path` | Explicit path to a previous run's output directory |
| `--wandb-resume-job-id` | WandB run ID to resume (or path to file containing the ID) |

Accepts the same common options and synthesis parameter override syntax as `run`.

!!! note "Override scope on resume"
    `run generate` reloads the trained run's saved configuration. Only
    `generation` and `evaluation` overrides (and `emit_telemetry`) supplied via
    `--config` or `--section__field` flags take effect; fields you do not set
    keep their saved values. `training`, `data`, `privacy`, and `time_series`
    are always inherited from the trained run and cannot be changed at generate
    time, since they describe how the adapter was produced.

### `run --validate`

Validate your dataset and configuration without starting training or
downloading model weights. The tokenizer and model config may still be
fetched from the Hugging Face cache (needed for token-budget checks).
Available on `run` and `run train`.

```bash
safe-synthesizer run --data-source data.csv --validate
```

Pre-flight runs the following core checks, grouped by stage. Stages
execute in order (`config` → `dataframe` → `metadata` → `advisory`).

| Check name | Stage | What it validates |
|-------|-------|-------------------|
| `gpu.cuda` | config | PyTorch is importable and a CUDA GPU is visible |
| `env.inference` | config | Inference endpoint settings for LLM-assisted features: `NSS_INFERENCE_KEY` is set, `NSS_INFERENCE_MODEL` is non-empty, and `NSS_INFERENCE_ENDPOINT` is a valid http(s) URL (warnings only; not required for PII replacement) |
| `env.hf_model_availability` | config | The pretrained model reference is usable locally or can be fetched from Hugging Face; warns about a missing HF token only when online HF access may be needed |
| `dataset.size` | dataframe | Training split meets the hard minimum row count |
| `columns.groupby` | dataframe | `group_training_examples_by` column is present and has no nulls |
| `columns.orderby` | dataframe | `order_training_examples_by` column is present |
| `columns.pseudo` | dataframe | Input does not use the reserved `__nss_sequence_id` column name |
| `columns.constant` | dataframe | No column is constant (warning only) |
| `timeseries.timestamp` | dataframe | Timestamp column is present and has no nulls (time-series mode) |
| `timeseries.shape` | dataframe | Timestamp formats parse cleanly and groups have matching lengths, intervals, starts, and stops (time-series mode) |
| `gpu.vram` | metadata | Free VRAM headroom for the chosen model, quantization load mode, and per-device batch size; emits `low_vram` as a warning and `vram_exceeds_capacity` as an error when the estimate is far above capacity |
| `token_budget` | metadata | Schema prompt, sampled records, and top groups each fit in the model's context window |
| `dataset.row_count` | advisory | Training split is above a comfort threshold (warning only) |
| `training.oversampling` | advisory | Sampling fraction is not extreme (warning only) |

If a non-advisory check produces an error, later checks that declare it
as a `requires` dependency are marked `skipped`. Advisory-stage errors
never gate other checks. For example, a missing group-by column fails
`columns.groupby`; `token_budget` still runs (its schema and sampled-
record branches don't depend on the group-by column), but the
per-group budget branch is skipped automatically when the column is
absent.

Pre-flight runs against the training split produced by `Holdout`, not
the full input dataset. Row-count, oversampling, and
token-budget messages all report on the training partition the model
will actually see. The runtime-info block at the top of the report shows
both the input dataset size and the training-split size so the scope of
each check is unambiguous.

Token budget checks use the same budget computation as the training
assembler, so the numbers pre-flight reports match what assembly will
enforce. Pre-flight samples rows and top groups for tokenization rather
than scanning the full dataset, so a long-tail outlier outside the
sample can still trip a context-length error at assembly time -- see
the best-effort caveat below.

!!! warning "`--validate` is best-effort"

    A clean `--validate` run does not guarantee a full run will succeed.
    Known gaps:

    - PII replacement is skipped in validate mode, so the training split
      that preflight sees is the *pre-replacement* data; replacement text
      can be shorter or longer than the original and shift token budgets.
      If PII replacement alters time-series timestamp or group columns, a
      later training-time invariant check can still fail.
    - Token-budget checks sample rows and top groups -- a long-tail
      outlier outside the sample can still exceed the budget at assembly.
    - VRAM headroom includes a coarse bf16 compute activation term
      (`batch_size` x `metadata.max_seq_length` x width x depth) plus base
      weights using the configured quantization mode; LoRA adapters,
      optimizer state, and attention blocks beyond that shape are not modeled
      tightly. Full training may still OOM.

    Treat `--validate` as a quick fail-fast gate, not a full-run guarantee.

#### Interpreting the Output

The CLI report is split into labeled sections:

- runtime info -- model, input dataset size, training-split size (the partition pre-flight actually checked)
- validation checks -- per-check results with status indicators (`✓` pass, `✗` error, `⚠` warning) and issue codes
- output locations -- artifact directory tree with resolved config and log file paths
- next steps -- a ready-to-copy `safe-synthesizer run` command using the resolved config

Errors block the pipeline. Warnings are informational -- review them, but they
do not prevent a run. When all checks pass, the output includes a follow-up
command:

```text
Run with the resolved configuration:
  safe-synthesizer run --data-source data.csv \
    --config ./safe-synthesizer-artifacts/<project>/<timestamp>/safe-synthesizer-config.yaml
```

The resolved config captures all auto-resolved parameters (rope scaling,
learning rate, record sampling), so the subsequent run uses deterministic
settings.

#### SDK Equivalent

In the SDK, call `process_data(check_only=True)`:

```python
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

nss = SafeSynthesizer().with_data_source("data.csv")
try:
    nss.process_data(check_only=True)
except ParameterError:
    # Preflight surfaced at least one error; the report is populated
    # before the raise so you can inspect it either way.
    pass

report = nss.preflight_report
if report is not None:
    print(f"{len(report.errors)} errors, {len(report.warnings)} warnings")
    for issue in report.errors + report.warnings:
        print(f"[{issue.severity}] {issue.code}: {issue.message}")
```

`process_data(check_only=True)` raises `ParameterError` if any check
produces errors. `nss.preflight_report` is assigned *before* the raise
when the failure comes from preflight itself, so the `try` block above
will always leave a structured report behind for the error path.

Failures that happen *before* preflight runs -- for example a
group-by column that does not exist in the input, which is checked
up-front by `process_data` -- also raise `ParameterError` but leave
`nss.preflight_report` as `None`; always guard on
`report is not None` before indexing into it.

On the CLI, `run --validate` wraps the same call: if preflight
surfaces errors, the full Rich report is rendered before the process
exits non-zero, so users see the same detail they would on a passing
validation rather than only a traceback.

#### Issue Codes

Each issue carries a short code (e.g., `no_gpu`, `schema_exceeds_context`,
`dataset_small`) for programmatic matching. The full table of codes and
severities is in [Troubleshooting -- Pre-flight Validation Codes](troubleshooting.md#pre-flight-validation-codes).

### `artifacts clean`

Delete artifacts from a previous run:

```bash
safe-synthesizer artifacts clean --artifact-path ./safe-synthesizer-artifacts
safe-synthesizer artifacts clean --caches-only   # training cache only
safe-synthesizer artifacts clean --dry-run        # preview what would be deleted
```

| Option | Description |
|--------|-------------|
| `--artifact-path` | Path to artifact directory (defaults to `./safe-synthesizer-artifacts`) |
| `--dry-run` | Preview deletions without actually deleting |
| `--caches-only` | Only delete the training cache, keep everything else |
| `--force` | Skip confirmation prompt |

---

## Data Input

Provide your dataset as a file path, URL, DataFrame (SDK), or dataset
registry name.

Data source options:

- CLI / dataset registry: `--data-source data.csv` -- supports `.csv`, `.json`, `.jsonl`, `.parquet`, `.txt`
- URL: `--data-source https://example.com/data.csv`
- DataFrame (SDK): `.with_data_source(df)` -- supports any format you can load into pandas
- CSV path (SDK): `.with_data_source("data.csv")` -- loaded via `pd.read_csv`; for non-CSV formats, load into a DataFrame first
- Dataset registry name: `--data-source my_dataset` (with `--dataset-registry registry.yaml`)

### Grouping and Ordering

Use `data.group_training_examples_by` to group records by a column (e.g.,
customer ID) so related rows are trained together. Use
`data.order_training_examples_by` to sort within groups (requires group_by).

!!! info "When to use grouped mode"
    Grouping is recommended when there is a natural ordering within each
    group -- i.e., `data.order_training_examples_by` points to a valid
    ordering field such as a date or sequence number. If your data has no
    meaningful intra-group order, tabular mode with shuffled records is
    usually sufficient.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --data__group_training_examples_by customer_id \
      --data__order_training_examples_by transaction_date \
      --data-source transactions.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("transactions.csv")
        .with_data(
            group_training_examples_by="customer_id",
            order_training_examples_by="transaction_date",
        )
    )
    ```

=== "Config reference"

    ```yaml
    data:
      group_training_examples_by: "customer_id"
      order_training_examples_by: "transaction_date"
    ```

!!! info "What the model sees"

    With grouping enabled, each training example is tokenized as:

    ```text
    [schema prompt] <BOS> group1-record1
    group1-record2 <EOS> <BOS> group2-record1
    group2-record2 <EOS>
    ```

    Here `<BOS>` and `<EOS>` represent the model's begin-of-sequence and
    end-of-sequence tokens; the exact strings are taken from the selected
    model's metadata and may differ across model families.

    `data.max_sequences_per_example` controls how many groups are packed
    into a single example (default: `"auto"`, which resolves to 10 without
    DP). Fewer groups per example means more training examples overall.
    See [Example Generation](../developer-guide/example-generation.md) for a full walkthrough.

### Dataset Registry

Define named datasets in a YAML file to reference them by name:

```yaml
base_url: "/data/datasets"
datasets:
  - name: "customer_transactions"
    url: "customers/transactions.csv"
    overrides:
      data:
        group_training_examples_by: "customer_id"
```

```bash
safe-synthesizer run --dataset-registry registry.yaml --data-source customer_transactions
```

See [Configuration Reference -- Data](configuration.md#data) for the full parameter table.

---

## PII Replacement

Enabled-by-default stage that runs before training. It builds a *replacement plan*
from the dataframe -- which columns hold PII, which entity type each holds, and which
columns describe the same person -- then substitutes values, drawing one synthetic
identity per person so a row's name, email, and address stay consistent. That reduces
exposure of detected PII; heuristic discovery can miss columns or mistype them, so
review `pii_replacement_plan.yaml` before training. Values replaced in structured
columns are also propagated into free-text columns, so notes agree with the columns
they describe. In heuristic mode, free text is scanned only when at least one
structured entity column was identified; see
[PII Replacement](../product-overview/pii_replacement.md) for that gate and for
how `person.backend: faker` omits `ethnic_background` matching.

Discovery reads column names, values, and dtypes; no inference endpoint is required.
PII replacement is on by default in both the CLI and SDK, so no config flag is needed
to enable it. See [PII Replacement](../product-overview/pii_replacement.md) for the
detection signals, entity vocabulary, and plan format.

!!! tip "Skip PII replacement"
    If your dataset does not contain PII, you may disable this stage to reduce pipeline
    runtime:

    - CLI: `--no-replace-pii`
    - SDK: `.with_replace_pii(enable=False)`

=== "CLI"

    Default (PII on, no config needed):

    ```bash
    safe-synthesizer run --data-source data.csv
    ```

    Override individual settings with the `__` syntax:

    ```bash
    safe-synthesizer run --data-source data.csv \
      --replace_pii__person__backend faker \
      --replace_pii__replacement__locale en_GB \
      --replace_pii__replacement__seed 42
    ```

    A replacement plan cannot be expressed as a flag -- put it in a config file and
    pass it with `--config` (see the Config reference tab).

=== "SDK"

    PII replacement is on by default -- no `with_replace_pii()` call is needed
    for the standard case.  Call it only to customize the config or to disable:

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    # Default: PII on, no call needed
    synthesizer = SafeSynthesizer().with_data_source("data.csv").with_train()

    # Customize: Faker personas, a fixed seed, and an edited plan
    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_replace_pii(
            person={"backend": "faker"},
            replacement={"locale": "en_US", "seed": 42},
            replacement_plan="pii_replacement_plan.yaml",
        )
        .with_train()
        .with_generate(num_records=5000)
    )
    ```

    Partial overrides merge with the defaults of
    [`PiiReplacerConfig`][nemo_safe_synthesizer.config.replace_pii.PiiReplacerConfig],
    so you only pass the fields you are changing.

=== "Config reference"

    ```yaml
    replace_pii:
      # "auto_discovery" (default), a path to a plan file, or an inline plan
      replacement_plan: pii_replacement_plan.yaml
      replacement:
        locale: en_US
        seed: 42
      person:
        # managed (default) or faker
        backend: faker
    ```

    Every field has a default, so `replace_pii: {}` behaves like the default run.
    Each run writes its discovered plan to `pii_replacement_plan.yaml` in the run
    directory; edit that file and point `replacement_plan` at it to take control of
    which columns are replaced and which belong to the same persona.

### Managed persona assets

The default `person.backend: managed` samples synthetic identities from curated
[Nemotron-Personas](https://huggingface.co/collections/nvidia/nemotron-personas)
parquet files packaged for Data Designer on NGC (not the Hugging Face narrative
exports, which omit name fields). Download those NGC resources with the NGC CLI
and install them under the managed-assets root.

If the locale file is missing, preflight warns (`pii_managed_assets_missing`) and
replacement falls back to Faker for that run. Use `person.backend: faker` when you
want that path on purpose and need no downloads.

Supported managed locales (filename must match `replace_pii.replacement.locale`):

| Locale | Region | NGC package | Faker draws |
|--------|--------|-------------|-------------|
| `en_US` | United States | `nemotron-personas-dataset-en_us` | same locale |
| `en_IN` | India (English) | `nemotron-personas-dataset-en_in` | same locale |
| `en_SG` | Singapore (English) | `nemotron-personas-dataset-en_sg` | falls back to `en_US` |
| `fr_FR` | France (French) | `nemotron-personas-dataset-fr_fr` | same locale |
| `hi_Deva_IN` | India (Devanagari script) | `nemotron-personas-dataset-hi_deva_in` | falls back to `hi_IN` |
| `hi_Latn_IN` | India (Latin script) | `nemotron-personas-dataset-hi_latn_in` | falls back to `hi_IN` |
| `ja_JP` | Japan | `nemotron-personas-dataset-ja_jp` | same locale |
| `ko_KR` | South Korea | `nemotron-personas-dataset-ko_kr` | same locale |
| `pt_BR` | Brazil (Portuguese) | `nemotron-personas-dataset-pt_br` | same locale |

Managed apply still constructs Faker for standalone identifiers, middle names, and
missing-asset fallback even when the parquet loads. Locales without a Faker
provider (`en_SG`, `hi_Deva_IN`, `hi_Latn_IN`) keep parquet sampling under the
configured locale and remap Faker draws as above; preflight warns
(`pii_managed_faker_locale_fallback`).

#### 1. Install and configure the NGC CLI

1. Obtain an NGC API key from [NVIDIA GPU Cloud](https://ngc.nvidia.com/).
2. Install the [NGC CLI](https://org.ngc.nvidia.com/setup/installers/cli).
3. Configure credentials (creates `~/.ngc/config`):

```bash
ngc config set
```

Browse available packages in the
[NGC Nemotron-Personas catalog](https://catalog.ngc.nvidia.com/search?orderBy=scoreDESC&query=nemotron+personas).

#### 2. Download locales with the NGC CLI

Run these from a working directory where you can unpack the downloads (for example
your home directory). Download only the locales you need:

```bash
# United States (matches the default replace_pii.replacement.locale)
ngc registry resource download-version "nvidia/nemotron-personas/nemotron-personas-dataset-en_us"

# Optional additional locales
ngc registry resource download-version "nvidia/nemotron-personas/<NGC package>"
```

#### 3. Install parquet files as `{locale}.parquet`

Safe Synthesizer loads
`{managed_assets_root}/datasets/{locale}.parquet` (for example
`~/.data-designer/managed-assets/datasets/en_US.parquet`). Create the directory and
install each downloaded package under that exact name:

```bash
mkdir -p ~/.data-designer/managed-assets/datasets

# Adjust the left-hand glob if your NGC CLI unpacked a versioned folder name.
mv nemotron-personas-dataset-en_us*/*.parquet \
  ~/.data-designer/managed-assets/datasets/en_US.parquet

# Examples for other locales:
# mv nemotron-personas-dataset-ja_jp*/*.parquet \
#   ~/.data-designer/managed-assets/datasets/ja_JP.parquet
```

Override the root with `NSS_MANAGED_ASSETS_PATH` or
`replace_pii.person.managed_assets_path` when assets live elsewhere (for example a
shared cluster mount). See
[Environment Variables -- NSS_MANAGED_ASSETS_PATH](environment.md#nss_managed_assets_path).

### LLM-assisted discovery

Discovery is heuristic in this release and needs no inference endpoint or API key.
`replace_pii.llm_enhancement: true` is reserved for a future release and currently
raises `ParameterError`.

The `NSS_INFERENCE_ENDPOINT`, `NSS_INFERENCE_KEY`, and `NSS_INFERENCE_MODEL`
variables are plumbed for that future work and do not affect PII replacement today.
Preflight still reports an `env.inference` warning when `NSS_INFERENCE_KEY` is unset;
it is advisory and safe to ignore.

See [Configuration Reference -- Replacing PII](configuration.md#replacing-pii) for the full parameter reference.

---

## Training

Fine-tunes a pretrained LLM on your data using LoRA (Low-Rank Adaptation).
LoRA inserts a small set of trainable adapter weights into the frozen pretrained
model. Only the adapter is updated during training, which keeps VRAM
requirements low and produces a compact artifact that can be reused for
generation without re-training.

Training uses the HuggingFace backend -- LoRA fine-tuning via PEFT with
4-bit/8-bit quantization support and optional differential privacy (DP-SGD)
via [Opacus](https://opacus.ai/).

Three models have been extensively tested:

| Family | HuggingFace ID |
|--------|----------------|
| SmolLM3 (default) | `HuggingFaceTB/SmolLM3-3B` |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

We recommend you start with the default, `HuggingFaceTB/SmolLM3-3B`. However, depending on your use case, you may find a different model to be a better fit.

Based on testing, some trade-offs identified compared to SmolLM3 on average:

- TinyLlama runs ~17% faster, while Mistral takes ~2x as long to run.
- Mistral has ~6% increase in valid record fraction, while TinyLlama has ~7% decrease.
- Mistral has ~5% higher job completion rate and TinyLlama has ~3% higher.
- Mistral is comparable to SmolLM3 in Data Privacy Score, while TinyLlama has ~0.1 point decrease.
- All 3 have comparable Synthetic Quality Scores.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --training__learning_rate 0.001 \
      --training__batch_size 4 \
      --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_train(learning_rate=0.001, batch_size=4)
    )
    ```

=== "Config reference"

    ```yaml
    training:
      pretrained_model: "HuggingFaceTB/SmolLM3-3B"
      learning_rate: 0.001
      batch_size: 4
    ```

### Quantization

Enabling quantization reduces VRAM consumption at the cost of some numerical
precision. Set `training.quantize_model` to `true` and choose a bit width with
`training.quantization_bits`.

| Setting | VRAM | Precision | Speed | Notes |
|---------|------|-----------|-------|-------|
| No quantization | Highest | Full | Baseline | Use when VRAM is not a constraint |
| 8-bit | ~50% reduction | Near-full | Slightly slower | Good balance for most cases |
| 4-bit | ~75% reduction | Reduced | Faster | Use when VRAM is tight; may affect output quality |

=== "CLI"

    ```bash
    safe-synthesizer run \
      --training__quantize_model true \
      --training__quantization_bits 4 \
      --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_train(quantize_model=True, quantization_bits=4)
    )
    ```

=== "Config reference"

    ```yaml
    training:
      quantize_model: true
      quantization_bits: 4
    ```

### Attention Backends

`training.attn_implementation` controls which attention kernel is used when
loading the model. The default is `sdpa`, which uses PyTorch scaled
dot-product attention and does not require a separate attention package.

Common values:

- `sdpa`: PyTorch scaled dot-product attention -- default and broadest compatibility
- `kernels-community/vllm-flash-attn3`: Flash Attention 3 (requires `kernels` package and a compatible prebuilt kernel)
- `flash_attention_2`: Flash Attention 2 (requires `flash-attn` package)
- `flash_attention_3`: Flash Attention 3 (requires `flash-attn-3` support)
- `eager`: standard PyTorch attention -- useful for debugging

!!! note "Training vs generation attention backends"
    The training attention backend (`training.attn_implementation`) and the
    generation attention backend (`generation.attention_backend` /
    `VLLM_ATTENTION_BACKEND`) are independent settings.

### Differential Privacy

Differential privacy (DP) provides a formal bound on what an adversary can
learn about any individual record. Safe Synthesizer implements Differentially Private Stochastic Gradient Descent (DP-SGD) via [Opacus](https://opacus.ai/).

=== "CLI"

    ```bash
    safe-synthesizer run \
      --privacy__dp_enabled true \
      --privacy__epsilon 8.0 \
      --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_differential_privacy(dp_enabled=True, epsilon=8.0)
    )
    ```

=== "Config reference"

    ```yaml
    privacy:
      dp_enabled: true
      epsilon: 8.0
    ```

Compatibility constraints when DP is enabled:

- `data.max_sequences_per_example` must be `1` (or `"auto"`, which resolves to `1` when DP is enabled)
- Gradient checkpointing is disabled (incompatible with Opacus)

!!! note "DP training trade-offs"
    DP training is slower and typically requires more epochs to reach the same
    loss as non-DP training. Start with `epsilon: 8.0` -- a common, practical
    threshold -- and lower it only if your privacy requirements demand it.
    Very low epsilon values (e.g., below 1.0) significantly degrade model
    utility.

See [Configuration Reference -- Differential Privacy](configuration.md#differential-privacy) for the full parameter table.

---

## Generation

Produces synthetic records using the trained LoRA adapter via vLLM. The
generation stage runs a sampling loop: the model generates batches of records,
each record is validated against the training dataset schema (correct columns,
correct types, no malformed values), and valid records accumulate until
`num_records` is reached. If too many consecutive batches produce mostly invalid
records, the loop stops early.

```mermaid
flowchart TD
    start([Start]) --> batch[Generate batch]
    batch --> validate["Validate records\nagainst schema"]
    validate --> accum[Accumulate\nvalid records]
    accum --> enough{Enough\nrecords?}
    enough -- No --> patience{Patience\nexceeded?}
    patience -- No --> batch
    patience -- Yes --> stop([Stop early])
    enough -- Yes --> done([Done])
```

=== "CLI"

    ```bash
    safe-synthesizer run \
      --generation__num_records 5000 \
      --generation__temperature 0.7 \
      --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_generate(num_records=5000, temperature=0.7)
    )
    ```

=== "Config reference"

    ```yaml
    generation:
      num_records: 5000
      temperature: 0.7
    ```

### Structured Generation

Set `generation.structured_generation.enabled` to `true` to constrain the model's
output toward the dataset schema. This usually reduces the fraction of invalid
records, typically at the cost of reducing the quality of the generated records.
Use it when the pipeline struggles to produce valid records.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --generation__structured_generation__enabled true \
      --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_generate(structured_generation={"enabled": True})
    )
    ```

    Pass `structured_generation` as a nested dict or
    `StructuredGenerationParameters` object. Do not use a bare `enabled=True`
    shortcut for structured generation.

=== "Config reference"

    ```yaml
    generation:
      structured_generation:
        enabled: true
        schema_method: "auto"
    ```

- `"auto"`: picks `"structural_tag"` when `structured_generation.backend` is `"auto"` or `"xgrammar"`, otherwise `"regex"`.
- `"structural_tag"`: uses XGrammar Structural Tag to compose schema-constrained JSONL output.
- `"regex"`: constructs a custom regex from the dataset schema. More comprehensive but slower.
- `"json_schema"`: passes a JSON Schema to the backend. Faster, but may miss edge cases.

### Stopping Conditions

Generation stops early when too many consecutive batches produce mostly invalid
records. `generation.patience` controls how many bad batches to tolerate;
`generation.invalid_fraction_threshold` defines what counts as "bad." If the
pipeline stops early, check the generation logs for the invalid record
fraction per batch.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --generation__patience 5 \
      --generation__invalid_fraction_threshold 0.6 \
      --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_generate(patience=5, invalid_fraction_threshold=0.6)
    )
    ```

=== "Config reference"

    ```yaml
    generation:
      patience: 5
      invalid_fraction_threshold: 0.6
    ```

!!! tip "Early stopping"
    If the pipeline stops early due to patience, try enabling
    `generation.structured_generation.enabled: true` to constrain outputs to
    the dataset schema, or lower `temperature` to reduce the chance of malformed
    records.

See [Configuration Reference -- Generation](configuration.md#generation) for the full parameter table.

---

## Evaluation

Measures quality and privacy of synthetic data and produces an HTML report
with interactive visualizations. Scores are from 0-10, and higher is better. Two composite scores are reported:

- SQS (Synthetic Quality Score) -- composite quality score with five subscores:
    - Column Correlation Stability -- measures the correlation across every combination of two numeric and categorical columns
    - Deep Structure Stability -- compares numeric and categorical columns in the training and synthetic data using Principal Component Analysis (PCA)
    - Column Distribution Stability -- measures the distribution of each numeric and categorical column
    - Text Structure Similarity -- measures the sentence, word, and character counts for text columns
    - Text Semantic Similarity -- measures whether the semantic meaning in text columns held after synthesizing
- DPS (Data Privacy Score) -- composite privacy score with three subscores:
    - Membership Inference Protection -- measures whether a model trained on the data can distinguish training records from held-out records
    - Attribute Inference Protection -- measures whether an attacker can infer a sensitive attribute from quasi-identifiers in the synthetic data
    - PII Replay Detection -- checks whether PII from training appears in synthetic data

See [Evaluation](../product-overview/evaluation.md) for details on score
interpretation.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --evaluation__mia_enabled false \
      --evaluation__aia_enabled false \
      --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_evaluate(mia_enabled=False, aia_enabled=False)
    )
    ```

=== "Config reference"

    ```yaml
    evaluation:
      mia_enabled: true
      aia_enabled: true
      pii_replay_enabled: true
    ```

### Disable Evaluation

To skip evaluation entirely (e.g., for faster iteration during development):

=== "CLI"

    ```bash
    safe-synthesizer run \
      --evaluation__enabled false \
      --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("data.csv")
        .with_evaluate(enabled=False)
    )
    ```

=== "Config reference"

    ```yaml
    evaluation:
      enabled: false
    ```

See [Configuration Reference -- Evaluation](configuration.md#evaluation) for the full parameter table.

---

## Time Series Mode

!!! warning "Experimental"
    Time series synthesis is an experimental feature. APIs and behavior may
    change between releases.

Enable time series mode by setting `time_series.is_timeseries: true` and
providing timestamp configuration. Use `data.group_training_examples_by` to
group records by entity (e.g., sensor ID) and `data.order_training_examples_by`
to sort within groups.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --time_series__is_timeseries true \
      --time_series__timestamp_column timestamp \
      --time_series__timestamp_interval_seconds 60 \
      --data__group_training_examples_by sensor_id \
      --data-source sensor_data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = (
        SafeSynthesizer()
        .with_data_source("sensor_data.csv")
        .with_time_series(
            is_timeseries=True,
            timestamp_column="timestamp",
            timestamp_interval_seconds=60,
        )
        .with_data(
            group_training_examples_by="sensor_id",
            order_training_examples_by="timestamp",
        )
    )
    ```

=== "Config reference"

    ```yaml
    time_series:
      is_timeseries: true
      timestamp_column: "timestamp"
      timestamp_interval_seconds: 60
    data:
      group_training_examples_by: "sensor_id"
      order_training_examples_by: "timestamp"
    ```

See [Configuration Reference -- Time Series](configuration.md#time-series) for the full parameter table.
See [Troubleshooting -- Time Series](troubleshooting.md#time-series) for common issues.

!!! note "How time-series examples are assembled"
    Each training example contains records from a single group in
    chronological order. The model learns to continue a sequence --
    not to produce independent records. See
    [Example Generation](../developer-guide/example-generation.md) for assembly details.

---

## Run Individual Stages

### Train only

=== "CLI"

    ```bash
    safe-synthesizer run train --config config.yaml --data-source data.csv
    ```

=== "SDK"

    ```python
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

    synthesizer = SafeSynthesizer().with_data_source("data.csv")
    synthesizer.process_data()
    synthesizer.train()
    ```

### Generate only

Use `--auto-discover-adapter` to find the latest trained adapter, or
`--run-path` for an explicit location. See [`run generate`](#run-generate) in
the CLI Commands section for all options.

=== "CLI"

    ```bash
    safe-synthesizer run generate \
      --config config.yaml \
      --data-source data.csv \
      --auto-discover-adapter
    ```

=== "SDK"

    ```python
    from pathlib import Path
    from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer
    from nemo_safe_synthesizer.config import SafeSynthesizerParameters
    from nemo_safe_synthesizer.cli.artifact_structure import Workdir

    config = SafeSynthesizerParameters.from_yaml("config.yaml")
    workdir = Workdir.from_path(
        Path("./safe-synthesizer-artifacts/myconfig---mydata/2026-01-15T12:00:00")
    )
    synthesizer = SafeSynthesizer(config, workdir=workdir)
    synthesizer.load_from_save_path()
    synthesizer.process_data()
    synthesizer.generate()
    synthesizer.evaluate()
    synthesizer.save_results()
    ```

### Stepwise execution (SDK)

For full control, call each stage individually:

```python
synthesizer = (
    SafeSynthesizer()
    .with_data_source(df)
    .process_data()
    .train()
    .generate()
    .evaluate()
)

results = synthesizer.results
synthesizer.save_results()
```

---

## Artifacts and Output

Each run writes to a directory named `<config-stem>---<dataset-stem>/<run_name>`
under the artifact path. The config and dataset stems are derived from the
filenames you pass to `--config` and `--data-source`, making it easy to identify runs
at a glance. `<run_name>` defaults to an ISO 8601 timestamp (e.g., `2026-01-15T12:00:00`).

To use an explicit output directory (skipping the auto-generated
`<config>---<dataset>/<run_name>` structure), pass `--run-path`:

```bash
safe-synthesizer run --config config.yaml --data-source data.csv --run-path ./my-run
```

```text
<artifact-path>/<config>---<dataset>/<run_name>/
├── train/
│   ├── safe-synthesizer-config.json
│   └── adapter/                     # trained PEFT adapter
├── generate/
│   ├── logs.jsonl                   # generate-only workflow
│   ├── info.json                    # generate-only workflow
│   ├── synthetic_data.csv
│   ├── evaluation_report.html
│   └── evaluation_metrics.json      # machine-readable metrics
├── dataset/
│   ├── training.csv
│   ├── test.csv
│   ├── validation.csv               # when training.validation_ratio > 0
│   └── transformed_training.csv     # when PII replacement transforms the data
└── logs/
    └── <phase>.jsonl                # e.g. end_to_end.jsonl or train.jsonl
```

Key outputs:

- `generate/synthetic_data.csv`: the synthetic dataset
- `generate/evaluation_report.html`: quality and privacy report
- `generate/evaluation_metrics.json`: machine-readable evaluation scores and timing
- `train/adapter/`: LoRA weights for resuming generation
- `train/safe-synthesizer-config.json`: resolved config snapshot

!!! tip "Clean up artifacts"
    Adapter weights and training caches can consume significant disk space
    during iterative development. Run `safe-synthesizer artifacts clean` to
    remove them when no longer needed. Use `--caches-only` to keep the adapter
    but reclaim training cache space.

### SDK Results Access

`run()` automatically saves `synthetic_data.csv`, `evaluation_report.html`,
and `evaluation_metrics.json` to the artifacts directory unless an
`output_file` override is provided. For stepwise execution, call
`save_results()` explicitly after `evaluate()`.

```python
results = synthesizer.results
df = results.synthetic_data
summary = results.summary
# synthesizer.save_results()  # only needed for stepwise execution; run() saves automatically
```

### Cleaning Up

See [`artifacts clean`](#artifacts-clean) in the CLI Commands section for options.

---

## Running in Offline Environments

Pre-cache models by running once with internet access, then set
`HF_HUB_OFFLINE=1` in your target environment. Export it before launching
`safe-synthesizer` (or pass `--disable-huggingface-remote`) -- huggingface_hub
reads the value once at import time, so setting it after the process starts has
no effect. For detailed cache setup and environment variables (`HF_HOME`,
`HF_HUB_OFFLINE`, `VLLM_CACHE_ROOT`), see
[Environment Variables -- Hugging Face cache and offline](environment.md#hugging-face-cache-and-offline).

For offline-specific errors, see [Program Runtime](troubleshooting.md).

---

## Logging and Experiment Tracking

### Log Format

| Method | Setting |
|--------|---------|
| CLI | `--log-format json` or `--log-format plain` |
| Environment | `NSS_LOG_FORMAT=json` |

The format auto-detects from the terminal: `plain` when stdout is a TTY, `json` otherwise.

=== "Plain"

    Human-readable columns separated by `|`. Used by default in interactive terminals.

    ```text
    2026-01-15T12:03:42.001 | Nemo Safe Synthesizer | user    | info  | training.py:TrainingBackend.train:87
    Training complete

    2026-01-15T12:03:42.105 | Nemo Safe Synthesizer | runtime | info  | generation.py:VllmBackend._generate:214
    Batch complete: {'valid': 48, 'invalid': 2}
    ```

=== "JSON"

    One JSON object per line. Used by default in non-TTY environments (CI, containers, log aggregators).

    ```json
    {"timestamp": "2026-01-15T12:03:42.001000Z", "level": "info", "filename": "training.py", "lineno": 87, "category": "user", "message": "Training complete"}
    {"timestamp": "2026-01-15T12:03:42.105000Z", "level": "info", "filename": "generation.py", "lineno": 214, "category": "runtime", "message": "Batch complete", "valid": 48, "invalid": 2}
    ```

Log categories in both formats:

- `user` -- user-relevant progress and results (training complete, generation done)
- `runtime` -- internal operational details (memory, timings, batch stats)
- `system` -- system-level events (startup, config loaded)
- `backend` -- logs from dependencies (vLLM, HuggingFace, etc.)

Verbosity: `-v` for debug, `-vv` for debug + dependencies.

### WandB Integration

WandB is configured via CLI flags or environment variables -- not in the YAML
config file.

=== "CLI"

    ```bash
    safe-synthesizer run \
      --config config.yaml \
      --data-source data.csv \
      --wandb-mode online \
      --wandb-project my-experiments
    ```

=== "SDK"

    ```python
    import os
    import wandb

    os.environ["WANDB_API_KEY"] = "your-api-key"  # pragma: allowlist secret
    wandb.init(project="my-experiments", mode="online")

    synthesizer = SafeSynthesizer().with_data_source("data.csv")
    synthesizer.run()
    ```

    Unlike the CLI, the SDK does not auto-initialize WandB. You must call
    `wandb.init(...)` before `synthesizer.run()`.

=== "Environment variables"

    ```bash
    export WANDB_API_KEY="your-api-key"  # pragma: allowlist secret
    export WANDB_PROJECT="my-experiments"
    export NSS_WANDB_MODE="online"
    ```

    These environment variables are read by the CLI only. SDK users must
    call `wandb.init(...)` explicitly.

#### Evaluation scorecard and report upload

For CLI-managed runs, final scalar `eval/*`, `gen/*`, timing, failure, and
vLLM-completion values update the W&B run summary rather than creating history
points. Existing training curves remain W&B history. After results are saved,
the CLI publishes an `evaluation/scorecard` table panel containing final
`eval/*` values.

When W&B is active, the CLI publishes the HTML evaluation report to the
user-configured project as the `evaluation/report` panel and the
`evaluation-report` artifact. To skip the HTML report panel and evaluation-report
artifact while retaining summary metrics and the evaluation scorecard, pass:

    safe-synthesizer run \
      --config config.yaml \
      --data-source data.csv \
      --wandb-mode online \
      --no-wandb-upload-evaluation-report

or set `NSS_WANDB_UPLOAD_EVALUATION_REPORT=false`. When enabled, the artifact
is named `safe-synthesizer-evaluation-report-<run-id>` and may contain
`evaluation_report.html` and `evaluation_metrics.json` when those files are
available. SDK callers retain scalar `log_wandb()` behavior and do not upload
evaluation media automatically.

For parameter precedence (CLI flags vs environment variables vs YAML), see
[Environment Variables -- Precedence](environment.md#precedence).
