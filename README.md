# Nemo Safe Synthesizer

This package makes synthetic data, safely.

## Installation

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) - Python package manager (>=0.9.14, <0.10.0)
- Git

### Quick Start

Bootstrap development tools (installs `uv`, `ruff`, `ty`, `yq`, and more):

```bash
make bootstrap-tools
```

Then bootstrap the project package with your desired extras - likely `cpu|cuda` .

```bash
# CPU-only (for development on Linux without GPU, or macOS)
make bootstrap-nss cpu

# CUDA 12.8 (for Linux with NVIDIA GPU)
make bootstrap-nss cuda

# Engine only (synthesis engine dependencies, no torch/training)
make bootstrap-nss engine

# Dev only (minimal dev dependencies, no engine or torch)
make bootstrap-nss dev
```

## Running

Run the CLI using `safe-synthesizer`:

```bash
> safe-synthesizer --help
Usage: safe-synthesizer [OPTIONS] COMMAND [ARGS]...

  NeMo Safe Synthesizer command-line interface. This application is used to
  run the Safe Synthesizer pipeline. It can be used to train a model, generate
  synthetic data, and evaluate the synthetic data. It can also be used to
  modify a config file.

Options:
  --help  Show this message and exit.

Commands:
  artifacts  Artifacts management commands.
  config     Manage Safe Synthesizer configurations.
  run        Run the Safe Synthesizer end-to-end pipeline.
```

## Running the Pipeline

The `run` command executes the Safe Synthesizer pipeline. Without a subcommand, it runs the full end-to-end pipeline:

```bash
> uv run safe-synthesizer run --help
Usage: safe-synthesizer run [OPTIONS] COMMAND [ARGS]...

  Run the Safe Synthesizer end-to-end pipeline.

  Without a subcommand, runs the full end-to-end pipeline. Use 'run train' or
  'run generate' for individual stages.

Options:
  --config TEXT                   path to a yaml config file
  --url TEXT                      Dataset name, URL, or path to CSV dataset.
                                  For 'run generate', this is optional if a
                                  cached dataset exists in the workdir.
  --artifact-path DIRECTORY       Base directory for all runs. Runs are
                                  created as <artifact-
                                  path>/<config>---<dataset>/<timestamp>/. Can
                                  also be set via NSS_ARTIFACTS_PATH env var.
                                  [default: ./safe-synthesizer-artifacts]
  --run-path DIRECTORY            Explicit path for this run's output
                                  directory. When specified, outputs go
                                  directly to this path. Overrides --artifact-
                                  path.
  --output-file PATH              Path to output CSV file. Overrides the
                                  default workdir output location.
  --log-format [json|plain]       Log format for console output. File logging
                                  will always be JSON. Can also be set via
                                  NSS_LOG_FORMAT env var. [default: plain]
  --log-color / --no-log-color    Whether to colorize the log output on the
                                  console. [default: --log-color]
  --log-file PATH                 Path to log file. Defaults to a file nested
                                  under the run directory. Can also be set via
                                  NSS_LOG_FILE env var.
  --wandb-mode [online|offline|disabled]
                                  Wandb mode. 'online' will upload logs to
                                  wandb, 'offline' will save logs to a local
                                  file, 'disabled' will not upload logs to
                                  wandb. Can also be set via WANDB_MODE env
                                  var. [default: disabled]
  --wandb-project TEXT            Wandb project. Can also be set via
                                  WANDB_PROJECT env var.
  -v                              Verbose logging. 'v' shows debug info from
                                  main program, 'vv' shows debug from
                                  dependencies too
  --dataset-registry TEXT         URL or path of a dataset registry YAML file.
                                  If provided, datasets in the registry may be
                                  referenced by name in --url. Can also be set
                                  via NSS_DATASET_REGISTRY env var. If both
                                  env var and CLI option are provided, the CLI
                                  option takes precedence.
  --help                          Show this message and exit.

Commands:
  generate  Run the generation stage only.
  train     Run the training stage only.
```

### Subcommands

- `safe-synthesizer run train` - Run only the training stage, saving the adapter to the run directory.
- `safe-synthesizer run generate` - Run only the generation stage using a saved adapter.

```bash
> uv run safe-synthesizer run generate --help
Usage: safe-synthesizer run generate [OPTIONS]

  Run the generation stage only.

  This command loads a trained adapter and generates synthetic data. Requires
  'run train' to have been executed first.

  Use --run-path to specify the exact run directory containing the trained
  model, or use --auto-discover-adapter with --artifact-path to automatically
  find the latest trained run.

Options:
  --config TEXT                   path to a yaml config file
  --url TEXT                      Dataset name, URL, or path to CSV dataset.
                                  [required]
  --artifact-path DIRECTORY       Base directory for all runs. Runs are
                                  created as <artifact-path>/<config>-
                                  <dataset>/<timestamp>/. [default: ./safe-
                                  synthesizer-artifacts]
  --run-path DIRECTORY            Explicit path for this run's output
                                  directory. When specified, outputs go
                                  directly to this path. Overrides --artifact-
                                  path.
  --output-file PATH              Path to output CSV file. Overrides the
                                  default workdir output location.
  --log-format [json|plain]       Log format for console output. File logging
                                  will always be JSON.
  --log-color / --no-log-color    Whether to colorize the log output on the
                                  console
  --log-file PATH                 Path to log file. Defaults to a file nested
                                  under the run directory.
  -v                              Verbose logging. 'v' shows debug info from
                                  main program, 'vv' shows debug from
                                  dependencies too
  --wandb-mode [online|offline|disabled]
                                  Wandb mode. 'online' will upload logs to
                                  wandb, 'offline' will save logs to a local
                                  file, 'disabled' will not upload logs to
                                  wandb.
  --wandb-project TEXT            Wandb project. If not specified, the project
                                  will be taken from the environment variable
                                  WANDB_PROJECT.
  --auto-discover-adapter         Automatically find the latest trained
                                  adapter in --artifact-path. Without this
                                  flag, --run-path must point to a specific
                                  trained run.
  --help                          Show this message and exit.
```

## Managing Configurations

The `config` command provides tools to validate and modify configuration files:

```bash
> uv run safe-synthesizer config --help
Usage: safe-synthesizer config [OPTIONS] COMMAND [ARGS]...

  Manage Safe Synthesizer configurations.

Options:
  --help  Show this message and exit.

Commands:
  modify    Modify a Safe Synthesizer configuration.
  validate  Validate a Safe Synthesizer configuration.
```

## Artifacts and Workdirs

Safe Synthesizer uses a structured directory format to manage artifacts (trained models, synthetic data, logs).

### Directory Layout

By default, runs are nested under `--artifact-path` using the project name (`<config>---<dataset>`) and a unique run name.

```text
<artifact-path>/<config>---<dataset>/<run_name>/
├── safe-synthesizer-config.json  # Root config for the run
├── train/
│   ├── safe-synthesizer-config.json
│   └── adapter/                  # Trained PEFT adapter
│       ├── adapter_config.json
│       ├── metadata_v2.json
│       └── dataset_schema.json
├── generate/
│   ├── safe-synthesizer-config.json
│   ├── logs.jsonl               # Generation logs
│   ├── synthetic_data.csv       # Default output location
│   └── evaluation_report.html   # HTML evaluation report
└── dataset/                      # Processed dataset splits
    ├── training.csv
    ├── test.csv
    └── validation.csv
```

### Run Names

If not provided with `--run-path`, run names are automatically generated using the current `<timestamp>`.

### Overriding Paths

- Use `--run-path` to specify an explicit directory for the run, bypassing the `<project>/<timestamp>` nesting.
- Use `--output-file` to specify an explicit path for the final synthetic CSV, overriding the default location in the `generate/` directory.

## WandB Logging

Safe Synthesizer supports Weights & Biases (WandB) for experiment tracking.

### Configuration

You can enable WandB logging using CLI options or environment variables:

- `--wandb-mode [online|offline|disabled]`: Set the WandB mode. Default is `disabled`.
- `--wandb-project <name>`: Specify the WandB project name.
- `WANDB_API_KEY`: Ensure your API key is set in your environment.

### Logged Data

The following information is logged to WandB:

- Configuration parameters
- Training metrics (if supported by the backend)
- Generation statistics
- Evaluation results
- Timing information

## Dataset Registry

Safe Synthesizer supports a *dataset registry* to simplify working with a standard set of datasets.
Datasets in the registry may be referenced by name, rather than repeatedly specifying long URLS or file paths on the command line.
Additionally, the registry supports custom config overrides or args that are specific to individual datasets.

### Providing a Dataset Registry

You can supply a dataset registry (YAML file) via either the CLI or an environment variable:

- **CLI Option**:
`--dataset-registry <path_or_url>`
- **Environment Variable**:
Set `NSS_DATASET_REGISTRY` to point to your YAML file (path or URL).

If both are provided, the CLI option takes precedence.

### Referencing Datasets

When a dataset registry is provided, you can use dataset names defined in the registry with the `--url` argument.
For example:

```bash
nemo-safe-synthesizer run --dataset-registry my_registry.yaml --url my_dataset
```

This will load the dataset from the url plus apply any overrides for `my_dataset` from the registry YAML.

### Dataset Registry YAML Format

The registry file should conform to the pydantic model defined by `DatasetRegistry` in `cli/datasets.py`. For example,

```yaml
# registry.yaml
base_url: /root/data/location
datasets:
- name: dataset1
  url: dataset1.csv
- name: dataset2
  url: dataset2.jsonl
  overrides:
    data:
      group_training_examples_by: id
- name: dataset3
  url: /absolute/path/to/dataset.csv
- name: dataset4
  url: https://myhost.com/path/to/dataset.json
  load_args:
    keyword: custom_arg_for_data_reader
```

- Minimal requirements for each entry in the `datasets:` list are a `name` and a `url`.
`url` may be a URL or a file path, anything that data readers like `pd.read_csv` will accept.
- `base_url` - Any relative urls or paths will be prepended with the `base_url` before attempting to load the dataset.
This only applies to the named datasets in the registry which have a relative url.
Passing a relative `--url` on the CLI will attempt to load the file relative to your current working directory, regardless of whether a registry is provided or whether `base_url` is set.
`base_url` is optional, if not provided, it is recommended to use absolute urls or file paths for all entries.
- `overrides` - Dataset specific config overrides, such as a dataset that should always be run with `group_training_examples_by`.
Config values passed as CLI arguments always take precendence, then any overrides from the registry, and finally values from the `--config` yaml file.
- `load_args` - Extra arguments needed by the data reader for a specific dataset.
For example, changing the separator used by `pd.read_csv` for a `.csv` file with a different delimiter.

## Slurm Jobs

For running on Slurm clusters, Safe Synthesizer provides a set of helper scripts in `script/slurm/`.

These scripts support:

- **Matrix runs**: Launching jobs across multiple configurations and datasets.
- **Two-stage pipelines**: Running training and generation as separate jobs with dependencies.
- **Containerized execution**: Running jobs inside enroot containers.

See [script/slurm/README.md](script/slurm/README.md) for detailed instructions on cluster setup and job submission.

## Testing

We have pytest set up for unit, integration, and end-to-end tests.

### Running Tests

You can run tests using `make` targets or `pytest` directly.

```bash
# Run unit tests (excludes slow and e2e tests)
make test

# Run all tests including slow tests (excludes e2e)
make test-slow

# Run SDK-related tests (config, sdk, cli, api)
make test-sdk-related

# Run GPU integration tests (requires CUDA)
make test-gpu-integration

# Run end-to-end tests (requires CUDA)
make test-e2e

# Run specific test files directly
uv run pytest tests/cli/test_run.py
```

### Container-Based Testing

You can run the CI test suite locally in a Linux container using Docker or Podman:

```bash
# Build the test container and run CI tests
make test-ci-container
```

This builds a container image from `containers/Dockerfile.test_ci` and runs `make test-ci` inside it. This is useful for verifying tests pass in a Linux environment when developing on macOS.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for full setup instructions and contribution guidelines.

### Setup

```bash
# 1. Bootstrap development tools
make bootstrap-tools

# 2. Install Python dependencies and package
make bootstrap-nss cpu    # or: cuda, engine, dev

# 3. Run tests
make test

# 4. Format and lint
make format
make lint
```

### NMP Integration

NeMo Safe Synthesizer is developed as a standalone package and published to NVIDIA Artifactory. The NMP platform consumes it as an external dependency.

#### Publishing to Artifactory

The `publish-internal` Makefile target builds a wheel and uploads it to NVIDIA Artifactory:

```bash
make publish-internal
```

This requires `TWINE_REPOSITORY_URL`, `TWINE_USERNAME`, and `TWINE_PASSWORD` environment variables. CI handles this automatically on tagged releases.

#### Local Development with NMP

The NMP service (`services/safe-synthesizer/pyproject.toml`) pulls `nemo-safe-synthesizer` from the `nv-shared-pypi-local` Artifactory index. It's used with a wrapper package called `safe-synthesizer-sdk`.

When iterating on NSS changes that need to be tested in the NMP service, use the Makefile targets in the NMP repo's `services/safe-synthesizer/` directory:

```bash
# In the NMP repo, from services/safe-synthesizer/
make use-nss-local          # Build local wheel and patch pyproject.toml
make use-nss-artifactory    # Revert to Artifactory (always do this before committing)
```

See the NMP service README (`services/safe-synthesizer/README.md`) in NMP for details.

Run `make help` to see all available Makefile targets.