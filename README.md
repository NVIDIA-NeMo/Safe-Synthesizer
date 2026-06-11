# 🛡️ NeMo Safe Synthesizer

NVIDIA NeMo Safe Synthesizer creates private, safe versions of sensitive tabular datasets -- entirely synthetic data with no one-to-one mapping to your original records. Purpose-built for privacy compliance and sensitive information protection while preserving data utility for downstream AI tasks.

## Quick Start

Read detailed usage below, or jump to the documentation with [Getting Started](https://nvidia-nemo.github.io/Safe-Synthesizer/user-guide/getting-started/) or the [Safe Synthesizer 101](https://nvidia-nemo.github.io/Safe-Synthesizer/tutorials/safe-synthesizer-101/) notebook.


### Prerequisites

- Python 3.11–3.13 (`.python-version` pins 3.13 for local/dev bootstrap; any 3.11, 3.12, or 3.13 interpreter works. Python 3.14+ is NOT supported because vLLM currently declares `<3.14` support while upstream resolves Python 3.14 wheel compatibility across its dependency stack)
- [uv](https://docs.astral.sh/uv/) (recommended) or pip -- Python package manager
- NVIDIA GPU (A100 or larger) for training and generation
- Linux only -- macOS, Windows, and Apple Silicon are not supported for training or generation. A CPU-only install is available for development and configuration validation.

### Installation

```bash
# With uv (recommended):
uv pip install "nemo-safe-synthesizer[cu129,engine]" \
  --index https://flashinfer.ai/whl/cu129 \
  --index https://download.pytorch.org/whl/cu129 \
  --index https://wheels.vllm.ai/88d34c6409e9fb3c7b8ca0c04756f061d2099eb1/cu129 \
  --index-strategy unsafe-best-match

# With pip:
pip install "nemo-safe-synthesizer[cu129,engine]" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://flashinfer.ai/whl/cu129 \
  --extra-index-url https://wheels.vllm.ai/88d34c6409e9fb3c7b8ca0c04756f061d2099eb1/cu129
```

Or install from source:

```bash
git clone https://github.com/NVIDIA-NeMo/Safe-Synthesizer.git
cd Safe-Synthesizer
make setup # installs pinned mise, pinned tools from mise.lock, and .venv
mise run bootstrap-nss cuda
```

Development tools (`ruff`, `ty`, `yq`, `gh`, etc.) are managed via [mise](https://mise.jdx.dev/). Tool versions are declared in `.mise.toml` and locked in `mise.lock` (committed). mise also manages environment variables -- place project-local secrets or overrides in `.env` or `.env.local` (both git-ignored, auto-loaded by mise).

Project commands run through mise tasks under `.mise/tasks/`: `*.toml` files for declarative tasks, executable scripts for bash-heavy logic.

```bash
mise tasks                # list public tasks
mise tasks --hidden       # include helper and legacy alias tasks
mise tasks deps validate  # inspect the pre-PR validation graph
mise run validate         # check + lock-check + CI unit tests
```

### Running

Activate Python virtual environment and run the CLI using `safe-synthesizer`:

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
  --data-source TEXT                      Dataset name, URL, or path to CSV dataset.
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
                                  referenced by name in --data-source. Can also be set
                                  via NSS_DATASET_REGISTRY env var. If both
                                  env var and CLI option are provided, the CLI
                                  option takes precedence.
  --emit_telemetry BOOLEAN        Whether to emit anonymous Safe Synthesizer
                                  telemetry events.
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
  --data-source TEXT                      Dataset name, URL, or path to CSV dataset.
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

## Attention Configuration

Safe Synthesizer exposes attention implementation settings for both training and generation.

### Training (`attn_implementation`)

Controls the HuggingFace attention backend used during model loading for training. Set via config YAML, CLI, or SDK:

```yaml
# config.yaml
training:
  attn_implementation: "sdpa"
```

```bash
# CLI override
safe-synthesizer run --training__attn_implementation sdpa --data-source my_data.csv
```

| Value | Description | Requires |
|-------|-------------|----------|
| `sdpa` | PyTorch scaled dot product attention (default) | None (built-in) |
| `eager` | Standard PyTorch attention | None (built-in) |
| `kernels-community/flash-attn2` | Flash Attention 2 via HuggingFace Kernels Hub | `kernels` pip package |
| `kernels-community/vllm-flash-attn3` | Flash Attention 3 via HuggingFace Kernels Hub | `kernels` pip package and compatible prebuilt kernel |
| `flash_attention_2` | Flash Attention 2 (traditional) | `flash-attn` pip package |
| `flash_attention_3` | Flash Attention 3 (traditional) | `flash-attn-3` support |

If a `kernels-community/...` value is configured but the `kernels` package is not installed, the backend automatically falls back to `sdpa`.

### Generation (`attention_backend`)

Controls the vLLM attention backend used during synthetic data generation. Defaults to `"auto"`, which lets vLLM auto-select the best available backend.

```yaml
# config.yaml
generation:
  attention_backend: "FLASH_ATTN"
```

Common values: `FLASHINFER`, `FLASH_ATTN`, `TORCH_SDPA`, `TRITON_ATTN`, `FLEX_ATTENTION`.

## NIM Integration

Column classification uses a NIM/OpenAI-compatible endpoint to detect entity types
in your data. `NSS_INFERENCE_ENDPOINT` defaults to `https://integrate.api.nvidia.com/v1`;
override it to use a different endpoint.

When using the CLI or Python SDK, set `NSS_INFERENCE_KEY` (and `NSS_INFERENCE_ENDPOINT` only if not
using the default) so column classification can run.

### Local Endpoint

To point to a locally hosted LLM, add the variables to `.env.local` (git-ignored, auto-loaded by mise):

```bash
# .env.local
NSS_INFERENCE_ENDPOINT=https://your-local-nim-endpoint
NSS_INFERENCE_KEY=your-api-key  # pragma: allowlist secret
```

Or export them in your shell:

```bash
export NSS_INFERENCE_ENDPOINT="https://your-local-nim-endpoint"
export NSS_INFERENCE_KEY="your-api-key"  # pragma: allowlist secret
```

### Disable Classification

To disable classification entirely:

```yaml
replace_pii:
  globals:
    classify:
      enable_classify: false
```

When classification is disabled, NSS falls back to default entity types.

## Artifacts and Workdirs

Safe Synthesizer uses a structured directory format to manage artifacts (trained models, synthetic data, logs).

### Directory Layout

By default, runs are nested under `--artifact-path` using the project name (`<config>---<dataset>`) and a unique run name.

```text
<artifact-path>/<config>---<dataset>/<run_name>/
├── train/
│   ├── safe-synthesizer-config.json
│   └── adapter/                     # trained PEFT adapter
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── metadata_v2.json
│       └── dataset_schema.json
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

- CLI Option:
`--dataset-registry <path_or_url>`
- Environment Variable:
Set `NSS_DATASET_REGISTRY` to point to your YAML file (path or URL).

If both are provided, the CLI option takes precedence.

### Referencing Datasets

When a dataset registry is provided, you can use dataset names defined in the registry with the `--data-source` argument.
For example:

```bash
nemo-safe-synthesizer run --dataset-registry my_registry.yaml --data-source my_dataset
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
Passing a relative `--data-source` on the CLI will attempt to load the file relative to your current working directory, regardless of whether a registry is provided or whether `base_url` is set.
`base_url` is optional, if not provided, it is recommended to use absolute urls or file paths for all entries.
- `overrides` - Dataset specific config overrides, such as a dataset that should always be run with `group_training_examples_by`.
Config values passed as CLI arguments always take precendence, then any overrides from the registry, and finally values from the `--config` yaml file.
- `load_args` - Extra arguments needed by the data reader for a specific dataset.
For example, changing the separator used by `pd.read_csv` for a `.csv` file with a different delimiter.

## Telemetry & Privacy

NeMo Safe Synthesizer includes an optional function to share anonymous telemetry data with NVIDIA for product improvement. Data collected is limited to run-level operational metrics (such as final run status, processing time, record and token counts, configuration parameters, top-level quality and privacy scores, base model used, deployment type, and GPU type). No user or device information is collected. This data is used to prioritize product improvements and will be shared in aggregate with the community. It is not used to track any individual user behavior.

You may opt out of telemetry collection at any time. Opting out applies only to data collection by the NeMo Safe Synthesizer library itself. To disable telemetry in a YAML config, set:

```yaml
emit_telemetry: false
```

To disable telemetry for one CLI invocation, pass `--emit_telemetry false`:

```bash
safe-synthesizer run --emit_telemetry false --data-source my_data.csv
```

To disable telemetry for the current shell, set `NEMO_TELEMETRY_ENABLED=false` (other accepted disabling values: `0`, `no`) in your environment before running:

```bash
export NEMO_TELEMETRY_ENABLED=false
```

Use of third-party endpoints, including NVIDIA Build: NeMo Safe Synthesizer can be configured to use various inference endpoints, including build.nvidia.com (NVIDIA Build). If you choose to use NVIDIA Build or any other third-party endpoint, that endpoint's own terms of service and privacy practices apply independently of this library. Any opt-out you exercise within NeMo Safe Synthesizer does not extend to data collection by your chosen endpoint. NVIDIA Build is intended for evaluation and testing purposes only and may not be used in production environments. Do not submit any confidential information or personal data when using NVIDIA Build.

## License

NeMo Safe Synthesizer is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/LICENSE).

## Contact

- [Need help? Ask us a question](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/discussions)
- [Report a bug](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues/new?template=bug-report.yml)
- [Make a feature request](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/issues/new?template=feature-request.yml)
- [Report a security vulnerability](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/security/policy)
