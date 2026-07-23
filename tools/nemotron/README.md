<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Nemotron vLLM kernel tuning

`mise run tune-nemotron-vllm-kernels` tunes the vLLM 0.24.0 Mamba
`selective_state_update` kernel for the exact configuration used by Nemotron 3
Nano BF16 on an NVIDIA A100-SXM4-80GB:

```text
headdim=80,dstate=128,device_name=NVIDIA_A100-SXM4-80GB,cache_dtype=float32
```

The sweep covers effective batches from 128 through 65,536. vLLM defines
effective batch as decoder batch multiplied by Mamba heads and keys the tuned
configuration by that product. The wrapper uses a 128-head representation for
each target value. Larger generic shapes are outside this validated sweep.

The task fetches the pinned vLLM `v0.24.0` benchmark source at revision
`ee0da84ab9e04ac7610e28580af62c365e898389`, stages only the benchmark and its
reference implementation, and runs it against the installed vLLM wheel. It
never modifies site-packages. The default output is
`$XDG_CACHE_HOME/nemo-safe-synthesizer/vllm-tuned-configs`, falling back to
`$HOME/.cache/nemo-safe-synthesizer/vllm-tuned-configs`.

Run it only on an otherwise idle A100:

```bash
mise run tune-nemotron-vllm-kernels
```

Use a specific durable output directory or an existing source checkout when
the node has no network access:

```bash
mise run tune-nemotron-vllm-kernels -- --output-dir /path/to/configs
mise run tune-nemotron-vllm-kernels -- --source-dir /path/to/vllm-v0.24.0
```

The task validates the generated JSON and prints the required runtime export:

```bash
export VLLM_TUNED_CONFIG_FOLDER=/path/to/configs
safe-synthesizer run --config config.yaml --data-source data.csv
```

The variable must be present before vLLM imports its environment settings.
Validate or reuse an existing folder without accessing the GPU:

```bash
uv run --frozen python tools/nemotron/tune_vllm_ssu.py validate /path/to/configs
uv run --frozen python tools/nemotron/tune_vllm_ssu.py env /path/to/configs
```

The tool rejects another vLLM version, a different GPU model, unavailable
CUDA, and a selected GPU with active compute processes. `--allow-active-gpu`
is an explicit override; using it can invalidate timings or disrupt another
workload. The sweep uses BF16 activations and a float32 SSM cache. It does not
tune or enable FP8.
