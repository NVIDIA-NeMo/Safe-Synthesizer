# NeMo Safe Synthesizer on DGX Spark

Generate synthetic tabular data with quality and privacy guarantees — train, generate, and evaluate in one command.

## Quick Start

### 1. Build and launch the container

```bash
git clone https://github.com/NVIDIA-NeMo/Safe-Synthesizer.git
cd Safe-Synthesizer
docker build -f containers/Dockerfile.cuda-aarch64 -t nss-spark .
docker run --gpus all --ipc=host --ulimit memlock=-1 -it --ulimit stack=67108864 nss-spark
```

> If HTTPS clone fails with authentication errors, use SSH:
> `git clone git@github.com:NVIDIA-NeMo/Safe-Synthesizer.git`

### 2. Run

```python
python -c "
import pandas as pd, numpy as np
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

# Sample data — replace with your own CSV or DataFrame
np.random.seed(42)
df = pd.DataFrame({
    'age': np.random.randint(18, 85, 500),
    'income': np.random.lognormal(10.5, 0.8, 500).astype(int),
    'credit_score': np.random.randint(300, 850, 500),
    'default': np.random.choice(['yes', 'no'], 500, p=[0.15, 0.85]),
})

builder = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_replace_pii()
    .with_generate(num_records=500)
    .with_evaluate()
)
builder.run()

s = builder.results.summary
print(f'Quality (SQS): {s.synthetic_data_quality_score}/10')
print(f'Privacy (DPS): {s.data_privacy_score}/10')
builder.save_results()
"
```

Expected: SQS ~8-9, DPS ~9-10.

> **First run is slower.** Model weights (~6 GB) download from HuggingFace and Triton
> JIT-compiles LoRA kernels for the GB10. Subsequent runs reuse cached weights and kernels.

## Use Your Own Data

```python
from nemo_safe_synthesizer.sdk.library_builder import SafeSynthesizer

builder = (
    SafeSynthesizer()
    .with_data_source("your_data.csv")  # or pass a DataFrame
    .with_replace_pii()
    .with_generate(num_records=1000)
    .with_evaluate()
)
builder.run()
builder.save_results()
```

Outputs are saved to `safe-synthesizer-artifacts/` — synthetic CSV and an HTML evaluation report.

## Optional: Improve PII Detection

Set a NIM API key for LLM-based column classification (more accurate than NER-only):

```bash
export NIM_ENDPOINT_URL="https://integrate.api.nvidia.com/v1"
export NIM_API_KEY="<your-api-key>"  # pragma: allowlist secret  # get one at build.nvidia.com/settings/api-keys
```

## Optional: Differential Privacy

```python
builder = (
    SafeSynthesizer()
    .with_data_source(df)
    .with_replace_pii()
    .with_generate(num_records=1000)
    .with_differential_privacy(dp_enabled=True, epsilon=8.0)
    .with_evaluate()
)
```

## Troubleshooting

**Slow first generation batch?** Triton JIT-compiles LoRA kernels for the GB10 on first use. This is normal and only happens once per container session.

**Memory issues between runs?** Flush the cache:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

**Why a container?** DGX Spark's CUDA 13 + aarch64 requires specific Triton, vLLM, and PyTorch versions. The container (`nvcr.io/nvidia/vllm:26.02-py3`) provides a tested stack where Unsloth training and vLLM generation work natively.

**Full documentation:** [Safe Synthesizer User Guide](https://github.com/NVIDIA-NeMo/Safe-Synthesizer/blob/main/docs/user-guide/getting-started.md)
