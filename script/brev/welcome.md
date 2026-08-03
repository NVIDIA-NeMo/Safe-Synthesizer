<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# NeMo Safe Synthesizer

Create private, safe versions of sensitive tabular data -- entirely synthetic
records with no one-to-one mapping back to your originals.

Everything is installed and ready. Nothing to set up.

## Start here

Open `tutorials/safe-synthesizer-101.ipynb` and run the cells top to bottom.
It takes about 15 minutes and walks through the full pipeline on a sample
dataset.

The other two notebooks go deeper:

- `tutorials/differential-privacy.ipynb` -- formal privacy guarantees (~1 hour)
- `tutorials/time-series-financial-transactions.ipynb` -- sequential data (~20 minutes)

## Using your own data

Upload a CSV anywhere you like -- drag and drop into the file browser on the
left -- and point the notebook at it:

```python
from nemo_safe_synthesizer import SafeSynthesizer

results = SafeSynthesizer().with_data_source("my-data.csv").run()
```

There is no required location for your files. Put them wherever suits you.

## Good to know

- Notebooks already run on the right Python. You should never need to change
  the kernel; if you do switch it, pick "Safe Synthesizer".
- Model weights download on first use, so the first run is slower than later
  ones.
- This instance bills continuously and cannot be paused. Delete it when you are
  finished, and download anything you want to keep first.
- Provisioning log: `.nss-setup.log` (hidden files are off by default in the
  file browser).

## Learn more

- Documentation: <https://nvidia-nemo.github.io/Safe-Synthesizer/>
- Source: <https://github.com/NVIDIA-NeMo/Safe-Synthesizer>
