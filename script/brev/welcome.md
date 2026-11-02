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

The other three notebooks go deeper:

- `tutorials/differential-privacy.ipynb` -- formal privacy guarantees (~1 hour)
- `tutorials/time-series-financial-transactions.ipynb` -- sequential data (~20 minutes)
- `tutorials/healthcare-hospital-readmissions.ipynb` -- privacy-aware healthcare analytics (~1 hour)

## Using your own data

Upload a CSV anywhere you like -- drag and drop into the file browser on the
left -- and point the notebook at it:

```python
from nemo_safe_synthesizer import SafeSynthesizer

results = SafeSynthesizer().with_data_source("my-data.csv").run()
```

There is no required location for your files. Put them wherever suits you.

## Working from Cursor or VS Code

To attach your own editor, run `brev open <instance-name> cursor` (or `code`)
on your machine. Safe Synthesizer is already installed in a Python 3.13
environment at `~/.nss-venv`. Point the editor at it rather than creating a new
one:

1. Install the Python and Jupyter extensions in the remote window if the
   editor offers them.
2. Run **Python: Select Interpreter** from the Command Palette, choose
   **Enter interpreter path...**, and enter `/home/<user>/.nss-venv/bin/python`.
   Run `echo $HOME` in a terminal to find the path. The folder name starts
   with a dot, so the editor does not list it on its own.
3. In a notebook, click **Select Kernel**, then **Python Environments**, then
   `.nss-venv`.

The editor's terminal already uses this environment: `which python` prints a
path ending in `.nss-venv/bin/python`, and `uv pip install <package>` installs
into it. Do not select or install into `~/.venv`. That is the Python behind
JupyterLab in the browser, and breaking it breaks those notebooks.

If you set `HF_TOKEN` or `NSS_INFERENCE_KEY` when launching, they are saved in
`~/.nss-env.sh`. The terminal loads them, but a notebook kernel the editor
starts may not. If the kernel list under **Select Kernel > Jupyter Kernel...**
shows **Safe Synthesizer**, pick it instead: it runs the same environment and
includes those values.

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
