<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tutorials

Interactive Jupyter notebook tutorials for NeMo Safe Synthesizer.

[![Launch on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg#only-light)](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3HBtA2NKQaBukL2TyDphWUcvQ17)
[![Launch on Brev](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-light.svg#only-dark)](https://brev.nvidia.com/launchable/deploy/now?launchableID=env-3HBtA2NKQaBukL2TyDphWUcvQ17)

These notebooks need a Linux machine with an NVIDIA GPU. The launchable above provides
one with Safe Synthesizer and all four notebooks already installed.

!!! warning "The instance bills continuously"
    Most GPU providers on Brev do not support stopping an instance. Billing runs from
    creation until deletion, so delete the instance when you are finished and download
    anything you want to keep first.

## Available Tutorials

- [Safe Synthesizer 101](safe-synthesizer-101.ipynb) -- learn the fundamentals
- [Differential Privacy](differential-privacy.ipynb) -- enable differential privacy guarantees
- [Time-Series Financial Transactions](time-series-financial-transactions.ipynb) -- synthesize grouped transaction histories
- [Healthcare Hospital Readmissions](healthcare-hospital-readmissions.ipynb) -- explore privacy-aware healthcare utilization analytics

## Adding a Tutorial

To add a new tutorial:

1. Create a Jupyter notebook (`.ipynb`) in the `docs/tutorials/` directory
2. Add it to the `nav` section in `mkdocs.yml` under Tutorials
3. The notebook will be automatically rendered as a documentation page

!!! tip
    Notebooks are rendered with `mkdocs-jupyter`. Cell outputs are included as-is (notebooks are _not_ re-executed during the docs build). Make sure to run your notebook and save it with outputs before committing.

## Guidelines

- Use clear markdown cells to explain each step
- Include expected outputs so readers can follow along without running the notebook
- Keep notebooks focused on a single topic or workflow
- Name files descriptively, e.g., `basic_pipeline.ipynb`, `custom_evaluation.ipynb`
