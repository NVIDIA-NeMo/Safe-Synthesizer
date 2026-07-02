<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Differentially Private Fine-Tuning of Transformers

This package contains all modifications required to fine tune models with
differential privacy using the hugging face transformers library.
1. `dp_utils.py` contains `OpacusDPTrainer`, which is the DP replacement for
   `transformers.Trainer`, `DataCollatorForPrivateCausalLanguageModeling`, which
   is required for correct per-sample gradient accumulation, and `DPCallback`,
   which registers callbacks for `Trainer`.
2. `linear.py` is included for training with Opacus so that per sample gradients
   are correctly calculated. It is imported for effects, never directly used.
3. `privacy_args.py` contains `PrivacyArguments`, a store for privacy arguments
   and `find_noise_multiplier()` to determine scale of Gaussian noise to be
   added to gradients based on sample size, batch size and privacy parameters.
4. `sampler.py` contains samplers used during batch creation.
