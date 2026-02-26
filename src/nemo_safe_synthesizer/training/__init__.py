# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training backends for LLM fine-tuning.

Provides :class:`~.backend.TrainingBackend` and two concrete implementations:
:class:`~.huggingface_backend.HuggingFaceBackend` (standard HuggingFace Trainer)
and :class:`~.unsloth_backend.UnslothTrainer` (Unsloth-optimized training).
"""
