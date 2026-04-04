# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable

def register_grad_sampler(target_class_or_fn: Any) -> Callable[..., Any]: ...
