# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe Synthesizer tokenizer contracts."""

from .base import EngineParity, NssTokenizer
from .persistence import NSS_TOKENIZER_MANIFEST, load_nss_tokenizer, save_nss_tokenizer
from .registry import NssTokenizerRegistry, RegistryEntry, builtin_registry
from .runtime import create_runtime_nss_tokenizer, resolve_native_provenance
from .spec import NssTokenizerSpec, PolicyEpochs
from .tabular import TabularNssTokenizer
from .timeseries import TimeSeriesNssTokenizer
from .types import (
    FramingPolicy,
    FrozenJsonObject,
    PaddedTokenBatch,
    PromptEncoding,
    RecordBatch,
    RecordEncoding,
    TabularContext,
    TimeSeriesContext,
    TokenBatch,
    TokenizerCapabilities,
    TrainingCapacity,
    TrainingEncoding,
    WorkloadContext,
    WorkloadKind,
)

__all__ = [
    "EngineParity",
    "FramingPolicy",
    "FrozenJsonObject",
    "NssTokenizer",
    "NSS_TOKENIZER_MANIFEST",
    "NssTokenizerRegistry",
    "NssTokenizerSpec",
    "PaddedTokenBatch",
    "PolicyEpochs",
    "PromptEncoding",
    "RecordBatch",
    "RecordEncoding",
    "RegistryEntry",
    "TabularContext",
    "TabularNssTokenizer",
    "TimeSeriesContext",
    "TimeSeriesNssTokenizer",
    "TokenizerCapabilities",
    "TrainingCapacity",
    "TokenBatch",
    "TrainingEncoding",
    "WorkloadContext",
    "WorkloadKind",
    "builtin_registry",
    "create_runtime_nss_tokenizer",
    "load_nss_tokenizer",
    "resolve_native_provenance",
    "save_nss_tokenizer",
]
