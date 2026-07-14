# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal runtime knobs for the tabular PII engine."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..config.pii_replacement import ReplacePiiConfig

__all__ = ["RuntimeConfig", "runtime_config_from_replace_pii"]


@dataclass
class RuntimeConfig:
    """Internal engine config built from ``ReplacePiiConfig``.

    The first six fields mirror user-facing ``ReplacePiiConfig`` settings.
    Remaining fields are internal MVP defaults (not exposed on ``ReplacePiiConfig``).
    """

    locale: str
    random_seed: int
    replace_group_key: bool
    persona_backend: str
    sdg_pgms_src: str
    managed_assets_path: str | None
    use_race_constraint: bool = True
    low_card_max: int = 12
    dominant_pattern_min_coverage: float = 85.0
    value_match_threshold: float = 0.999
    id_unique_ratio: float = 0.999
    name_fuzzy_threshold: float = 0.86
    freetext_name_token_aliases: bool = True
    freetext_alias_min_token_len: int = 3
    infer_value_patterns: bool = True
    pattern_class_max: int = 6
    pattern_rare_char_frac: float = 0.01
    pattern_sample_cap: int = 5000
    pool_min_size: int = 3_000
    pool_oversample: int = 6


def runtime_config_from_replace_pii(config: ReplacePiiConfig) -> RuntimeConfig:
    seed = config.replacement.seed
    if seed is None:
        seed = int(os.environ.get("PERSON_RANDOM_SEED", "42") or "42")
    managed_path = config.person.resolved_managed_assets_path()
    return RuntimeConfig(
        locale=config.replacement.locale,
        random_seed=seed,
        replace_group_key=config.discovery.replace_group_key,
        persona_backend=config.person.backend.value,
        sdg_pgms_src=config.person.sdg_pgms_src,
        managed_assets_path=str(managed_path),
    )
