# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default enhancer when LLM enhancement is disabled."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ...config.pii_replacement import PiiReplacementPlan
from ..entities import Config
from ..models import DiscoveryResult, FreeTextDetection, PersonaInstance


class NoopEnhancer:
    """Pass-through enhancer: heuristics are authoritative."""

    def review_discovery(
        self,
        df: pd.DataFrame,
        discovery: DiscoveryResult,
        cfg: Config,
    ) -> DiscoveryResult:
        return discovery

    def infer_persona_demographics(
        self,
        df: pd.DataFrame,
        instances: list[PersonaInstance],
        cfg: Config,
    ) -> list[PersonaInstance]:
        return instances

    def detect_freetext_entities(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        plan: PiiReplacementPlan,
        cfg: Config,
    ) -> list[FreeTextDetection]:
        return []
