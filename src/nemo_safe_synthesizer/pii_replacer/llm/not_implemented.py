# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Enhancer used when ``llm_enhancement=True`` until a real provider lands."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from ...config.pii_replacement import PiiReplacementPlan
from ...errors import ParameterError
from ..entities import Config
from ..models import DiscoveryResult, FreeTextDetection, PersonaInstance

_LLM_NOT_IMPLEMENTED = (
    "replace_pii.llm_enhancement=True is not supported in this release; "
    "set replace_pii.llm_enhancement to false (the default)."
)


class NotImplementedEnhancer:
    """Raises at the future call site so enabling the flag fails consistently."""

    def review_discovery(
        self,
        df: pd.DataFrame,
        discovery: DiscoveryResult,
        cfg: Config,
    ) -> DiscoveryResult:
        raise ParameterError(_LLM_NOT_IMPLEMENTED)

    def infer_persona_demographics(
        self,
        df: pd.DataFrame,
        instances: list[PersonaInstance],
        cfg: Config,
    ) -> list[PersonaInstance]:
        raise ParameterError(_LLM_NOT_IMPLEMENTED)

    def detect_freetext_entities(
        self,
        df: pd.DataFrame,
        columns: Sequence[str],
        plan: PiiReplacementPlan,
        cfg: Config,
    ) -> list[FreeTextDetection]:
        raise ParameterError(_LLM_NOT_IMPLEMENTED)
