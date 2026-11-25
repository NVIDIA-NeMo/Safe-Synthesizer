# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_datasets import (
    EvaluationDatasets,
)
from ...observability import get_logger

logger = get_logger(__name__)


class EvaluationReport(BaseModel):
    """Container for a completed evaluation -- dataset, components, and scores.

    Subclassed by ``MultimodalReport``, which assembles the components.
    Provides score serialization helpers.
    """

    evaluation_datasets: EvaluationDatasets = Field(
        description="The paired training/synthetic data used for evaluation."
    )
    components: list[Component] = Field(
        default=list(), description="Ordered list of evaluation components with their scores."
    )

    def get_dict(self) -> dict:
        """Return component scores as a ``{name: score_dict}`` mapping."""
        return {c.name: c.score.model_dump(mode="json") for c in self.components if c.score.score is not None}

    def get_json(self) -> str:
        """Return component scores as a JSON string."""
        return json.dumps(self.get_dict())

    def get_score_by_name(self, name: str) -> float | None:
        """Look up a component's numeric score by display name."""
        for c in self.components:
            if c.name == name:
                return c.score.score
        return None
