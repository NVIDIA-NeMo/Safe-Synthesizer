# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import re
from functools import cached_property

from pydantic import BaseModel, Field

from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_dataset import (
    EvaluationDataset,
)
from ...observability import get_logger

logger = get_logger(__name__)


class EvaluationReport(BaseModel):
    """Container for a completed evaluation -- dataset, components, and scores.

    Subclassed by ``MultimodalReport`` to add report-specific rendering
    logic. Provides serialization helpers and a Jinja2 context property
    consumed by the HTML report template.
    """

    evaluation_dataset: EvaluationDataset = Field(description="The paired reference/output data used for evaluation.")
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

    @cached_property
    def jinja_context(self) -> dict:
        """Jinja2 template context mapping snake-cased component names to their contexts."""
        d = dict()
        for c in self.components:
            snake_name = re.sub(" ", "_", c.name).lower()
            # Key on component name to prevent any collisions
            d[snake_name] = c.jinja_context
        return d
