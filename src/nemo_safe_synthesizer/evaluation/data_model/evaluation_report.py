# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import json
import re
from functools import cached_property

from nemo_safe_synthesizer.evaluation.components.component import Component
from nemo_safe_synthesizer.evaluation.data_model.evaluation_dataset import (
    EvaluationDataset,
)
from nemo_safe_synthesizer.observability import get_logger
from pydantic import BaseModel, Field

logger = get_logger(__name__)


class EvaluationReport(BaseModel):
    evaluation_dataset: EvaluationDataset = Field()
    components: list[Component] = Field(default=list())

    def get_dict(self) -> dict:
        return {c.name: c.score.model_dump(mode="json") for c in self.components if c.score.score is not None}

    def get_json(self) -> str:
        return json.dumps(self.get_dict())

    def get_score_by_name(self, name: str) -> float | None:
        for c in self.components:
            if c.name == name:
                return c.score.score
        return None

    @cached_property
    def jinja_context(self) -> dict:
        d = dict()
        for c in self.components:
            snake_name = re.sub(" ", "_", c.name).lower()
            # Key on component name to prevent any collisions
            d[snake_name] = c.jinja_context
        return d
