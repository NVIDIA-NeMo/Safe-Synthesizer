# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from functools import cached_property
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field

from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.data_model.evaluation_dataset import EvaluationDataset
from ...evaluation.data_model.evaluation_score import EvaluationScore
from . import multi_modal_figures as figures


class Component(ABC, BaseModel):
    name: str = Field(
        description="Override this with the fancy display name of your component. It is used for json summaries and rendering scores."
    )
    score: EvaluationScore = Field(default=EvaluationScore())

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> Component:
        return Component(name="")

    def get_json(self) -> str:
        return self.score.model_dump_json()

    @cached_property
    def jinja_context(self) -> dict[str, Any]:
        # Dict values are typed as "Any" but err on the side of primitives (html strings, not plotly.Figure e.g.).
        # Prepping up front saves formatting logic inlined in templates.
        d = dict()
        d["name"] = self.name
        d["score"] = self.score.model_dump(mode="json")
        d["score"]["figure"] = figures.gauge_chart(self.score, min=True).to_html(
            full_html=False, include_plotlyjs=False
        )
        return d

    @staticmethod
    def is_nonempty(dfs: None | pd.DataFrame | list[pd.DataFrame | None]) -> bool:
        """
        Util for components that need to check dataframes before attempting to render (correlation and PCA)
        """
        if dfs is None:
            return False
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        return all([df is not None and not df.empty for df in dfs])
