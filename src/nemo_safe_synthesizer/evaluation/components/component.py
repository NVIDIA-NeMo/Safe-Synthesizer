# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC

import pandas as pd
from pydantic import BaseModel, Field

from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_score import EvaluationScore


class Component(ABC, BaseModel):
    """Abstract base for all evaluation components.

    Each component computes one quality or privacy metric from an
    ``EvaluationDatasets``. ``build_report_data`` exports its results, and
    the HTML report is rendered from that export.

    Subclasses should override ``from_evaluation_datasets`` to perform
    their metric-specific computation.
    """

    name: str = Field(description="Display name used in JSON summaries and the HTML report.")
    score: EvaluationScore = Field(
        default=EvaluationScore(), description="The computed EvaluationScore for this component."
    )

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets, config: SafeSynthesizerParameters | None = None
    ) -> Component:
        """Create a component from an ``EvaluationDatasets``.

        Subclasses override this to compute their specific metric.

        Args:
            evaluation_datasets: Paired training/synthetic data.
            config: Optional pipeline configuration parameters.

        Returns:
            A new component instance with computed scores.
        """
        return Component(name="Placeholder")

    def get_json(self) -> str:
        """Serialize the component score to a JSON string."""
        return self.score.model_dump_json()

    @staticmethod
    def is_nonempty(dfs: None | pd.DataFrame | list[pd.DataFrame | None]) -> bool:
        """Return ``True`` if all provided DataFrames are non-``None`` and non-empty."""
        if dfs is None:
            return False
        if isinstance(dfs, pd.DataFrame):
            dfs = [dfs]
        return all([df is not None and not df.empty for df in dfs])
