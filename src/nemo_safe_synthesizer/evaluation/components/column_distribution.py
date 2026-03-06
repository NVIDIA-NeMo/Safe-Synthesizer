# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

import pandas as pd
from plotly.graph_objects import Figure
from pydantic import BaseModel, Field

from ...artifacts.analyzers.field_features import FieldType
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_dataset import EvaluationDataset
from ...evaluation.data_model.evaluation_field import EvaluationField
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from ...pii_replacer.transform_result import ColumnStatistics
from . import multi_modal_figures as figures

logger = get_logger(__name__)


class ColumnDistributionPlotRow(BaseModel):
    """A pair of side-by-side column distribution plots for the HTML report."""

    name1: str = Field(description="Name of the first column in the plot row.")
    name2: str | None = Field(description="Name of the second column in the plot row, if present.")
    figure: str = Field(description="Rendered HTML of the side-by-side distribution plot.")

    @staticmethod
    def _get_figure_for_field(f: EvaluationField | None, reference: pd.Series, output) -> Figure | None:
        if f is None:
            return None
        if f.reference_field_features.type != FieldType.NUMERIC or f.output_field_features.type != FieldType.NUMERIC:
            if f.reference_distribution is not None and f.output_distribution is not None:
                figure = figures.bar_chart(f.reference_distribution, f.output_distribution)
            else:
                figure = None
        else:
            figure = figures.histogram_figure(reference, output)
        return figure

    @staticmethod
    def _from_fields(
        field1: EvaluationField, figure1: Figure, field2: EvaluationField | None, figure2: Figure | None
    ) -> ColumnDistributionPlotRow:
        figs = [figure1]
        titles = [field1.name]
        if figure2 is not None and field2 is not None:
            figs.append(figure2)
            titles.append(field2.name)
        fig = figures.combine_subplots(
            figures=figs,
            titles=titles,
            shared_xaxes=False,
            shared_yaxes=False,
            height=500,
            margin={"t": 36, "b": 0, "l": 0, "r": 0},
        )
        fig.update_xaxes(showticklabels=False, visible=False).to_html(full_html=False, include_plotlyjs=False)
        fig = fig.to_html(full_html=False, include_plotlyjs=False)

        return ColumnDistributionPlotRow(name1=field1.name, name2=field2.name if field2 else None, figure=fig)

    @staticmethod
    def from_evaluation_dataset(evaluation_dataset: EvaluationDataset) -> list[dict[str, str]]:
        tups = []
        result_rows = []

        tabular_columns = set(evaluation_dataset.get_tabular_columns())
        tabular_fields = [f for f in evaluation_dataset.evaluation_fields if f.name in tabular_columns]

        for f in tabular_fields:
            figure = ColumnDistributionPlotRow._get_figure_for_field(
                f, evaluation_dataset.reference[f.name], evaluation_dataset.output[f.name]
            )
            if figure is not None:
                tups.append((f, figure))

        for i in range(0, len(tups), 2):
            field1, figure1 = tups[i]
            if i + 1 < len(tups):
                field2, figure2 = tups[i + 1]
            else:
                field2, figure2 = None, None
            result_rows.append(ColumnDistributionPlotRow._from_fields(field1, figure1, field2, figure2))
        return [r.model_dump() for r in result_rows]


class ColumnDistribution(Component):
    """Column Distribution Stability metric.

    Computes per-column Jensen-Shannon divergence between reference and
    output distributions, averages across all tabular columns, and maps
    the result to a 0--10 score.  Also carries data for the per-column
    histogram figures and the Reference Columns table in the HTML report.
    """

    name: str = Field(default="Column Distribution Stability")
    # Keep a copy to simplify rendering
    column_statistics: dict[str, ColumnStatistics] | None = Field(
        default=None, description="Per-column PII entity and transform metadata."
    )
    evaluation_fields: list[EvaluationField] = Field(
        default=list(), description="Per-column evaluation metadata and distribution scores."
    )

    @cached_property
    def jinja_context(self):
        """Template context with evaluation fields and column statistics for the report."""
        d = super().jinja_context
        d["anchor_link"] = "#distribution-stability"
        if self.evaluation_fields:
            d["evaluation_fields"] = [f.model_dump(mode="json") for f in self.evaluation_fields]
        if self.column_statistics:
            d["column_statistics"] = {k: v.model_dump for k, v in self.column_statistics.items()}
        # Figures are set up with ColumnDistributionPlotRow. It requires an EvaluateDataset so needs
        # to be done out of band as separate call from enclosing report class.
        return d

    @staticmethod
    def from_evaluation_dataset(
        evaluation_dataset: EvaluationDataset, config: SafeSynthesizerParameters | None = None
    ) -> ColumnDistribution:
        """Compute column distribution stability from the evaluation dataset."""
        tabular_columns = set(evaluation_dataset.get_tabular_columns())
        tabular_fields = [f for f in evaluation_dataset.evaluation_fields if f.name in tabular_columns]
        if tabular_fields:
            average_divergence = EvaluationField.get_average_divergence(tabular_fields)
            score = EvaluationField.get_field_distribution_stability(average_divergence)
            return ColumnDistribution(
                score=score,
                column_statistics=evaluation_dataset.column_statistics,
                evaluation_fields=evaluation_dataset.evaluation_fields,
            )
        else:
            return ColumnDistribution(
                score=EvaluationScore(notes="No tabular columns detected."),
                column_statistics=evaluation_dataset.column_statistics,
                evaluation_fields=evaluation_dataset.evaluation_fields,
            )
