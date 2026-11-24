# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

import pandas as pd
from plotly.graph_objects import Figure
from pydantic import BaseModel, Field, PrivateAttr

from ...artifacts.analyzers.field_features import FieldType
from ...config.parameters import SafeSynthesizerParameters
from ...evaluation.components.component import Component
from ...evaluation.data_model.evaluation_datasets import EvaluationDatasets
from ...evaluation.data_model.evaluation_field import EvaluationField
from ...evaluation.data_model.evaluation_score import EvaluationScore
from ...observability import get_logger
from ...pii_replacer.transform_result import ColumnStatistics
from . import multi_modal_figures as figures

logger = get_logger(__name__)

# Training values and value-level distributions are not needed to render the
# report, and anything in its context is one template edit away from the HTML.
_COLUMN_STATISTICS_REPORT_EXCLUDE = {"detected_entity_values"}
_EVALUATION_FIELD_REPORT_EXCLUDE = {
    "training_distribution": True,
    "synthetic_distribution": True,
    "training_field_features": {"unique_values_list"},
    "synthetic_field_features": {"unique_values_list"},
    "column_statistics": _COLUMN_STATISTICS_REPORT_EXCLUDE,
}


class ColumnDistributionPlotRow(BaseModel):
    """A pair of side-by-side column distribution plots for the HTML report."""

    name1: str = Field(description="Name of the first column in the plot row.")
    name2: str | None = Field(description="Name of the second column in the plot row, if present.")
    figure: str = Field(description="Rendered HTML of the side-by-side distribution plot.")

    @staticmethod
    def _get_figure_for_field(f: EvaluationField | None, training: pd.Series, synthetic: pd.Series) -> Figure | None:
        """Return the distribution chart for one column, or ``None`` if it has none.

        Numeric training columns get binned bars; other columns get a category
        bar chart when both distributions are available.
        """
        if f is None:
            return None
        # A numeric training column always gets binned bars: a bar per distinct
        # numeric value would name individual training values.
        if f.training_field_features.type == FieldType.NUMERIC:
            return figures.histogram_figure(training, synthetic)
        if f.training_distribution is not None and f.synthetic_distribution is not None:
            return figures.bar_chart(f.training_distribution, f.synthetic_distribution, len(training))
        return None

    @staticmethod
    def _from_figures(names: list[str], figs: list[Figure]) -> ColumnDistributionPlotRow:
        fig = figures.combine_subplots(
            figures=figs,
            titles=names,
            shared_xaxes=False,
            shared_yaxes=False,
            height=500,
            margin={"t": 36, "b": 0, "l": 0, "r": 0},
        )
        fig.update_xaxes(showticklabels=False, visible=False)
        return ColumnDistributionPlotRow(
            name1=names[0], name2=names[1] if len(names) > 1 else None, figure=Component.figure_html(fig)
        )

    @staticmethod
    def from_figures(column_figures: dict[str, Figure]) -> list[dict[str, str]]:
        """Pair per-column figures into side-by-side rows for the HTML report, in column order."""
        items = list(column_figures.items())
        rows = []
        for i in range(0, len(items), 2):
            pair = items[i : i + 2]
            rows.append(ColumnDistributionPlotRow._from_figures([name for name, _ in pair], [fig for _, fig in pair]))
        return [r.model_dump() for r in rows]

    @staticmethod
    def figures_for_datasets(evaluation_datasets: EvaluationDatasets) -> dict[str, Figure]:
        """Return each tabular column's distribution chart, keyed by column name, skipping columns without one."""
        tabular_columns = set(evaluation_datasets.get_tabular_columns())
        column_figures = {}
        for f in evaluation_datasets.evaluation_fields:
            if f.name not in tabular_columns:
                continue
            figure = ColumnDistributionPlotRow._get_figure_for_field(
                f, evaluation_datasets.training[f.name], evaluation_datasets.synthetic[f.name]
            )
            if figure is not None:
                column_figures[f.name] = figure
        return column_figures


class ColumnDistribution(Component):
    """Column Distribution Stability metric.

    Computes per-column Jensen-Shannon divergence between training and
    synthetic distributions, averages across all tabular columns, and maps
    the result to a 0--10 score.  Also carries data for the per-column
    histogram figures and the Training Data Columns table in the HTML report.
    """

    name: str = Field(default="Column Distribution Stability")
    # Keep a copy to simplify rendering
    column_statistics: dict[str, ColumnStatistics] | None = Field(
        default=None, description="Per-column PII entity and transform metadata."
    )
    evaluation_fields: list[EvaluationField] = Field(
        default=list(), description="Per-column evaluation metadata and distribution scores."
    )

    _evaluation_datasets: EvaluationDatasets | None = PrivateAttr(default=None)

    @cached_property
    def report_figures(self) -> dict[str, Figure]:
        """Per-column training vs. synthetic distribution charts, keyed by column name.

        Empty unless the component was built with ``from_evaluation_datasets``, since the charts need the column values.
        """
        if self._evaluation_datasets is None:
            return {}
        return ColumnDistributionPlotRow.figures_for_datasets(self._evaluation_datasets)

    @cached_property
    def jinja_context(self) -> dict:
        """Template context with evaluation fields, column statistics, and paired distribution figures."""
        d = super().jinja_context
        d["anchor_link"] = "#distribution-stability"
        d["figures"] = ColumnDistributionPlotRow.from_figures(self.report_figures)
        if self.evaluation_fields:
            d["evaluation_fields"] = [
                f.model_dump(mode="json", exclude=_EVALUATION_FIELD_REPORT_EXCLUDE) for f in self.evaluation_fields
            ]
        if self.column_statistics:
            d["column_statistics"] = {
                k: v.model_dump(exclude=_COLUMN_STATISTICS_REPORT_EXCLUDE) for k, v in self.column_statistics.items()
            }
        return d

    @staticmethod
    def from_evaluation_datasets(
        evaluation_datasets: EvaluationDatasets, config: SafeSynthesizerParameters | None = None
    ) -> ColumnDistribution:
        """Compute column distribution stability from the evaluation dataset."""
        tabular_columns = set(evaluation_datasets.get_tabular_columns())
        tabular_fields = [f for f in evaluation_datasets.evaluation_fields if f.name in tabular_columns]
        if tabular_fields:
            average_divergence = EvaluationField.get_average_divergence(tabular_fields)
            score = EvaluationField.get_field_distribution_stability(average_divergence)
        else:
            score = EvaluationScore(notes="No tabular columns detected.")
        component = ColumnDistribution(
            score=score,
            column_statistics=evaluation_datasets.column_statistics,
            evaluation_fields=evaluation_datasets.evaluation_fields,
        )
        component._evaluation_datasets = evaluation_datasets
        return component
