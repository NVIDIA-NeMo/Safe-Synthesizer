# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Registry of time-series metric cards in the multi-modal report.

Each time-series metric registers one ``TimeSeriesCard``. The report runs every
registered component, builds each card's figures, and renders the cards in
registry order, so adding a metric does not touch the report code or the
report template.

Classes:
    TimeSeriesCard: How one time-series metric is computed and rendered.

Functions:
    figure_rows: Convert figure specs into template rows.

Attributes:
    TIME_SERIES_CARDS: Registered cards, in display order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

from ....evaluation.assets.text.multi_modal_tooltips import tooltips
from ....evaluation.components.autocorrelation_similarity import AutocorrelationSimilarity
from ....evaluation.components.autocorrelation_similarity_figures import add_autocorrelation_context
from ....evaluation.components.component import Component

FigureSpec = tuple[str, go.Figure, bool, str]
"""Chart title, figure, whether it gets the column select, and its tooltip key."""


@dataclass(frozen=True)
class TimeSeriesCard:
    """How one time-series metric is computed and rendered in the report.

    Attributes:
        context_key: Key of the component's entry in the report context.
        card_id: HTML id of the metric card.
        component_cls: Component class that computes the metric.
        description: One-line description shown on the card.
        tooltip_key: Key of the card's info tooltip in ``tooltips``.
        template: Jinja template rendered inside the card.
        add_context: Adds the card's summary fields to its context entry and
            returns its figures.
        score_colored_traces: Whether chart traces are colored by score, so
            the report keeps their colors instead of applying dataset colors.
    """

    context_key: str
    card_id: str
    component_cls: type[Component]
    description: str
    tooltip_key: str
    template: str
    add_context: Callable[[dict[str, Any]], list[FigureSpec]]
    score_colored_traces: bool = False


TIME_SERIES_CARDS: tuple[TimeSeriesCard, ...] = (
    TimeSeriesCard(
        context_key="autocorrelation_similarity",
        card_id="autocorrelation-similarity",
        component_cls=AutocorrelationSimilarity,
        description="Measures whether numeric columns change over time like the training data",
        tooltip_key="autocorrelation_similarity_info",
        template="jinja/components/autocorrelation_similarity.j2",
        add_context=add_autocorrelation_context,
        score_colored_traces=True,
    ),
)


def figure_rows(figures: Sequence[FigureSpec]) -> list[dict[str, Any]]:
    """Convert figure specs into the rows the chart grid template renders."""
    return [
        {
            "title": title,
            "html": figure.to_html(full_html=False, include_plotlyjs=False),
            "column_select": column_select,
            "tooltip_id": tooltip_key.replace("_", "-"),
            "tooltip": tooltips[tooltip_key],
        }
        for title, figure, column_select, tooltip_key in figures
    ]
