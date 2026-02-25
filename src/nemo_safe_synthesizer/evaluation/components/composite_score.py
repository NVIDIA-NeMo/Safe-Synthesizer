# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property

from ..components.component import Component
from ..data_model.evaluation_score import EvaluationScore, PrivacyGrade
from . import multi_modal_figures as figures


class CompositeScore(Component):
    @cached_property
    def jinja_context(self):
        d = super().jinja_context
        # This is some "plotly magic."  The figure is a div with an id and an inlined script.
        # If you attempt to reuse the figure (we do), it won't render for the second one.
        d["figure"] = figures.gauge_chart(self.score).to_html(full_html=False, include_plotlyjs=False)
        d["figure_overview"] = figures.gauge_chart(self.score).to_html(full_html=False, include_plotlyjs=False)
        return d

    @staticmethod
    def from_components(components: list[Component] | Component, name: str) -> CompositeScore:
        if isinstance(components, Component):
            return CompositeScore(score=components.score, name=name)
        if (
            components is None
            or len(components) == 0
            or all([True for c in components if c.score is None or c.score.score is None])
        ):
            return CompositeScore(score=EvaluationScore(), name=name)

        # Take the mean
        total = 0.0
        total_components = 0
        is_privacy = False
        for component in components:
            if component.score.score:
                total += component.score.score
                total_components += 1
            if component.score.grade and isinstance(component.score.grade, PrivacyGrade):
                is_privacy = True
        if total_components > 0:
            score = total / total_components
            return CompositeScore(
                score=EvaluationScore.finalize_grade(raw_score=score, score=score, is_privacy=is_privacy), name=name
            )
        else:
            return CompositeScore(score=EvaluationScore(), name=name)
