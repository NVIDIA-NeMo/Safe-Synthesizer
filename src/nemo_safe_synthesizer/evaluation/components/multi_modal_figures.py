# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import cast

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy import cos, pi, sin
from plotly.subplots import make_subplots

from ...evaluation.data_model.evaluation_score import (
    EvaluationScore,
    PrivacyGrade,
)
from ...evaluation.statistics.stats import get_numeric_distribution_bins

_REPORT_PALETTE = ["#3C2ED1", "#1AA2E6"]
_GRAPH_BARGAP = 0.2  # gap between bars of adjacent location coordinates
_GRAPH_BARGROUPGAP = 0.1  # gap between bars of the same location coordinates

GRAPH_BACKGROUND_COLOR = "#EBEAF0"

INFERENCE_ATTACK_VALUES_FOR_GRAPHS = {
    "Poor": "rgb(203, 210, 252)",
    "Moderate": "rgb(145, 156, 237)",
    "Good": "rgb(102, 101, 222)",
    "Very Good": "rgb(59, 46, 208)",
    "Excellent": "rgb(12, 0, 117)",
}


def degree_to_radian(degrees):
    return degrees * pi / 180


def gauge_chart(evaluation_score: EvaluationScore, degree_start=210, degree_end=-30, min=False, dps=False) -> go.Figure:
    if isinstance(evaluation_score.grade, PrivacyGrade):
        dps = True

    radian_start = degree_to_radian(degree_start)
    radian_end = degree_to_radian(degree_end)

    fig = go.Figure()

    color = GRAPH_BACKGROUND_COLOR
    value = radian_start
    text = "--"

    if evaluation_score.score is not None:
        if dps:
            color = _REPORT_PALETTE[1]
        else:
            color = _REPORT_PALETTE[0]
        value = degree_to_radian(evaluation_score.score * (degree_end - degree_start) / 10 + degree_start)
        text = evaluation_score.score

    non_filled = np.linspace(value, radian_end, 200)

    # Draw non-filled part
    fig.add_trace(
        go.Scatter(
            x=cos(non_filled),
            y=sin(non_filled),
            mode="markers",
            marker_symbol="circle",
            marker_size=25 if not min else 10,
            marker=dict(color=GRAPH_BACKGROUND_COLOR),
            showlegend=False,
        )
    )
    if evaluation_score.score is not None:
        filled = np.linspace(radian_start, value, 200)

        # Draw filled part
        fig.add_trace(
            go.Scatter(
                x=cos(filled),
                y=sin(filled),
                mode="markers",
                marker_symbol="circle",
                marker_size=25 if not min else 10,
                marker=dict(
                    line=dict(color=color, width=1),
                    color=color,
                ),
                showlegend=False,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[-0.05],
            mode="text",
            text=text,
            textfont=dict(color=color, size=55 if not min else 20),
            textposition="middle center",
            showlegend=False,
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_range=[-1.3, 1.3],
        yaxis_range=[-1, 1.3],
        xaxis_visible=False,
        xaxis_showticklabels=False,
        yaxis_visible=False,
        yaxis_showticklabels=False,
        template="plotly_white",
        width=215 if not min else 90,
        height=180 if not min else 75,
        xaxis_fixedrange=True,
        yaxis_fixedrange=True,
        hovermode=False,
    )

    return fig


def pie(
    labels: list[str],
    values: pd.Series,
    textinfo: str = "label+percent",
    sort: bool = True,
) -> go.Figure:
    return go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                textinfo=textinfo,
                insidetextorientation="radial",
                marker=dict(
                    colors=[
                        INFERENCE_ATTACK_VALUES_FOR_GRAPHS[label] for label in labels if label not in [None, np.nan]
                    ]
                ),
                sort=sort,
                hovertemplate="%{label}<br>%{percent}<extra></extra>",
            )
        ]
    )


def scatter(
    x: pd.Series,
    y: pd.Series,
    mode: str = "markers",
    color: str = _REPORT_PALETTE[0],
    name: str = "Reference",
) -> go.Scatter:
    return go.Scatter(
        x=x,
        y=y,
        mode=mode,
        marker=dict(
            showscale=False,
            line=dict(width=0.5, color=color),
            color=color,
        ),
        opacity=0.7,
        name=name,
    )


def histogram(
    x: pd.Series,
    color: str = _REPORT_PALETTE[0],
    name: str = "Reference",
    **kwargs,
) -> go.Histogram:
    return go.Histogram(
        x=x,
        marker=dict(
            showscale=False,
            line=dict(width=0.5, color=color),
            color=color,
        ),
        name=name,
        opacity=0.7,
        showlegend=False,
        **kwargs,
    )


def get_auto_bins(x1: pd.Series, x2: pd.Series) -> dict:
    """
    Get common bin edges for the training and synthetic principal components.
    """
    data = pd.concat([x1, x2]).to_numpy()
    edges = np.histogram_bin_edges(data, bins="auto")

    # Bin start, end, and size
    start = edges[0]
    end = edges[-1]
    bin_size = edges[1] - edges[0]

    return dict(start=start, end=end, size=bin_size)


def generate_mia_figure(df: pd.DataFrame) -> go.Figure:
    # Done for legend ordering
    PROTECTION_COLUMN = "Protection"
    df[PROTECTION_COLUMN] = df[PROTECTION_COLUMN].astype("category")
    df[PROTECTION_COLUMN] = df[PROTECTION_COLUMN].cat.set_categories(
        [grade.value for grade in PrivacyGrade if grade.value != "Unavailable"], ordered=True
    )
    df.sort_values(by=PROTECTION_COLUMN, inplace=True, ascending=False)

    fig = pie(
        # the cast done bc ty infers this as a list of bools instead of str.
        labels=cast(list[str], df[PROTECTION_COLUMN].dropna().astype(str).tolist()),
        values=df["Attack Percentage"].replace({0: np.nan}),
        sort=False,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=24, b=0),
    )

    return fig


def generate_aia_figure(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    df.sort_values(by="Risk", inplace=True, ascending=True)
    for grade in PrivacyGrade:
        if grade.value == "Unavailable":
            continue
        filtered_df = df.query(f"Protection == '{grade.value}'")
        bar = go.Bar(
            x=filtered_df["Risk"],
            y=filtered_df["Column"],
            orientation="h",
            marker=dict(color=[INFERENCE_ATTACK_VALUES_FOR_GRAPHS[grade.value]] * filtered_df.shape[0]),
            width=0.3,
            showlegend=True,
            name=grade.value,
        )

        scatter = go.Scatter(
            x=df["Risk"],
            y=df["Column"],
            mode="markers",
            marker=dict(
                color="rgb(124, 135, 233)",
                size=10,
                line=dict(width=2, color=INFERENCE_ATTACK_VALUES_FOR_GRAPHS[grade.value]),
            ),
            showlegend=False,
            hoverinfo="none",
        )
        fig.add_trace(bar)
        fig.add_trace(scatter)

    fig.update_xaxes(title_text="Protection", range=[0, 11])
    fig.update_yaxes(title_text="Column", range=[-1, df.shape[0]])
    fig.update_layout(
        legend_traceorder="reversed",
        height=35 * df.shape[0] + 200,
        width=900,
        margin=dict(l=0, r=0, t=24, b=0),
    )

    return fig


def correlation_heatmap(matrix: pd.DataFrame, name: str = "Correlation") -> go.Figure:
    """
    Generate the figure for a list of correlation matrices.

    Arguments:
        matrix: The correlation matrix computed by dython.
        name: Name to use in add_trace.

    Returns:
        A plotly.graph_objects.Figure, a subplot with heatmaps.
    """
    fig = go.Figure()
    fields = [x if len(x) <= 15 else x[0:14] + "..." for x in matrix.columns]
    fig.add_trace(
        go.Heatmap(
            z=matrix,
            y=fields,
            x=fields,
            xgap=1,
            ygap=1,
            coloraxis="coloraxis",
            name=name,
        )
    )
    fig.update_layout(
        coloraxis=dict(
            colorscale=[
                [0.0, "#E8F3C6"],
                [0.25, "#94E2BA"],
                [0.5, "#31B8C0"],
                [0.75, "#4F78B3"],
                [1.0, "#76137F"],
            ],
            cmax=1.0,
            cmin=0,
        ),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=6, r=6, t=24, b=12),
    )
    fig.update_yaxes(dtick=1)
    return fig


def _generate_correlation_hovertext(corr_reference: pd.DataFrame, corr_output: pd.DataFrame, corr_diff: pd.DataFrame):
    hovertext = list()
    # Loop through the y values
    for y in corr_reference.columns:
        # Create one list per y value
        next_ylist = list()

        # Loop through the x values, add an entry in the above next_ylist for each x value
        # with the text to be displayed
        for x in corr_reference.columns:
            corr_reference_value = corr_reference[x][y]
            corr_output_value = corr_output[x][y]
            corr_diff_value = corr_diff[x][y]
            text = "x: " + x + "<br>y: " + y + "<br>Reference correlation: " + str(round(corr_reference_value, 2))
            text = text + "<br>Output correlation: " + str(round(corr_output_value, 2))
            text = text + "<br>Correlation difference: " + str(round(corr_diff_value, 2))
            next_ylist.append(text)

        # Append the ylist to the final hovertext list
        hovertext.append(next_ylist)
    return hovertext


def generate_combined_correlation_figure(
    reference_correlation: pd.DataFrame,
    output_correlation: pd.DataFrame,
    correlation_difference: pd.DataFrame,
) -> go.Figure:
    """
    Combine two of the correlation figures into a single row.

    Returns:
        a new Figure.
    """
    hovertext = _generate_correlation_hovertext(reference_correlation, output_correlation, correlation_difference)

    reference_correlation_figure = correlation_heatmap(reference_correlation, "Reference Correlations")
    output_correlation_figure = correlation_heatmap(output_correlation, "Output Correlations")
    correlation_difference_figure = correlation_heatmap(correlation_difference, "Difference of Correlations")

    fig = combine_subplots(
        figures=[
            reference_correlation_figure,
            output_correlation_figure,
            correlation_difference_figure,
        ],
        titles=[
            "Reference Correlations",
            "Output Correlations",
            "Correlation Difference",
        ],
    )

    fig.update_traces(hoverinfo="text", text=hovertext)

    fig.update_traces(yaxis="y1")

    fig.update_xaxes(visible=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(visible=False, showticklabels=False, row=1, col=2)
    fig.update_xaxes(visible=False, showticklabels=False, row=1, col=3)
    fig.update_yaxes(visible=False, showticklabels=False, row=1, col=1)

    fig.update_layout(height=400, width=900)

    return fig


def scatter_plot(x: pd.Series, y: pd.Series, color=_REPORT_PALETTE[0], maximum_points=5000) -> go.Figure:
    """
    Make a plotly scatter plot.

    Args:
        x: series of x values
        y: corresponding series of y values
        color: marker color, defaults to _REPORT_PALETTE[0]
        maximum_points: this plot will crash the browser page if there are too many points. Cap at this number of points.
            No callers currently attempt to change the default value.  If you use a value of 0, the code will
            make NO ATTEMPT to cap the number of points and your page may crash.

    Returns:
        A plotly.graph_objects.Figure

    """
    # Sample training set to equal synthetic set or vice versa
    if maximum_points == 0:
        sample_size = min(len(x), len(y))
    else:
        sample_size = min(len(x), len(y), maximum_points)
    if len(x) > sample_size:
        x = x.sample(n=sample_size, random_state=777)
    if len(y) > sample_size:
        y = y.sample(n=sample_size, random_state=777)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="markers", marker=dict(size=2, color=color), showlegend=False),
    )

    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")
    fig.update_layout(hovermode=False)

    return fig


def structure_stability_figure(reference: pd.DataFrame, output: pd.DataFrame) -> go.Figure:
    reference_scatter = scatter_plot(x=reference["pc1"], y=reference["pc2"])
    output_scatter = scatter_plot(x=output["pc1"], y=output["pc2"], color=_REPORT_PALETTE[1])
    fig = combine_subplots(
        figures=[reference_scatter, output_scatter],
        titles=["Reference Data", "Output Data"],
    )

    fig.update_layout(
        height=420,
        width=900,
        margin=dict(l=6, r=6, t=24, b=12),
    )
    fig.update_xaxes(title_text="pc1", row=1, col=1)
    fig.update_xaxes(title_text="pc1", row=1, col=2)
    fig.update_yaxes(title_text="pc2", row=1, col=1)
    fig.update_xaxes(matches="x")
    fig.update_yaxes(matches="y")

    return fig


def combine_subplots(
    figures: list[go.Figure],
    titles: list[str] | None = None,
    general_title: str | None = None,
    subplot_type: str = "xy",
    shared_xaxes=True,
    shared_yaxes=True,
    height=None,
    margin=None,
) -> go.Figure:
    """
    Take a list of go.Figures and make a single go.Figure out of them.  They will all be on one row.

    Args:
        figures: List of go.Figures to combine.
        titles: List of subplot titles, must be same length as number of traces.
        general_title: Title for the entire figure.
        subplot_type: see https://plotly.com/python/subplots/#subplots-types,
        shared_xaxes: Passed into plotly make_subplots call, see
            https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html
        shared_yaxes: Passed into plotly make_subplots call, see
            https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html
        height: Passed to fig.update_layout if present. Local jupyter notebooks can look wonky if this
            is absent, other environments are usually okay without it.
        margin: Passed to fig.update_layout if present. Updates the margin of the figure.

    Returns:
        a single new plotly.graph_objects.Figure.
    """
    specs = [[{"type": subplot_type}] * len(figures)]

    fig = make_subplots(
        rows=1,
        cols=len(figures),
        specs=specs,
        shared_xaxes=shared_xaxes,
        shared_yaxes=shared_yaxes,
        subplot_titles=titles,
    )
    for i, f in enumerate(figures):
        for t in f.select_traces():
            fig.add_trace(trace=t, row=1, col=i + 1)
    # This is surprisingly expensive. We only need the last layout, though,
    # as long as all the subplots are of the same "type"
    # (used to update with every subfig layout).
    fig.layout.update(figures[-1].layout)
    if general_title:
        fig.update_layout(title_text="<b>" + general_title + "</b>")
    if height:
        fig.update_layout(height=height)
    if margin:
        fig.update_layout(margin=margin)

    return fig


def bar_chart(reference_distribution: dict, output_distribution: dict) -> go.Figure:
    """
    Generate a bar chart for a categorical distribution.

    Arguments:
        reference_distribution: dict of (distribution value, count) from the reference df
        output_distribution: dict of (distribution value, count) from the output df

    Returns:
        A plotly.graph_objects.Figure
    """
    columns = sorted(set(reference_distribution.keys()).union(output_distribution.keys()))
    reference_values = []
    output_values = []

    for column in columns:
        reference_values.append(reference_distribution.get(column, 0.0))
        output_values.append(output_distribution.get(column, 0.0))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=columns,
            y=reference_values,
            name="Reference",
            marker=dict(color=_REPORT_PALETTE[0]),
            opacity=0.7,
            width=[] if len(columns) < 60 else [len(columns) / 150] * len(columns),
            hovertemplate="(%{x}, %{y:.2f})",
        )
    )
    fig.add_trace(
        go.Bar(
            x=columns,
            y=output_values,
            name="Output",
            marker=dict(color=_REPORT_PALETTE[1]),
            opacity=0.7,
            width=[] if len(columns) < 60 else [len(columns) / 150] * len(columns),
            hovertemplate="(%{x}, %{y:.2f})",
        )
    )
    fig.update_layout(
        yaxis_title_text="Percentage",
        bargap=_GRAPH_BARGAP,
        bargroupgap=_GRAPH_BARGROUPGAP,
        barmode="group",
        showlegend=False,
    )

    return fig


def histogram_figure(reference: pd.Series, output: pd.Series) -> go.Figure | None:
    """
    Generate a histogram distplot for a numeric distribution.

    Arguments:
        reference: The reference pd.Series for which we make the histogram.
        output: The output pd.Series for which we make the histogram.

    Returns:
        A plotly.graph_objects.Figure
    """
    fig = go.Figure()
    fig.update_layout(
        yaxis_title_text="Percentage",
        bargap=_GRAPH_BARGAP,
        bargroupgap=_GRAPH_BARGROUPGAP,
        showlegend=False,
    )

    reference_copy = pd.Series(reference)
    reference_copy.dropna(inplace=True)
    output_copy = pd.Series(output)
    output_copy.dropna(inplace=True)

    # Quantile, min and max will fail on empty Series. Fail fast and return empty fig.
    if len(reference_copy) == 0 or len(output_copy) == 0:
        return fig

    max_range = max(max(reference_copy), max(output_copy)) + 1
    min_range = min(min(reference_copy), min(output_copy))

    # Calculate bin size for the plot, handling edge case of no variance as needed.
    binsize = 1
    if min_range == max_range:
        max_range = min_range + binsize
    else:
        # number of bins/bin size match the ones in JS divenrgence calculation.
        bins = get_numeric_distribution_bins(reference, output)
        binsize = bins[1] - bins[0]

    xbins = dict(start=min_range, end=max_range, size=binsize)
    fig.add_trace(
        histogram(
            x=reference_copy,
            histnorm="percent",
            xbins=xbins,
            hovertemplate="(%{x}, %{y:.2f})",
        )
    ).add_trace(
        histogram(
            x=output_copy,
            color=_REPORT_PALETTE[1],
            name="Output",
            histnorm="percent",
            xbins=xbins,
            hovertemplate="(%{x}, %{y:.2f})",
        )
    )
    return fig


def generate_text_structure_similarity_figures(
    training_statistics: pd.DataFrame, synthetic_statistics: pd.DataFrame, title: str
) -> go.Figure | None:
    statistics_keys = [
        "sentence_count",
        "average_words_per_sentence",
        "average_characters_per_word",
    ]
    figures = []
    for key in statistics_keys:
        if training_statistics.per_record_statistics.empty or synthetic_statistics.per_record_statistics.empty:
            break
        figure = histogram_figure(
            training_statistics.per_record_statistics[key],
            synthetic_statistics.per_record_statistics[key],
        )
        figures.append(figure)
    if not figures:
        return None

    result = combine_subplots(
        figures,
        titles=[
            "Sentence Count",
            "Words Per Sentence",
            "Characters Per Word",
        ],
        height=400,
        general_title=title,
        shared_xaxes=False,
        shared_yaxes=False,
        margin=dict(l=0, r=0, t=64, b=0),
    )
    result.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return result


def generate_text_semantic_similarity_figures(
    training_pca: pd.DataFrame, synthetic_pca: pd.DataFrame, title: str
) -> go.Figure | None:
    figures = []
    for key in training_pca.columns:
        for subkey in synthetic_pca.columns:
            if key != subkey and key == training_pca.columns[-1]:
                continue
            fig = go.Figure()

            if key != subkey:
                fig.add_trace(
                    scatter(
                        x=training_pca[key],
                        y=training_pca[subkey],
                    )
                ).add_trace(
                    scatter(
                        x=synthetic_pca[key],
                        y=synthetic_pca[subkey],
                        color=_REPORT_PALETTE[1],
                        name="Output",
                    )
                )
            else:
                common_bins = get_auto_bins(
                    x1=training_pca[key],
                    x2=synthetic_pca[subkey],
                )
                fig.add_trace(
                    histogram(
                        x=training_pca[key],
                        histnorm="probability density",
                        xbins=common_bins,
                    )
                ).add_trace(
                    histogram(
                        x=synthetic_pca[subkey],
                        color=_REPORT_PALETTE[1],
                        name="Output",
                        histnorm="probability density",
                        xbins=common_bins,
                    )
                )

                fig.update_layout(
                    bargap=0,
                    barmode="overlay",
                )
            figures.append(fig)
    if not figures:
        return None
    result = combine_subplots(
        figures,
        height=400,
        general_title=title,
        shared_xaxes=False,
        shared_yaxes=False,
        margin=dict(l=0, r=0, t=64, b=0),
    )
    for key in training_pca.columns:
        for subkey in synthetic_pca.columns:
            if key != subkey and key == synthetic_pca.columns[-1]:
                continue
            col = synthetic_pca.columns.get_loc(key) + synthetic_pca.columns.get_loc(subkey) + 1  # ty: ignore[unsupported-operator]
            result.update_xaxes(
                title_text=key,
                row=1,
                col=col,
            ).update_yaxes(
                title_text=subkey,
                row=1,
                col=col,
            )

    result.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return result
