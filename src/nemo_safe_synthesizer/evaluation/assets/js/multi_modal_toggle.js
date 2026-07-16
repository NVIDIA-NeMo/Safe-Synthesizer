const TRAINING_COLOR = "#3b82f6";
const SYNTHETIC_COLOR = "#f59e0b";
const SCORE_TIER_COLORS = {
    Excellent: "#22c55e",
    "Very Good": "#84cc16",
    Good: "#eab308",
    Moderate: "#f97316",
    Poor: "#ef4444",
};
const HEATMAP_SCALES = [
    [[0, "#e6f2ff"], [0.25, "#b3d9ff"], [0.5, "#66b3ff"], [0.75, "#3399ff"], [1, "#0066cc"]],
    [[0, "#fff5e6"], [0.25, "#ffe0b3"], [0.5, "#ffcc80"], [0.75, "#ffa64d"], [1, "#e67300"]],
    [[0, "#ffffff"], [0.25, "#fee2e2"], [0.5, "#fca5a5"], [0.75, "#f87171"], [1, "#dc2626"]],
];
const PLOTLY_ARRAY_TYPES = {
    i1: Int8Array,
    u1: Uint8Array,
    i2: Int16Array,
    u2: Uint16Array,
    i4: Int32Array,
    u4: Uint32Array,
    f4: Float32Array,
    f8: Float64Array,
};

function decodePlotlyArray(values) {
    if (Array.isArray(values)) {
        return values;
    }
    if (ArrayBuffer.isView(values)) {
        return Array.from(values);
    }
    const ArrayType = PLOTLY_ARRAY_TYPES[values?.dtype];
    if (!ArrayType || typeof values.bdata !== "string") {
        return [];
    }
    const binary = window.atob(values.bdata);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
    }
    const length = Math.floor(bytes.byteLength / ArrayType.BYTES_PER_ELEMENT);
    return Array.from(new ArrayType(bytes.buffer, 0, length));
}

function arcPoint(angle, centerX = 50, centerY = 55, radius = 38) {
    const radians = angle * Math.PI / 180;
    return {
        x: centerX + radius * Math.cos(radians),
        y: centerY - radius * Math.sin(radians),
    };
}

function scoreColor(score) {
    if (!Number.isFinite(score)) {
        return "#888888";
    }
    if (score >= 8) {
        return SCORE_TIER_COLORS.Excellent;
    }
    if (score >= 6) {
        return SCORE_TIER_COLORS["Very Good"];
    }
    if (score >= 4) {
        return SCORE_TIER_COLORS.Good;
    }
    if (score >= 2) {
        return SCORE_TIER_COLORS.Moderate;
    }
    return SCORE_TIER_COLORS.Poor;
}

function initializeSimpleScoreRing(ring, score, available) {
    const start = arcPoint(225);
    const end = arcPoint(-45);
    const track = ring.querySelector(".score-ring-simple-track");
    const progress = ring.querySelector(".score-ring-simple-progress");
    const fullPath = `M ${start.x} ${start.y} A 38 38 0 1 1 ${end.x} ${end.y}`;
    track.setAttribute("d", fullPath);

    if (!available || score <= 0) {
        progress.setAttribute("d", "");
        return;
    }

    const percentage = Math.min(100, Math.max(0, score * 10));
    const indicator = arcPoint(225 - percentage * 2.7);
    const progressPath = `M ${start.x} ${start.y} A 38 38 0 ${percentage > 50 ? 1 : 0} 1 ${indicator.x} ${indicator.y}`;
    ring.style.setProperty("--score-color", scoreColor(score));
    progress.setAttribute("d", progressPath);
}

function interpolateColor(colors, progress) {
    const position = progress * (colors.length - 1);
    const low = Math.floor(position);
    const high = Math.min(low + 1, colors.length - 1);
    const ratio = position - low;
    return colors[low].map((value, index) => Math.round(value + (colors[high][index] - value) * ratio));
}

function initializeGradientScoreRing(ring, score, available) {
    const canvas = ring.querySelector(".score-ring-canvas");
    const context = canvas.getContext("2d");
    const scale = 4;
    const centerX = 50 * scale;
    const centerY = 55 * scale;
    const radius = 38 * scale;
    const strokeWidth = 12 * scale;
    const startAngle = -225 * Math.PI / 180;
    const endAngle = 45 * Math.PI / 180;
    const sweep = endAngle - startAngle;
    const palette = available
        ? [[239, 68, 68], [249, 115, 22], [234, 179, 8], [132, 204, 22], [34, 197, 94]]
        : [[74, 74, 74], [74, 74, 74]];
    const segments = 256;

    context.clearRect(0, 0, canvas.width, canvas.height);
    context.lineWidth = strokeWidth;
    context.lineCap = "butt";
    for (let index = 0; index < segments; index += 1) {
        const segmentStart = startAngle + index * sweep / segments;
        const segmentEnd = segmentStart + sweep / segments + 0.005;
        const [red, green, blue] = interpolateColor(palette, index / (segments - 1));
        context.beginPath();
        context.arc(centerX, centerY, radius, segmentStart, segmentEnd, false);
        context.strokeStyle = `rgb(${red},${green},${blue})`;
        context.stroke();
    }

    const capRadius = strokeWidth / 2;
    [
        [startAngle, palette[0]],
        [endAngle, palette[palette.length - 1]],
    ].forEach(([angle, color]) => {
        context.beginPath();
        context.arc(
            centerX + radius * Math.cos(angle),
            centerY + radius * Math.sin(angle),
            capRadius,
            0,
            Math.PI * 2,
        );
        context.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
        context.fill();
    });

    const marker = ring.querySelector(".score-ring-marker");
    marker.hidden = !available;
    if (available) {
        const indicator = arcPoint(225 - Math.min(100, Math.max(0, score * 10)) * 2.7);
        marker.querySelectorAll("circle").forEach((circle) => {
            circle.setAttribute("cx", indicator.x);
            circle.setAttribute("cy", indicator.y);
        });
    }
}

function initializeScoreRings(container = document) {
    container.querySelectorAll("[data-score-ring]").forEach((ring) => {
        if (ring.dataset.initialized === "true") {
            return;
        }
        const score = Number(ring.dataset.score);
        const available = ring.dataset.available === "true";
        if (ring.dataset.size === "sm") {
            initializeSimpleScoreRing(ring, score, available);
        } else {
            initializeGradientScoreRing(ring, score, available);
        }
        ring.dataset.initialized = "true";
    });
}

function initializeScoreLabels(container = document) {
    container.querySelectorAll("[data-score-label]").forEach((label) => {
        const score = label.dataset.score === "" ? Number.NaN : Number(label.dataset.score);
        label.style.setProperty("--score-color", scoreColor(score));
    });
}

function resizePlotlyCharts(container) {
    if (!window.Plotly || !container) {
        return;
    }
    container.querySelectorAll(".js-plotly-plot").forEach((plot) => window.Plotly.Plots.resize(plot));
}

function makePlotResponsive(plot) {
    const layout = {...(plot.layout || {}), autosize: true};
    delete layout.width;
    plot.style.width = "100%";
    window.Plotly.react(plot, plot.data || [], layout, {displayModeBar: false, responsive: true});
}

function traceDataset(trace) {
    const name = String(trace.name || "").toLowerCase();
    if (name.includes("training") || name.includes("reference")) {
        return {color: TRAINING_COLOR, name: "Training Data"};
    }
    if (name.includes("synthetic") || name.includes("output")) {
        return {color: SYNTHETIC_COLOR, name: "Synthetic Data"};
    }
    return null;
}

function themeDatasetTraces(plot) {
    (plot.data || []).forEach((trace, index) => {
        const dataset = traceDataset(trace);
        if (!dataset || trace.type === "heatmap") {
            return;
        }
        const update = {name: dataset.name};
        if (trace.type !== "pie") {
            update["marker.color"] = dataset.color;
            update["marker.line.color"] = dataset.color;
        }
        if (trace.type === "scatter") {
            update["line.color"] = dataset.color;
            update["marker.size"] = 5;
            update["marker.opacity"] = 0.7;
        }
        window.Plotly.restyle(plot, update, [index]);
    });
}

function commonPlotUpdate(plot) {
    const update = {
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        "font.color": "rgba(255,255,255,0.72)",
        "font.family": "NVIDIA Sans, system-ui, sans-serif",
        "font.size": 11,
        "legend.font.color": "rgba(255,255,255,0.8)",
        "legend.bgcolor": "rgba(0,0,0,0)",
    };
    Object.keys(plot.layout || {})
        .filter((key) => /^(xaxis|yaxis)\d*$/.test(key))
        .forEach((axis) => {
            update[`${axis}.color`] = "rgba(255,255,255,0.72)";
            update[`${axis}.gridcolor`] = "rgba(255,255,255,0.1)";
            update[`${axis}.linecolor`] = "rgba(255,255,255,0.15)";
            update[`${axis}.zerolinecolor`] = "rgba(255,255,255,0.2)";
        });
    window.Plotly.relayout(plot, update);
}

function themeCorrelationPlot(plot) {
    let heatmapIndex = 0;
    const traces = (plot.data || []).map((trace) => {
        if (trace.type !== "heatmap") {
            return trace;
        }
        const themedTrace = {...trace};
        delete themedTrace.coloraxis;
        delete themedTrace.zmid;
        themedTrace.colorscale = HEATMAP_SCALES[heatmapIndex];
        themedTrace.zmin = 0;
        themedTrace.zmax = 1;
        themedTrace.showscale = true;
        themedTrace.colorbar = {
            thickness: 10,
            len: 1,
            x: [0.3, 0.66, 1.02][heatmapIndex],
            xpad: 2,
            tickfont: {size: 9},
        };
        heatmapIndex += 1;
        return themedTrace;
    });
    const layout = JSON.parse(JSON.stringify(plot.layout || {}));
    delete layout.coloraxis;
    Object.assign(layout, {
        height: 300,
        margin: {l: 10, r: 42, t: 28, b: 38},
        showlegend: false,
    });
    layout.xaxis = {...layout.xaxis, showticklabels: false};
    layout.yaxis = {...layout.yaxis, showticklabels: false};
    plot.style.height = "300px";
    window.Plotly.react(plot, traces, layout, {displayModeBar: false, responsive: true});
}

function themeDeepStructurePlot(plot) {
    [TRAINING_COLOR, SYNTHETIC_COLOR].forEach((color, index) => {
        if (!plot.data?.[index]) {
            return;
        }
        window.Plotly.restyle(plot, {
            "marker.color": color,
            "marker.opacity": 0.7,
            "marker.size": 5,
        }, [index]);
    });
}

function themeQualityPlot(plot, cardId, card) {
    const figureCount = card.querySelectorAll(".js-plotly-plot").length;
    if (cardId === "correlation-stability") {
        themeCorrelationPlot(plot);
        return;
    }
    if (cardId === "structure-stability") {
        themeDeepStructurePlot(plot);
    }
    const heights = {
        "structure-stability": 300,
        "semantic-similarity": 280,
        "structure-similarity": 280,
    };
    if (heights[cardId]) {
        const height = heights[cardId];
        plot.style.height = `${height}px`;
        const update = {
            height,
            margin: {l: 54, r: 16, t: 18, b: 48},
            showlegend: false,
        };
        if (figureCount === 1) {
            update["title.text"] = "";
        }
        window.Plotly.relayout(plot, update);
    }
}

function themeMembershipPlot(plot) {
    const source = plot.data?.[0];
    if (!source) {
        return;
    }
    const labels = ["Excellent", "Very Good", "Good", "Moderate", "Poor"];
    const sourceLabels = decodePlotlyArray(source.labels);
    const sourceValues = decodePlotlyArray(source.values);
    const values = labels.map((label) => {
        const index = sourceLabels.indexOf(label);
        return index >= 0 && Number.isFinite(Number(sourceValues[index])) ? Number(sourceValues[index]) : 0;
    });
    const data = [{
        type: "pie",
        labels,
        values,
        marker: {colors: labels.map((label) => SCORE_TIER_COLORS[label])},
        textinfo: "label+percent",
        textposition: "inside",
        textfont: {color: "#0c0c0c", size: 12},
        hoverinfo: "label+percent+value",
        hole: 0,
        sort: false,
    }];
    const layout = {
        paper_bgcolor: "transparent",
        plot_bgcolor: "transparent",
        font: {color: "rgba(255,255,255,0.8)", family: "NVIDIA Sans, system-ui, sans-serif"},
        height: 280,
        margin: {l: 20, r: 105, t: 10, b: 20},
        showlegend: true,
        legend: {x: 1.02, y: 1, xanchor: "left", yanchor: "top", font: {size: 11}, bgcolor: "transparent"},
    };
    plot.style.width = "100%";
    plot.style.maxWidth = "450px";
    plot.style.height = "280px";
    plot.parentElement.style.display = "flex";
    plot.parentElement.style.justifyContent = "center";
    window.Plotly.react(plot, data, layout, {displayModeBar: false, responsive: true});
}

function themeAttributePlot(plot) {
    const columns = new Set();
    (plot.data || []).forEach((trace, index) => {
        decodePlotlyArray(trace.y).filter((column) => column != null).forEach((column) => columns.add(column));
        if (trace.type === "scatter") {
            window.Plotly.restyle(plot, {visible: false}, [index]);
            return;
        }
        const originalName = String(trace.name || "");
        const color = SCORE_TIER_COLORS[originalName] || "#eab308";
        window.Plotly.restyle(plot, {
            "marker.color": color,
            name: originalName,
            width: 0.7,
        }, [index]);
    });
    const height = Math.max(280, columns.size * 40 + 80);
    plot.style.height = `${height}px`;
    window.Plotly.relayout(plot, {
        height,
        margin: {l: 140, r: 100, t: 10, b: 50},
        "xaxis.title.text": "Protection (out of 10)",
        "xaxis.range": [0, 10.5],
        "xaxis.dtick": 2,
        "yaxis.title.text": "Column",
        "yaxis.automargin": true,
        showlegend: true,
        "legend.x": 1.02,
        "legend.y": 1,
        "legend.xanchor": "left",
        "legend.yanchor": "top",
        bargap: 0.3,
    });
}

function themePlotlyCharts(container = document) {
    if (!window.Plotly) {
        return;
    }
    container.querySelectorAll(".js-plotly-plot").forEach((plot) => {
        commonPlotUpdate(plot);
        themeDatasetTraces(plot);
        const card = plot.closest("[data-metric-card]");
        if (!card) {
            return;
        }
        if (card.id === "mia") {
            themeMembershipPlot(plot);
        } else if (card.id === "aia") {
            themeAttributePlot(plot);
        } else {
            themeQualityPlot(plot, card.id, card);
        }
        makePlotResponsive(plot);
    });
}

function plainText(html) {
    const element = document.createElement("span");
    element.innerHTML = html || "";
    return element.textContent.trim();
}

function rebuildDistributionCharts(container) {
    if (!window.Plotly || !container || container.dataset.rebuilt === "true") {
        return;
    }
    const sourcePlots = Array.from(container.querySelectorAll(".js-plotly-plot"));
    if (!sourcePlots.length) {
        return;
    }
    const charts = [];
    sourcePlots.forEach((plot) => {
        const axisIds = Array.from(new Set((plot.data || []).map((trace) => trace.xaxis || "x")));
        const titles = Array.from(plot.layout?.annotations || []).map((annotation) => plainText(annotation.text));
        axisIds.forEach((axisId, axisIndex) => {
            const traces = (plot.data || [])
                .filter((trace) => (trace.xaxis || "x") === axisId)
                .map((trace) => {
                    const copy = JSON.parse(JSON.stringify(trace));
                    delete copy.xaxis;
                    delete copy.yaxis;
                    const dataset = traceDataset(copy);
                    if (dataset) {
                        copy.name = dataset.name;
                        copy.marker = {...copy.marker, color: dataset.color};
                    }
                    return copy;
                });
            if (traces.length) {
                charts.push({title: titles[axisIndex] || "", traces});
            }
        });
    });
    if (!charts.length) {
        return;
    }

    sourcePlots.forEach((plot) => window.Plotly.purge(plot));
    container.replaceChildren();
    container.dataset.rebuilt = "true";
    charts.forEach((chart) => {
        const wrapper = document.createElement("div");
        wrapper.className = "distribution-chart";
        const yLabel = document.createElement("span");
        yLabel.className = "distribution-chart-y-label";
        yLabel.textContent = "Percentage";
        const plot = document.createElement("div");
        plot.className = "distribution-chart-plot";
        wrapper.append(yLabel, plot);
        container.append(wrapper);
        window.Plotly.newPlot(plot, chart.traces, {
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            font: {color: "rgba(255,255,255,0.7)", family: "NVIDIA Sans, system-ui, sans-serif", size: 8},
            height: 120,
            margin: {l: 25, r: 5, t: 25, b: 30},
            barmode: chart.traces.some((trace) => trace.type === "histogram") ? "overlay" : "group",
            bargap: 0.15,
            bargroupgap: 0.1,
            showlegend: false,
            title: {text: chart.title, font: {size: 10, color: "rgba(255,255,255,0.8)"}, x: 0.5, y: 0.95},
            xaxis: {showgrid: false, showticklabels: false},
            yaxis: {showgrid: true, gridcolor: "rgba(255,255,255,0.1)", range: [0, 105]},
        }, {displayModeBar: false, responsive: true});
    });
}

function setMetricExpanded(card, expanded) {
    const toggle = card.querySelector("[data-metric-toggle]");
    const details = card.querySelector(".metric-card-details");
    if (!toggle || !details) {
        return;
    }
    toggle.setAttribute("aria-expanded", String(expanded));
    details.toggleAttribute("hidden", !expanded);
    card.classList.toggle("expanded", expanded);
    if (expanded) {
        window.requestAnimationFrame(() => {
            if (card.id === "distribution-stability") {
                rebuildDistributionCharts(details.querySelector(".distribution-charts"));
            }
            themePlotlyCharts(details);
            resizePlotlyCharts(details);
        });
    }
}

function toggleMetricCard(event) {
    if (event.target.closest("[data-tooltip-toggle]")) {
        return;
    }
    const card = event.currentTarget.closest("[data-metric-card]");
    setMetricExpanded(card, event.currentTarget.getAttribute("aria-expanded") !== "true");
}

function closeTooltips(except) {
    document.querySelectorAll("[data-tooltip-toggle]").forEach((button) => {
        if (button === except) {
            return;
        }
        button.setAttribute("aria-expanded", "false");
        document.getElementById(button.getAttribute("aria-controls"))?.setAttribute("hidden", "");
    });
}

function toggleTooltip(event) {
    event.stopPropagation();
    const button = event.currentTarget;
    const tooltip = document.getElementById(button.getAttribute("aria-controls"));
    const expanded = button.getAttribute("aria-expanded") !== "true";
    closeTooltips(button);
    button.setAttribute("aria-expanded", String(expanded));
    tooltip?.toggleAttribute("hidden", !expanded);
}

function reportViewForTarget(target) {
    if (!target) {
        return document.querySelector('[data-report-view="overview"]');
    }
    return target.matches("[data-report-view]") ? target : target.closest("[data-report-view]");
}

function activateReportView() {
    const target = document.getElementById(window.location.hash.slice(1));
    const view = reportViewForTarget(target);
    if (!view) {
        return;
    }
    document.querySelectorAll("[data-report-view]").forEach((candidate) => {
        candidate.classList.toggle("active", candidate === view);
    });
    document.querySelectorAll("[data-report-nav]").forEach((link) => {
        link.classList.toggle("active", link.dataset.reportNav === view.dataset.reportView);
    });
    const targetCard = target?.closest("[data-metric-card]");
    if (targetCard) {
        setMetricExpanded(targetCard, true);
    }
    window.requestAnimationFrame(() => {
        themePlotlyCharts(view);
        resizePlotlyCharts(view);
        if (target && target !== view) {
            target.scrollIntoView({block: "start"});
        } else {
            document.querySelector(".report-main")?.scrollTo({top: 0});
        }
    });
}

function toggleColumns(event) {
    const button = event.currentTarget;
    const expanded = button.getAttribute("aria-expanded") !== "true";
    document.querySelectorAll(".additional-column").forEach((row) => row.toggleAttribute("hidden", !expanded));
    button.setAttribute("aria-expanded", String(expanded));
    const label = button.querySelector("[data-columns-label]");
    if (label) {
        label.textContent = expanded ? "Show less" : `Show all ${button.dataset.columnsCount} columns`;
    }
}

document.querySelectorAll("[data-metric-toggle]").forEach((toggle) => toggle.addEventListener("click", toggleMetricCard));
document.querySelectorAll("[data-tooltip-toggle]").forEach((toggle) => toggle.addEventListener("click", toggleTooltip));
document.querySelectorAll("[data-dismiss]").forEach((button) => {
    button.addEventListener("click", () => button.closest("[data-dismissible]")?.remove());
});
document.querySelector("[data-columns-toggle]")?.addEventListener("click", toggleColumns);
document.addEventListener("click", (event) => {
    if (!event.target.closest(".metric-tooltip, [data-tooltip-toggle]")) {
        closeTooltips();
    }
});
document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
        closeTooltips();
    }
});
window.addEventListener("hashchange", activateReportView);
initializeScoreRings();
initializeScoreLabels();
themePlotlyCharts();
activateReportView();
document.fonts?.ready.then(() => {
    themePlotlyCharts();
    resizePlotlyCharts(document.querySelector(".report-view.active"));
});
