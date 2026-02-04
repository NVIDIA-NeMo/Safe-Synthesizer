# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .entity import Entity

spacy = None


def entities_to_html(text: str, entities: list):
    """Visualize detected entities as HTML
    text: raw text containing entities
    entities: a list of ``NERPrediction`` objects

    Returns: an HTML string
    """
    palette = [
        "#7aecec",
        "#bfeeb7",
        "#feca74",
        "#ff9561",
        "#aa9cfc",
        "#c887fb",
        "#9cc9cc",
        "#ffeb80",
        "#ff8197",
        "#ff8197",
        "#f0d0ff",
        "#bfe1d9",
        "#e4e7d2",
    ]

    colors = {}
    for index, name in enumerate(Entity):
        colors[name.name] = palette[index % len(palette)]

    options = {"ents": colors.keys(), "colors": colors}

    # Manually create display object in SpaCy format
    ents = []

    for x in sorted(entities, key=lambda i: i.start):
        ents.append({"start": x.start, "end": x.end, "label": x.label})

    ex = [{"text": text, "ents": ents, "title": None}]
    html = spacy.displacy.render(ex, style="ent", options=options, manual=True)
    return html
