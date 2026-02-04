# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import re

from ..entity import Entity
from ..regex import Pattern, RegexPredictor

LAT_RE = r"[-+]?(90(\.0+)?|([1-8]?\d)(\.\d+)?)"
LAT_LABELS = [
    "latitude",
    re.compile(r"^lat"),
    re.compile(r"loc.*lat"),
    re.compile(r"lat.*loc"),
    re.compile(r"geo.*lat"),
    re.compile(r"lat.*geo"),
    re.compile(r"gis.*lat"),
    re.compile(r"lat.*gis"),
    re.compile(r"map.*lat"),
    re.compile(r"lat.*map"),
    re.compile(r"point.*lat"),
    re.compile(r"lat.*point"),
    re.compile(r"[_\-\.\s]lat$"),
]

LON_RE = r"[-+]?(180(\.0+)?|((1[0-7]\d)|([1-9]?\d))(\.\d+)?)"

LON_TOKEN = r"(?:lon|lng)"
LON_LABELS = [
    "longitude",
    re.compile(rf"^{LON_TOKEN}"),
    re.compile(rf"loc.*{LON_TOKEN}"),
    re.compile(rf"{LON_TOKEN}.*loc"),
    re.compile(rf"geo.*{LON_TOKEN}"),
    re.compile(rf"{LON_TOKEN}.*geo"),
    re.compile(rf"gis.*{LON_TOKEN}"),
    re.compile(rf"{LON_TOKEN}.*gis"),
    re.compile(rf"map.*{LON_TOKEN}"),
    re.compile(rf"{LON_TOKEN}.*map"),
    re.compile(rf"point.*{LON_TOKEN}"),
    re.compile(rf"{LON_TOKEN}.*point"),
    re.compile(rf"[_\-\.\s]{LON_TOKEN}$"),
]

GPS_LABELS = [
    "coords",
    "coordinates",
    "location",
    re.compile(r"^gps"),
    re.compile(r"^loc"),
]

SEP = r"\s*,\s*"

NEG_LABELS = ["latency", re.compile(r"^long[\W_]"), re.compile(r"^lat[^i]")]


class Latitude(RegexPredictor):
    def __init__(self):
        entity = Entity.LATITUDE

        likely_match = Pattern(
            pattern=LAT_RE,
            header_contexts=LAT_LABELS,
            ignore_raw_score=True,
            neg_header_contexts=NEG_LABELS,
        )

        super().__init__(name="latitude", entity=entity, patterns=[likely_match])


class Longitude(RegexPredictor):
    def __init__(self):
        entity = Entity.LONGITUDE

        likely_match = Pattern(
            pattern=LON_RE,
            header_contexts=LON_LABELS,
            ignore_raw_score=True,
            neg_header_contexts=NEG_LABELS,
        )

        super().__init__(name="longitude", entity=entity, patterns=[likely_match])


class GPSCoords(RegexPredictor):
    def __init__(self):
        lat_lon = Pattern(
            pattern=r"({lat}{sep}{lon}|{lon}{sep}{lat})".format(lat=LAT_RE, sep=SEP, lon=LON_RE),
            header_contexts=LON_LABELS + LAT_LABELS + GPS_LABELS,
            ignore_raw_score=True,
        )

        super().__init__(name="gps_coordinates", entity=Entity.GPS_COORDINATES, patterns=[lat_lon])
