# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabular PII detection and replacement engine helpers.

Core algorithms for the tabular PII replacer:

* MVP detector (`detect_entities_mvp`) -- a fully programmatic, no-LLM/no-GPU
  approach: cardinality / column-dependency structural analysis, regex-based
  value-pattern detection (whole-column vs mixed), and fuzzy column-name matching.

The detector emits a plan shape that feeds the generation helpers:

* person entities -> demographically-matched persona sampling (PGM / managed / Faker);
* non-person entities (credit card, ip, uuid/unique_identifier, api key) ->
  Faker, with a stable ``original -> synthetic`` map per column.

Focused entity taxonomy:

* Person (whole-sale replacement per person):
  first_name, last_name, middle_name, full_name, email, phone_number,
  date_of_birth, street_address (street-line only), ssn, national_id,
  plus address parts city/state/zipcode (preserved-context helpers).
* Non-person (Faker): credit_debit_card, api_key, ipv4, ipv6, unique_identifier.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import string
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..defaults import default_managed_assets_path
from ..observability import get_logger

logger = get_logger(__name__)

try:  # Faker is the non-person generator (and an offline person fallback).
    from faker import Faker
except Exception:  # pragma: no cover - faker should be installed
    Faker = None  # type: ignore

# Data Designer constants (with offline-friendly fallbacks so the module imports
# even when DD is unavailable and only the MVP/Faker path is exercised).
try:
    from data_designer.config.utils.constants import (
        DEFAULT_AGE_RANGE,
        MAX_AGE,
        MIN_AGE,
    )
except Exception:  # pragma: no cover - environment dependent
    MIN_AGE, MAX_AGE, DEFAULT_AGE_RANGE = 0, 114, (18, 80)


# ===========================================================================
# Config
# ===========================================================================
@dataclass
class Config:
    """All knobs for detection + generation. Only sensible defaults required."""

    locale: str = "en_US"
    # Seed for persona/ID/faker generation. Env-overridable so a batched run can give
    # each row-batch a distinct seed -- otherwise every batch regenerates the SAME
    # unique-identifier sequence and they collide across batches (breaking injectivity).
    random_seed: int = field(default_factory=lambda: int(os.environ.get("PERSON_RANDOM_SEED", "42") or "42"))

    # Grouping (Safe-Synthesizer's group_training_records_by) is configured, not
    # auto-detected. A column is GROUP-level when single-valued within >= this
    # fraction of groups.
    group_constancy_threshold: float = 0.95
    # Whether the configured group key is itself a replacement candidate. When
    # True it is detected/replaced like any other (group-scoped) column -- e.g. a
    # `patient_id` grouping key is replaced as a unique_identifier. Row->group
    # mapping during apply always uses the ORIGINAL key values, so replacing the
    # output column is safe.
    replace_group_key: bool = True

    # Persona sampling.
    persona_backend: str = "managed"  # pgm | managed | faker
    sdg_pgms_src: str = "/root/sdg-pgms/src"
    managed_assets_path: str | None = None
    managed_sample_size: int | None = 150_000
    use_race_constraint: bool = True
    pool_min_size: int = 3_000
    pool_oversample: int = 6

    # --- MVP structural / value-pattern detection ---
    # Below this n_unique a column is treated as low-cardinality categorical and
    # is NOT a replacement candidate (e.g. sex / race / state / event_type).
    low_card_max: int = 12
    # Minimum percent of non-null values matching the dominant concrete pattern
    # required to classify a column as structured (not free text).
    dominant_pattern_min_coverage: float = 85.0
    # Legacy high bar for whole-column entity homogeneity (metadata only).
    value_match_threshold: float = 0.999
    # Structural unique_identifier gate: near-unique columns only.
    id_unique_ratio: float = 0.999
    # Free-text columns: long, varied object columns.
    free_text_min_len: float = 25.0
    free_text_min_unique_ratio: float = 0.3
    # A free-text column must read like natural-language PROSE: at least this many
    # whitespace-separated tokens on average. Used to reject single-token columns
    # such as URLs or short code columns (avg ~1 word). The MVP length +
    # unique_ratio gate above already excludes low-cardinality phrase columns.
    free_text_min_words: float = 1.5
    # Fuzzy column-name match acceptance.
    name_fuzzy_threshold: float = 0.86

    # --- Free-text name-token aliasing (BOTH modes) ---
    # When a person is identified only by a full name (no separate first/last columns),
    # also propagate the individual name TOKENS into free text so honorific/partial
    # mentions are caught consistently (e.g. provider "John Smith" -> synthetic "Robert
    # Jones" also rewrites a later "Dr. Smith" -> "Dr. Jones"). Tokens shorter than
    # freetext_alias_min_token_len are skipped to avoid over-matching short common words.
    freetext_name_token_aliases: bool = True
    freetext_alias_min_token_len: int = 3

    # --- Non-person value-pattern inference (Faker template) ---
    # When True, non-person ID columns are regenerated from an inferred template
    # that keeps constant characters literal (e.g. a 'pmc-' prefix) and constrains
    # low-entropy positions to their observed alphabet (e.g. first digit in {6,8})
    # instead of fully randomizing every character.
    infer_value_patterns: bool = True
    # A variable position whose observed alphabet is <= this many distinct chars is
    # emitted as an explicit class (e.g. "[68]"); larger -> a family token (#/^/@/&/%/*).
    pattern_class_max: int = 6
    # Characters covering < this fraction of a position are dropped as noise so a
    # rare outlier (e.g. a single 'pmc-7...') doesn't widen the template.
    pattern_rare_char_frac: float = 0.01
    # Cap on distinct sample values scanned when inferring a template.
    pattern_sample_cap: int = 5000

    def __post_init__(self) -> None:
        if self.managed_assets_path is None:
            self.managed_assets_path = str(default_managed_assets_path())


# Person identifying attributes (replaced by a synthetic persona).
PERSON_FIELD_LABELS = {
    "first_name",
    "last_name",
    "middle_name",
    "full_name",
    "email",
    "phone_number",
    "date_of_birth",
    "street_address",
    "city",
    "state",
    "zipcode",
    "ssn",
    "national_id",
}
# Non-person entities (replaced via Faker, consistent per value).
NON_PERSON_ENTITIES = {"credit_debit_card", "api_key", "ipv4", "ipv6", "unique_identifier"}
# Only sex and race are used to condition synthetic-name generation.
DEMO_KEYS = ("sex", "race")


# ===========================================================================
# Section 1 -- Structural analysis (cardinality, dependencies, group scope)
# ===========================================================================
def _sample_values(series: pd.Series, k: int = 8) -> list[str]:
    vals = series.dropna().unique().tolist()
    return [str(v) for v in vals[:k]]


def column_stats(df: pd.DataFrame) -> dict[str, dict]:
    n = len(df)
    stats: dict[str, dict] = {}
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        nun = int(non_null.nunique())
        stats[col] = {
            "dtype": str(s.dtype),
            "n_unique": nun,
            "unique_ratio": round(nun / n, 4) if n else 0.0,
            "null_rate": round(float(s.isna().mean()), 4),
            "samples": _sample_values(non_null),
        }
    return stats


def within_group_constancy(df: pd.DataFrame, key: str, col: str) -> float:
    """Fraction of `key` groups in which `col` has a single distinct value."""
    if key == col:
        return 1.0
    g = df.groupby(key, dropna=True)[col].nunique(dropna=True)
    return float((g <= 1).mean()) if len(g) else 0.0


def classify_columns_by_scope(df: pd.DataFrame, group_key: str, threshold: float) -> tuple[list[str], list[str]]:
    """Split non-key columns into (group-constant, record-varying)."""
    const_cols, vary_cols = [], []
    for c in df.columns:
        if c == group_key:
            continue
        if within_group_constancy(df, group_key, c) >= threshold:
            const_cols.append(c)
        else:
            vary_cols.append(c)
    return const_cols, vary_cols


def scoped_column_stats(
    df: pd.DataFrame, group_key: str | None, group_constancy_threshold: float = 0.95
) -> dict[str, dict]:
    """`column_stats`, but with cardinality measured against the right denominator.

    For a GROUP-CONSTANT column the uniqueness/cardinality of a per-group attribute
    should be measured against the number of groups, not the number of rows: its
    `unique_ratio` is recomputed as ``n_unique / n_groups``. This prevents a
    per-group identifier (e.g. one MRN per patient) from looking low-cardinality
    merely because its value repeats across every row of the group. Record-level
    columns keep the per-row denominator. Each entry also gets a ``scope`` tag.
    """
    stats = column_stats(df)
    if not (group_key and group_key in df.columns):
        for c in stats:
            stats[c]["scope"] = "record"
        return stats
    n_groups = int(df[group_key].nunique(dropna=True)) or len(df)
    const_cols, _ = classify_columns_by_scope(df, group_key, group_constancy_threshold)
    const_set = set(const_cols)
    for c in stats:
        if c == group_key:
            # The group key is group-constant by definition (n_unique == n_groups),
            # so its cardinality is also measured per group -> unique_ratio == 1.0.
            nun = stats[c]["n_unique"]
            stats[c]["unique_ratio"] = round(nun / n_groups, 4) if n_groups else 0.0
            stats[c]["scope"] = "key"
        elif c in const_set:
            nun = stats[c]["n_unique"]
            stats[c]["unique_ratio"] = round(nun / n_groups, 4) if n_groups else 0.0
            stats[c]["scope"] = "group"
        else:
            stats[c]["scope"] = "record"
    return stats


def column_dependency(df: pd.DataFrame, a: str, b: str) -> str:
    """Functional-dependency relationship between two columns.

    Returns one of "1-1", "1-n", "n-1", "n-n":
      * a_det_b: every value of `a` maps to a single value of `b`.
      * b_det_a: every value of `b` maps to a single value of `a`.
    "1-1" both; "n-1" only a_det_b (many a share one b); "1-n" only b_det_a; else "n-n".
    """
    sub = df[[a, b]].dropna()
    if sub.empty:
        return "n-n"
    a_det_b = sub.groupby(a)[b].nunique().max() <= 1
    b_det_a = sub.groupby(b)[a].nunique().max() <= 1
    if a_det_b and b_det_a:
        return "1-1"
    if a_det_b:
        return "n-1"
    if b_det_a:
        return "1-n"
    return "n-n"


# ===========================================================================
# Section 2 -- Value-pattern detection (regex; whole-column vs mixed)
# ===========================================================================
def _luhn_ok(digits: str) -> bool:
    if not digits.isdigit():
        return False
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Full-match (anchored) regexes used to classify a single cell's value.
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\Z")
_SSN_RE = re.compile(r"\d{3}-\d{2}-\d{4}\Z")
_IPV4_RE = re.compile(r"(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\Z")
_IPV6_RE = re.compile(r"(?=.*:)(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}\Z")
_UUID_RE = re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}\Z")
_PHONE_RE = re.compile(r"\+?\d[\d\-\.\s()]{7,}\d\Z")
_CARD_RE = re.compile(r"(?:\d[ -]?){12,18}\d\Z")
_API_PREFIXES = (
    "sk-",
    "pk-",
    "rk-",
    "ak-",
    "ghp_",
    "gho_",
    "github_pat_",
    "xoxb-",
    "xoxp-",
    "AKIA",
    "ASIA",
    "AIza",
    "Bearer ",
)
_API_RE = re.compile(r"[A-Za-z0-9_\-]{20,}\Z")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


# Ordered strftime formats for temporal value detection. First match wins.
# Datetime/time are checked before date-only formats. All date entries carry a
# separator so plain integers never match.
_DATETIME_FORMATS = [
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
]
_DATE_FORMATS = [
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%Y-%m-%d",
    "%m-%d-%Y",
    "%Y/%m/%d",
    "%m/%d/%y",
    "%d-%m-%Y",
    "%m/%Y",
    "%Y/%m",
    "%m-%Y",
    "%Y-%m",
]
_TIME_FORMATS = [
    "%H:%M:%S",
    "%H:%M",
    "%I:%M:%S %p",
    "%I:%M %p",
]
_ISO_DURATION_RE = re.compile(
    r"^P(?=\d|T)(?:\d+Y)?(?:\d+M)?(?:\d+D)?(?:T(?:\d+H)?(?:\d+M)?(?:\d+(?:\.\d+)?S)?)?$",
    re.IGNORECASE,
)
_HUMAN_DURATION_RE = re.compile(
    r"^\d+(?:\.\d+)?\s*(?:s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$",
    re.IGNORECASE,
)
_COMPACT_DURATION_RE = re.compile(r"^\d+[hms](?:\d+[hms])?$", re.IGNORECASE)

# Temporal entity labels identified only to keep columns out of free-text; never replaced.
IDENTIFIED_NOT_REPLACED_ENTITIES = frozenset({"date", "datetime", "time", "duration"})


def _try_strftime_formats(value: str, formats: list[str]) -> str | None:
    for fmt in formats:
        try:
            datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
        return fmt
    return None


def match_datetime_format(value: Any) -> str | None:
    """Return the strftime format for a datetime cell, or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or not any(ch.isdigit() for ch in s):
        return None
    if "T" not in s and " " not in s:
        return None
    return _try_strftime_formats(s, _DATETIME_FORMATS)


def match_date_format(value: Any) -> str | None:
    """Return the strftime format a date cell parses as (e.g. ``%m/%d/%Y``), or None.

    Cheap pre-filters (must contain a digit and a ``/`` or ``-`` separator) keep
    this from attempting ``strptime`` on obvious non-dates such as plain numbers,
    emails, or free text.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or not any(ch.isdigit() for ch in s) or ("/" not in s and "-" not in s):
        return None
    if match_datetime_format(s):
        return None
    return _try_strftime_formats(s, _DATE_FORMATS)


def match_time_format(value: Any) -> str | None:
    """Return the strftime format for a time-only cell, or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s or ":" not in s or not any(ch.isdigit() for ch in s):
        return None
    if "/" in s or "-" in s:
        return None
    if "::" in s or re.search(r"[A-Fa-f]", s):
        return None
    return _try_strftime_formats(s, _TIME_FORMATS)


def match_duration_format(value: Any) -> str | None:
    """Return a duration pattern label (``iso8601`` or ``human``), or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    if _ISO_DURATION_RE.match(s):
        return "iso8601"
    if _HUMAN_DURATION_RE.match(s) or _COMPACT_DURATION_RE.match(s):
        return "human"
    return None


def match_value_entity(value: Any) -> str | None:
    """Best entity label for one cell value via anchored regexes, or None.

    Order matters (specific -> general). Used both for column classification and
    to decide a single-pattern structured column's entity.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    if not s:
        return None
    if _EMAIL_RE.match(s):
        return "email"
    if _IPV4_RE.match(s):
        return "ipv4"
    if _IPV6_RE.match(s) and s.count(":") >= 2:
        # Guard against HH:MM:SS time strings (2 colons, all-decimal, no '::'):
        # require real ipv6 structure -- '::' compression, a hex letter, or 3+ colons.
        if "::" in s or s.count(":") >= 3 or re.search(r"[A-Fa-f]", s):
            return "ipv6"
    if _UUID_RE.match(s):
        return "unique_identifier"
    if _SSN_RE.match(s):
        return "ssn"
    if match_datetime_format(s):
        return "datetime"
    if match_date_format(s):
        return "date"
    if match_time_format(s):
        return "time"
    if match_duration_format(s):
        return "duration"
    if any(s.startswith(p) for p in _API_PREFIXES):
        return "api_key"
    d = _digits(s)
    if _CARD_RE.match(s) and 13 <= len(d) <= 19 and _luhn_ok(d):
        return "credit_debit_card"
    if _PHONE_RE.match(s) and 10 <= len(d) <= 15:
        return "phone_number"
    if _API_RE.match(s) and any(c.isdigit() for c in s) and any(c.isalpha() for c in s):
        return "api_key"
    return None


def column_value_signature(series: pd.Series, sample: int = 500) -> dict:
    """Legacy wrapper; prefer ``analyze_column_patterns``."""
    analysis = analyze_column_patterns(series, Config(), sample=sample)
    frac = analysis["coverage"] / 100.0 if analysis["coverage"] else 0.0
    return {
        "dominant": analysis["entity"],
        "match_fraction": round(frac, 4),
        "mixed": not analysis["structured"],
        "counts": {},
    }


def match_value_pattern(value: Any) -> tuple[str | None, str | None]:
    """Return ``(entity, concrete_pattern)`` for one cell value."""
    entity = match_value_entity(value)
    if entity is None:
        return None, None
    if entity == "date":
        fmt = match_date_format(value)
        return ("date", fmt) if fmt else (None, None)
    if entity == "datetime":
        fmt = match_datetime_format(value)
        return ("datetime", fmt) if fmt else (None, None)
    if entity == "time":
        fmt = match_time_format(value)
        return ("time", fmt) if fmt else (None, None)
    if entity == "duration":
        fmt = match_duration_format(value)
        return ("duration", fmt) if fmt else (None, None)
    return entity, None


def analyze_column_patterns(series: pd.Series, cfg: Config, sample: int = 500) -> dict:
    """Compute dominant entity/pattern and coverage for a column.

    Returns ``entity``, ``pattern`` (concrete format when known), ``coverage``
    (percent rounded to 1 decimal), and ``structured`` (coverage >= min threshold).
    """
    non_null = series.dropna()
    if non_null.empty:
        return {"entity": None, "pattern": None, "coverage": 0.0, "structured": False}

    if len(non_null) > sample:
        non_null = non_null.sample(sample, random_state=0)
    total = len(non_null)
    counts: Counter = Counter()
    for v in non_null:
        entity, pattern = match_value_pattern(v)
        if entity is None:
            counts[("__unmatched__", None)] += 1
        else:
            counts[(entity, pattern)] += 1

    typed = {k: c for k, c in counts.items() if k[0] != "__unmatched__"}
    if not typed:
        return {"entity": None, "pattern": None, "coverage": 0.0, "structured": False}

    (entity, pattern), dominant_count = max(typed.items(), key=lambda kv: kv[1])
    coverage = round(dominant_count / total * 100, 1)
    structured = coverage >= cfg.dominant_pattern_min_coverage
    return {
        "entity": entity,
        "pattern": pattern,
        "coverage": coverage,
        "structured": structured,
    }


def detect_entity_by_value(series: pd.Series, threshold: float = 0.999) -> str | None:
    """Dominant entity label if the WHOLE column is a single pattern, else None."""
    analysis = analyze_column_patterns(series, Config(dominant_pattern_min_coverage=threshold * 100))
    if analysis["entity"] and analysis["coverage"] >= round(threshold * 100, 1):
        return analysis["entity"]
    return None


def column_date_pattern(series: pd.Series, sample: int = 500) -> str | None:
    """Modal strftime format across a column's date-parseable values, or None."""
    analysis = analyze_column_patterns(series, Config(), sample=sample)
    if analysis["entity"] == "date":
        return analysis["pattern"]
    return None


# ===========================================================================
# Section 3 -- Column-name detection (regex + fuzzy)
# ===========================================================================
ENTITY_NAME_PATTERNS: dict[str, list[str]] = {
    # person identifying fields
    "first_name": [r"first[_ ]?name", r"^fname$", r"given[_ ]?name"],
    "last_name": [r"last[_ ]?name", r"^lname$", r"surname", r"family[_ ]?name"],
    "middle_name": [r"middle[_ ]?name", r"^mname$"],
    "full_name": [
        r"full[_ ]?name",
        r"^name$",
        r"provider[_ ]?name",
        r"physician",
        r"doctor",
        r"clinician",
        r"\bnurse\b",
        r"attending",
        r"patient[_ ]?name",
        r"person[_ ]?name",
        r"customer[_ ]?name",
        r"client[_ ]?name",
        r"employee[_ ]?name",
    ],
    "email": [r"e[-_ ]?mail"],
    "phone_number": [r"phone", r"mobile", r"telephone", r"\bfax\b"],
    "date_of_birth": [r"date[_ ]?of[_ ]?birth", r"birth[_ ]?date", r"\bdob\b"],
    "street_address": [r"street", r"address", r"addr"],
    "city": [r"^city$", r"\btown\b"],
    "state": [r"^state$", r"province"],
    "zipcode": [r"\bzip\b", r"postcode", r"postal"],
    "ssn": [r"\bssn\b", r"social[_ ]?security"],
    "national_id": [r"national[_ ]?id", r"\bnino\b", r"passport", r"tax[_ ]?id", r"\bnin\b"],
    # non-person entities
    "credit_debit_card": [
        r"credit[_ ]?card",
        r"debit[_ ]?card",
        r"\bcard[_ ]?(no|number|num)\b",
        r"\bccn\b",
        r"\bpan\b",
    ],
    "api_key": [r"api[_ ]?key", r"secret[_ ]?key", r"access[_ ]?key", r"\btoken\b", r"\bapikey\b"],
    "ipv4": [r"ipv4", r"ip[_ ]?addr", r"^ip$"],
    "ipv6": [r"ipv6"],
    "unique_identifier": [r"\buuid\b", r"\bguid\b", r"\b\w*_?id$", r"identifier", r"\bkey$"],
}

# Only sex and race condition synthetic-name generation, so those are the only
# demographics detected. (Age/DOB/occupation do not constrain persona sampling.)
DEMO_LABEL_PATTERNS: dict[str, list[str]] = {
    "sex": [r"^sex$", r"gender"],
    "race": [r"race", r"ethnic"],
}

_SECONDARY_ROLE_PATTERNS = [
    r"provider",
    r"physician",
    r"doctor",
    r"clinician",
    r"\bnurse\b",
    r"attending",
    r"staff",
    r"referr",
]
_ORG_KEYWORDS = [
    "hospital",
    "clinic",
    "center",
    "centre",
    "health",
    "medical",
    "institute",
    "department",
    "dept",
    "university",
    "inc",
    "llc",
    "ltd",
    "corp",
    "system",
    "associates",
]


def _match_label(col: str, patterns: dict[str, list[str]]) -> str | None:
    name = col.lower()
    for label, regexes in patterns.items():
        if any(re.search(rg, name) for rg in regexes):
            return label
    return None


# Curated, specific keyword spellings per label for the fuzzy backstop. These are
# deliberately NOT generic single words (no bare "name"/"date"/"id"), so a typo'd
# variant matches but unrelated columns (e.g. "event_name", "event_date") do not.
FUZZY_KEYWORDS: dict[str, list[str]] = {
    "first_name": ["firstname", "fname", "givenname", "forename"],
    "last_name": ["lastname", "lname", "surname", "familyname"],
    "middle_name": ["middlename", "mname"],
    "full_name": [
        "fullname",
        "patientname",
        "providername",
        "personname",
        "customername",
        "clientname",
        "employeename",
        "physicianname",
    ],
    "email": ["email", "emailaddress", "emailaddr"],
    "phone_number": ["phonenumber", "telephone", "mobilephone", "phoneno"],
    "date_of_birth": ["dateofbirth", "birthdate", "birthday"],
    "street_address": ["streetaddress", "homeaddress", "mailingaddress"],
    "ssn": ["socialsecurity", "socialsecuritynumber"],
    "national_id": ["nationalid", "passportnumber", "taxid"],
    "credit_debit_card": ["creditcard", "debitcard", "cardnumber"],
    "api_key": ["apikey", "secretkey", "accesskey", "apitoken"],
    "ipv4": ["ipaddress", "ipaddr"],
    "unique_identifier": ["uniqueidentifier"],
}


def fuzzy_match_label(col: str, patterns: dict[str, list[str]], threshold: float) -> str | None:
    """Regex match first; else fuzzy-compare the WHOLE column name to curated keywords.

    Comparing the entire normalized column name (not individual tokens) to specific
    keyword spellings catches typos/variants (e.g. "emial", "fname") without
    misfiring on columns that merely contain a generic token like "name" or "date".
    """
    exact = _match_label(col, patterns)
    if exact:
        return exact
    joined = re.sub(r"[^a-z0-9]+", "", col.lower())
    if not joined:
        return None
    best_label, best = None, 0.0
    for label, keys in FUZZY_KEYWORDS.items():
        for key in keys:
            sc = difflib.SequenceMatcher(None, joined, key).ratio()
            if sc > best:
                best_label, best = label, sc
    return best_label if best >= threshold else None


def _role_for_column(col: str) -> str:
    name = col.lower()
    if any(re.search(rg, name) for rg in _SECONDARY_ROLE_PATTERNS):
        return "secondary_person"
    return "primary_person"


def looks_like_person_name(value: Any) -> bool:
    """Light guard so an organization (e.g. 'St. Mary's Hospital') isn't a person."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "n/a"}:
        return False
    if any(kw in s.lower() for kw in _ORG_KEYWORDS):
        return False
    tokens = [t for t in re.split(r"\s+", s) if t]
    return 1 <= len(tokens) <= 5


def _freetext_structural_ok(series: pd.Series, stat: dict, cfg: Config) -> bool:
    """MVP structural gate: long enough + varied enough to be a free-text column."""
    if series.dtype != object:
        return False
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    return (
        non_null.str.len().mean() >= cfg.free_text_min_len
        and stat.get("unique_ratio", 0.0) >= cfg.free_text_min_unique_ratio
    )


def _freetext_prose_ok(series: pd.Series, cfg: Config) -> bool:
    """Reads like prose: multi-token on average (rejects single-token URL/code columns)."""
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    return non_null.str.split().str.len().mean() >= cfg.free_text_min_words


def detect_free_text_columns(df: pd.DataFrame, stats: dict[str, dict], exclude: set[str], cfg: Config) -> list[str]:
    cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if _freetext_structural_ok(df[col], stats[col], cfg) and _freetext_prose_ok(df[col], cfg):
            cols.append(col)
    return cols


# ===========================================================================
# Section 4 -- MVP detector (programmatic)
# ===========================================================================
def classify_column(series: pd.Series, stat: dict, cfg: Config) -> dict:
    """Per-column pattern metadata for discovery diagnostics."""
    if stat["n_unique"] <= cfg.low_card_max:
        kind = "low_card"
    else:
        analysis = analyze_column_patterns(series, cfg)
        kind = "structured" if analysis["structured"] else "mixed"
    analysis = analyze_column_patterns(series, cfg)
    return {
        "kind": kind,
        "dominant_entity": analysis["entity"],
        "dominant_pattern": analysis["pattern"],
        "dominant_pattern_coverage": analysis["coverage"],
        "n_unique": stat["n_unique"],
        "unique_ratio": stat["unique_ratio"],
    }


def _non_person_entry(col: str, analysis: dict, *, pattern: str | None = None) -> dict:
    return {
        "column": col,
        "entity": analysis["entity"],
        "pattern": pattern if pattern is not None else analysis["pattern"],
        "dominant_pattern_coverage": analysis["coverage"],
    }


def _detect_subset_mvp(df_subset: pd.DataFrame, stats: dict, cfg: Config) -> dict:
    """Programmatic person + non-person detection over a column subset."""
    fields_by_role: dict[str, dict[str, str]] = {}
    field_meta_by_role: dict[str, dict[str, dict]] = {}
    demo_by_role: dict[str, dict[str, str]] = {}
    non_person: list[dict] = []
    identified_not_replaced: list[str] = []
    consumed: set[str] = set()

    for col in df_subset.columns:
        stat = stats[col]
        series = df_subset[col]
        name_label = fuzzy_match_label(col, ENTITY_NAME_PATTERNS, cfg.name_fuzzy_threshold)
        analysis = analyze_column_patterns(series, cfg)
        value_entity = analysis["entity"] if analysis["structured"] else None
        demo_label = _match_label(col, DEMO_LABEL_PATTERNS)
        high_card = stat["n_unique"] > cfg.low_card_max

        # Non-person entity (value evidence wins; name as fallback). Gated on
        # high cardinality so we never touch low-card categoricals.
        np_entity = None
        if value_entity in NON_PERSON_ENTITIES:
            np_entity = value_entity
        elif name_label in NON_PERSON_ENTITIES and high_card:
            # unique_identifier by name needs near-unique values to qualify.
            if name_label != "unique_identifier" or stat["unique_ratio"] >= cfg.id_unique_ratio:
                np_entity = name_label
        if np_entity:
            entry = (
                _non_person_entry(col, analysis)
                if value_entity == np_entity
                else {
                    "column": col,
                    "entity": np_entity,
                    "pattern": None,
                    "dominant_pattern_coverage": analysis["coverage"] if value_entity else None,
                }
            )
            non_person.append(entry)
            consumed.add(col)
            logger.runtime.info(
                f"[PII Replacement] Structured non-person column {col!r} (entity={np_entity}, "
                f"pattern={entry.get('pattern')}, coverage={entry.get('dominant_pattern_coverage')})"
            )
            continue

        # Birth dates are replaced by age-preserving perturbation that does not
        # depend on which person they belong to, so they are emitted as a
        # (non-person) structured column keyed by value/scope rather than a
        # persona field. Detection is by column name; the concrete strftime
        # pattern + coverage drive whole-column vs per-value formatting at apply.
        if name_label == "date_of_birth" and high_card:
            non_person.append(
                {
                    "column": col,
                    "entity": "date_of_birth",
                    "pattern": analysis["pattern"] if analysis["structured"] else None,
                    "dominant_pattern_coverage": analysis["coverage"] if analysis["structured"] else None,
                }
            )
            consumed.add(col)
            logger.runtime.info(
                f"[PII Replacement] Birth-date column {col!r} (entity=date_of_birth, "
                f"pattern={analysis['pattern']}, coverage={analysis['coverage']}) — "
                "perturbed per record/group (not persona-tied)"
            )
            continue

        # Demographics (preserved; used only to constrain persona sampling).
        if demo_label:
            role = _role_for_column(col)
            demo_by_role.setdefault(role, {}).setdefault(demo_label, col)

        # Person identifying field (by name, or by person-ish value pattern).
        person_label = None
        if name_label in PERSON_FIELD_LABELS:
            person_label = name_label
        elif value_entity in {"email", "phone_number", "ssn"}:
            person_label = value_entity
        # Cardinality gate: identifying person fields (names/email/phone/ssn/dob) are
        # high-cardinality. Require high_card for NAME-matched fields so a low-card
        # categorical whose name merely contains a keyword (e.g. "Phone Service" -> phone,
        # "Streaming Movies") is not mistaken for PII. Value-pattern matches are exempt
        # (an email/ssn/phone-shaped column is identifying regardless).
        value_backed = value_entity in {"email", "phone_number", "ssn"}
        if person_label and person_label not in ("city", "state", "zipcode") and (high_card or value_backed):
            role = _role_for_column(col)
            fields_by_role.setdefault(role, {}).setdefault(person_label, col)
            # Attach dominant concrete pattern + coverage when the values have one
            # (e.g. a date_of_birth column's strftime format). Name-like fields have
            # no concrete pattern, so nothing is attached and the plan omits it.
            if analysis["pattern"]:
                field_meta_by_role.setdefault(role, {}).setdefault(
                    person_label,
                    {"pattern": analysis["pattern"], "dominant_pattern_coverage": analysis["coverage"]},
                )
            consumed.add(col)
            continue

        # Generic temporal columns (date/datetime/time/duration) detected by dominant
        # pattern. Identify only to keep them out of the free-text path; they are NOT
        # replaced and are excluded from the replacement plan entirely. Birth dates are
        # handled as a person field (``date_of_birth``), not here.
        if (
            value_entity in IDENTIFIED_NOT_REPLACED_ENTITIES
            and name_label != "date_of_birth"
            and demo_label is None
            and high_card
        ):
            identified_not_replaced.append(col)
            consumed.add(col)
            logger.runtime.info(
                f"[PII Replacement] Identified temporal column {col!r} (entity={value_entity}, "
                f"pattern={analysis['pattern']}, coverage={analysis['coverage']}) — excluded from replacement plan"
            )

    roles = []
    for role in sorted(set(fields_by_role) | set(demo_by_role)):
        fields = fields_by_role.get(role, {})
        demo = demo_by_role.get(role, {})
        if not fields and not demo:
            continue
        roles.append(
            {
                "role": role,
                "scope": "record",
                "fields": fields,
                "field_meta": field_meta_by_role.get(role, {}),
                "demographics": {k: demo.get(k) for k in DEMO_KEYS},
            }
        )

    exclude = set(consumed)
    for r in roles:
        exclude |= {v for v in r["demographics"].values() if v}
    exclude |= set(identified_not_replaced)
    free_text = detect_free_text_columns(df_subset, stats, exclude, cfg)
    return {
        "roles": roles,
        "free_text_columns": free_text,
        "non_person": non_person,
        "identified_not_replaced": identified_not_replaced,
    }


def _attach_value_patterns(df: pd.DataFrame, non_person: list[dict], cfg: Config) -> None:
    """Attach a Faker-style `pattern` template to each non-person entity in place."""
    if not cfg.infer_value_patterns:
        return
    for ent in non_person:
        # Temporal columns and birth dates carry strftime/duration patterns, not
        # Faker-style bothify templates, so their pattern must not be overwritten.
        if ent.get("entity") in IDENTIFIED_NOT_REPLACED_ENTITIES or ent.get("entity") == "date_of_birth":
            continue
        col = ent.get("column")
        if col not in df.columns:
            continue
        pat = infer_value_pattern(df[col].dropna().unique(), cfg)
        if pat:
            ent["pattern"] = pat


def detect_entities_mvp(df: pd.DataFrame, group_key: str | None, stats: dict, cfg: Config) -> dict:
    """Programmatic (no-LLM) entity plan, two-pass by group/record scope."""
    classification = {c: classify_column(df[c], stats[c], cfg) for c in df.columns}

    def _run(sub_cols: list[str], scope: str) -> dict:
        if not sub_cols:
            return {"roles": [], "free_text_columns": [], "non_person": []}
        sub = df[sub_cols]
        out = _detect_subset_mvp(sub, {c: stats[c] for c in sub_cols}, cfg)
        for r in out["roles"]:
            r["scope"] = scope
        for e in out["non_person"]:
            e["scope"] = scope
        return out

    if group_key and group_key in df.columns:
        const_cols, vary_cols = classify_columns_by_scope(df, group_key, cfg.group_constancy_threshold)
        if const_cols:
            group_repr = df.groupby(group_key, dropna=True)[const_cols].first().reset_index()
            gout = _detect_subset_mvp(group_repr, {c: stats[c] for c in group_repr.columns}, cfg)
            for r in gout["roles"]:
                r["scope"] = "group"
            for e in gout["non_person"]:
                e["scope"] = "group"
        else:
            gout = {"roles": [], "free_text_columns": [], "non_person": []}
        rout = _run(vary_cols, "record")
        roles = gout["roles"] + rout["roles"]
        non_person = gout["non_person"] + rout["non_person"]
        free_text = list(dict.fromkeys(gout["free_text_columns"] + rout["free_text_columns"]))
    else:
        out = _run([c for c in df.columns], "record")
        roles, non_person, free_text = out["roles"], out["non_person"], out["free_text_columns"]

    # Optionally keep the configured group key out of replacement (legacy behavior).
    if not cfg.replace_group_key:
        non_person = [e for e in non_person if e["column"] != group_key]
    _attach_value_patterns(df, non_person, cfg)
    return {
        "group_key": group_key if (group_key and group_key in df.columns) else None,
        "free_text_columns": free_text,
        "roles": roles,
        "non_person": non_person,
        "column_classification": classification,
        "detector": "mvp",
    }


# ===========================================================================
# Section 5 -- Person instance extraction + demographic matching
# ===========================================================================
def detect_date_format(sample: str) -> str:
    return match_datetime_format(sample) or match_date_format(sample) or "%Y-%m-%d"


def age_from_dob(value: str) -> int | None:
    fmt = detect_date_format(value)
    try:
        dob = datetime.strptime(str(value).strip(), fmt).date()
    except (ValueError, TypeError):
        return None
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


CATEGORY_FUZZY_THRESHOLD = 0.82


def _norm_cat(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _alias_score(value_tokens: list[str], value_join: str, alias: str) -> float:
    a = _norm_cat(alias)
    if not a:
        return 0.0
    a_tokens = a.split()
    if len(a_tokens) == 1:
        if a in value_tokens:
            return 1.0
        return difflib.SequenceMatcher(None, value_join, a).ratio()
    if a in value_join:
        return 1.0
    if all(t in value_tokens for t in a_tokens):
        return 0.95
    return difflib.SequenceMatcher(None, value_join, a).ratio()


def fuzzy_category(
    value: Any, options: dict[str, list[str]], threshold: float = CATEGORY_FUZZY_THRESHOLD
) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    vj = _norm_cat(value)
    if not vj:
        return None
    vt = vj.split()
    best_key, best = None, 0.0
    for key, aliases in options.items():
        for alias in (key, *aliases):
            sc = _alias_score(vt, vj, alias)
            if sc > best:
                best_key, best = key, sc
    return best_key if best >= threshold else None


SEX_ALIASES: dict[str, list[str]] = {
    "Male": ["m", "man", "boy", "masculine", "male"],
    "Female": ["f", "woman", "girl", "feminine", "female"],
}


def norm_sex(value: Any) -> str | None:
    return fuzzy_category(value, SEX_ALIASES)


# --- Fine-grained map: dataset value -> a SINGLE PGM `ethnic_background` category.
# Keys here are EXACTLY the PGM's `ethnic_background` vocabulary (39 categories, see
# sdg-pgms USPersonGenerator). When the dataset carries a specific value (e.g.
# "Mexican", "Native Hawaiian", "Vietnamese") we map it 1:1 to the matching PGM
# category so the granularity present in the data is preserved for name generation,
# instead of collapsing into a broad bucket. Generic values (e.g. bare "Asian",
# "Hispanic") deliberately do NOT match here and fall through to the broad buckets
# below, which constrain to the whole subgroup.
_PGM_ETHNIC_ALIASES: dict[str, list[str]] = {
    "white": ["white", "caucasian", "european", "anglo", "euro american"],
    "black": ["black", "african american", "african", "afro caribbean", "afro", "afro american"],
    "east asian": ["east asian", "chinese", "japanese", "korean", "taiwanese", "mongolian", "han chinese"],
    "south asian": [
        "south asian",
        "asian indian",
        "indian american",
        "pakistani",
        "bangladeshi",
        "sri lankan",
        "nepali",
        "nepalese",
        "desi",
    ],
    "southeast asian": [
        "southeast asian",
        "south east asian",
        "filipino",
        "filipina",
        "vietnamese",
        "thai",
        "indonesian",
        "malaysian",
        "cambodian",
        "khmer",
        "burmese",
        "laotian",
        "hmong",
    ],
    "central asian": ["central asian", "kazakh", "uzbek", "turkmen", "kyrgyz", "tajik", "afghan"],
    "asian other": ["asian other", "other asian"],
    "mexican": ["mexican", "chicano", "chicana"],
    "puerto rican": ["puerto rican", "puerto rico", "boricua"],
    "cuban": ["cuban"],
    "dominican": ["dominican"],
    "guatemalan": ["guatemalan"],
    "honduran": ["honduran"],
    "salvadoran": ["salvadoran", "salvadorian", "el salvador"],
    "nicaraguan": ["nicaraguan"],
    "panamanian": ["panamanian"],
    "costa rican": ["costa rican", "costa rica"],
    "argentinean": ["argentinean", "argentine", "argentinian"],
    "bolivian": ["bolivian"],
    "chilean": ["chilean"],
    "colombian": ["colombian"],
    "ecuadorian": ["ecuadorian", "ecuadorean"],
    "paraguayan": ["paraguayan"],
    "peruvian": ["peruvian"],
    "uruguayan": ["uruguayan"],
    "venezuelan": ["venezuelan"],
    "spaniard": ["spaniard"],
    "spanish": ["spanish"],
    "spanish american": ["spanish american"],
    "hispanic or latino other": ["hispanic or latino other", "other hispanic", "other latino"],
    "other central american": ["other central american"],
    "other south american": ["other south american"],
    "american indian": ["american indian", "native american", "amerindian", "first nations", "indigenous american"],
    "alaska native": ["alaska native", "alaskan native", "alaska"],
    "latin american indian": ["latin american indian"],
    "pacific islander": [
        "pacific islander",
        "native hawaiian",
        "hawaiian",
        "samoan",
        "tongan",
        "fijian",
        "chamorro",
        "guamanian",
    ],
    "polynesian": ["polynesian", "maori", "tahitian"],
    "melanesian": ["melanesian", "papuan"],
    "micronesian": ["micronesian", "marshallese", "palauan"],
}
# Closed vocabulary used by the PGM persona sampler.
PGM_ETHNIC_CATEGORIES: tuple[str, ...] = tuple(_PGM_ETHNIC_ALIASES)

# --- Broad buckets: fallback for generic/coarse inputs. Each maps to the FULL set
# of PGM subcategories so a generic value (e.g. "Asian") constrains to the whole
# subgroup rather than dropping any subcategory.
_ETHNICITY_GROUPS: dict[str, set[str]] = {
    "white": {"white"},
    "black": {"black"},
    "asian": {"asian other", "east asian", "south asian", "central asian", "southeast asian"},
    "hispanic": {
        "argentinean",
        "bolivian",
        "chilean",
        "colombian",
        "costa rican",
        "cuban",
        "dominican",
        "ecuadorian",
        "guatemalan",
        "hispanic or latino other",
        "honduran",
        "latin american indian",
        "mexican",
        "nicaraguan",
        "other central american",
        "other south american",
        "panamanian",
        "paraguayan",
        "peruvian",
        "puerto rican",
        "salvadoran",
        "spaniard",
        "spanish",
        "spanish american",
        "uruguayan",
        "venezuelan",
    },
    "american_indian": {"american indian", "latin american indian", "alaska native"},
    "pacific": {"polynesian", "pacific islander", "melanesian", "micronesian"},
}
_ETHNICITY_ALIASES: dict[str, list[str]] = {
    "white": ["white", "caucasian", "european", "anglo"],
    "black": ["black", "african american", "african", "afro caribbean", "afro"],
    "asian": [
        "asian",
        "east asian",
        "south asian",
        "southeast asian",
        "asian indian",
        "chinese",
        "japanese",
        "korean",
        "filipino",
        "vietnamese",
    ],
    "hispanic": ["hispanic", "latino", "latina", "latinx", "latin american", "spanish"],
    "american_indian": [
        "american indian",
        "alaska native",
        "native american",
        "indigenous",
        "first nations",
        "amerindian",
    ],
    "pacific": ["native hawaiian", "hawaiian", "pacific islander", "pacific", "polynesian", "samoan"],
}


def ethnicity_to_pgm(value: Any) -> list[str] | None:
    """Map a dataset race/ethnicity value to PGM `ethnic_background` category(ies).

    Prefers a fine-grained 1:1 mapping (preserving the dataset's granularity); falls
    back to a broad bucket (the whole subgroup) only for generic/coarse inputs.
    """
    fine = fuzzy_category(value, _PGM_ETHNIC_ALIASES)
    if fine:
        return [fine]
    grp = fuzzy_category(value, _ETHNICITY_ALIASES)
    return sorted(_ETHNICITY_GROUPS[grp]) if grp else None


def race_to_sfv(race_value: Any, cfg: Config) -> dict[str, list[str]] | None:
    if not cfg.use_race_constraint:
        return None
    cats = ethnicity_to_pgm(race_value)
    return {"ethnic_background": cats} if cats else None


def _sval(value: Any) -> str | None:
    return None if pd.isna(value) else str(value)


def extract_age(row: pd.Series, demo: dict) -> int | None:
    if demo.get("age") and pd.notna(row.get(demo["age"])):
        try:
            return int(float(row[demo["age"]]))
        except (ValueError, TypeError):
            pass
    if demo.get("dob") and pd.notna(row.get(demo["dob"])):
        return age_from_dob(row[demo["dob"]])
    return None


def _instance_is_person(field_cols: dict, originals: dict) -> bool:
    if "full_name" in field_cols and "full_name" in originals:
        return looks_like_person_name(originals["full_name"])
    return True


def _representative_row(gdf: pd.DataFrame, field_cols: dict) -> pd.Series:
    name_cols = [c for lab, c in field_cols.items() if lab in ("first_name", "last_name", "full_name")]
    if name_cols:
        mask = gdf[name_cols].notna().any(axis=1)
        if mask.any():
            return gdf[mask].iloc[0]
    return gdf.iloc[0]


def _make_instance(role, scope, match, originals, field_cols, demo, row, cfg, field_meta=None) -> dict:
    sex = norm_sex(row[demo["sex"]]) if demo.get("sex") else None
    # `age` is kept ONLY for free-text age-drift propagation (note text tracking the
    # regenerated DOB); it no longer constrains persona matching. Persona name realism
    # depends on sex + race (ethnic_background) only -- see PgmPersonaPool / the PGM,
    # where names are conditioned on sex and ethnic_background, not age/occupation.
    age = extract_age(row, demo)
    race_val = row[demo["race"]] if demo.get("race") and pd.notna(row.get(demo["race"])) else None
    return {
        "role": role,
        "scope": scope,
        "match": match,
        "field_cols": dict(field_cols),
        "field_meta": dict(field_meta or {}),
        "originals": {lab: str(v) for lab, v in originals.items()},
        "sex": sex,
        "age": age,
        # Raw race value is kept for audit output. Matching uses select_field_values,
        # set programmatically by ethnicity_to_pgm.
        "race_raw": _sval(race_val),
        "select_field_values": race_to_sfv(race_val, cfg),
    }


def extract_instances(df: pd.DataFrame, plan: dict, cfg: Config) -> list[dict]:
    instances: list[dict] = []
    gk = plan["group_key"]
    for role in plan["roles"]:
        field_cols = {lab: c for lab, c in role["fields"].items() if c in df.columns}
        if not field_cols:
            continue
        demo = role["demographics"]
        field_meta = role.get("field_meta") or {}
        if role["scope"] == "group" and gk:
            for gval, gdf in df.groupby(gk, dropna=True):
                rep = _representative_row(gdf, field_cols)
                originals = {lab: rep[c] for lab, c in field_cols.items() if pd.notna(rep[c])}
                if not originals or not _instance_is_person(field_cols, originals):
                    continue
                inst = _make_instance(
                    role["role"], "group", ("group", gval), originals, field_cols, demo, rep, cfg, field_meta
                )
                inst["group_key"] = gk
                inst["row_indices"] = list(gdf.index)
                instances.append(inst)
        else:
            # Group rows by signature so each record instance knows ALL the rows it
            # covers (its unit's row indices), used for the replacement-map unit key.
            sig_rows: dict[tuple, list] = {}
            sig_first: dict[tuple, pd.Series] = {}
            for idx, row in df.iterrows():
                sig = tuple((c, _sval(row[c])) for c in field_cols.values())
                if all(v is None for _, v in sig):
                    continue
                sig_rows.setdefault(sig, []).append(idx)
                sig_first.setdefault(sig, row)
            for sig, idxs in sig_rows.items():
                row = sig_first[sig]
                originals = {lab: row[c] for lab, c in field_cols.items() if pd.notna(row[c])}
                if not originals or not _instance_is_person(field_cols, originals):
                    continue
                match = ("record", {c: _sval(row[c]) for c in field_cols.values() if pd.notna(row[c])})
                inst = _make_instance(role["role"], "record", match, originals, field_cols, demo, row, cfg, field_meta)
                inst["group_key"] = gk
                inst["row_indices"] = list(idxs)
                instances.append(inst)
    return instances


# ===========================================================================
# Section 6 -- Persona generation backends (pgm / managed / faker)
# ===========================================================================
def _pgm_persona(row: pd.Series) -> dict:
    p = {k: (None if pd.isna(v) else v) for k, v in row.items()}
    if p.get("postcode") in (None, "") and p.get("zipcode") not in (None, ""):
        p["postcode"] = p["zipcode"]
    return p


def _load_pgm_generator(cfg: Config):
    """Import sdg-pgms' USPersonGenerator (en_US). Returns None if unavailable."""
    import sys

    if cfg.locale != "en_US":
        logger.runtime.warning(f"[pgm] LOCALE={cfg.locale} not supported by sdg-pgms (en_US only); falling back.")
        return None
    try:
        if cfg.sdg_pgms_src not in sys.path:
            sys.path.insert(0, cfg.sdg_pgms_src)
        if "sudachipy" not in sys.modules:
            import types as _types

            _stub = _types.ModuleType("sudachipy")

            class _StubDict:
                def __init__(self, *a, **k):
                    pass

                def create(self, *a, **k):
                    return None

            _stub.dictionary = _types.SimpleNamespace(Dictionary=_StubDict)
            sys.modules["sudachipy"] = _stub
            sys.modules["sudachipy.dictionary"] = _stub.dictionary
        from pgms.generators.us_person_generator import USPersonGenerator

        return USPersonGenerator()
    except Exception as exc:
        logger.runtime.warning(f"[pgm] USPersonGenerator unavailable ({exc}); falling back to managed/faker.")
        return None


def prepare_managed_assets(assets_root: str, locale: str, sample_size: int | None) -> str:
    import duckdb

    src = Path(assets_root) / "datasets" / f"{locale}.parquet"
    if sample_size is None or not src.exists():
        return assets_root
    cache_root = Path.home() / ".cache" / "pii_replacer" / f"{locale}_{sample_size}"
    dst = cache_root / "datasets" / f"{locale}.parquet"
    if dst.exists():
        return str(cache_root)
    con = duckdb.connect()
    try:
        total = con.execute(f"select count(*) from '{src}'").fetchone()[0]
        if total <= sample_size:
            return assets_root
        dst.parent.mkdir(parents=True, exist_ok=True)
        con.execute(f"copy (select * from '{src}' using sample {int(sample_size)} rows) to '{dst}' (format parquet)")
        logger.runtime.info(f"Prepared downsampled person dataset: {total:,} -> ~{sample_size:,} rows ({dst})")
        return str(cache_root)
    finally:
        con.close()


def make_person_loader(cfg: Config):
    if not Path(cfg.managed_assets_path).exists():
        logger.runtime.warning(f"Managed assets not found at {cfg.managed_assets_path}; using Faker person sampler.")
        return None
    try:
        from data_designer.engine.resources.person_reader import create_person_reader
        from data_designer.engine.sampling_gen.entities.person import load_person_data_sampler

        assets_root = prepare_managed_assets(cfg.managed_assets_path, cfg.locale, cfg.managed_sample_size)
        reader = create_person_reader(assets_root)
        return partial(load_person_data_sampler, reader=reader)
    except Exception as exc:
        logger.runtime.warning(f"Managed person reader unavailable ({exc}); using Faker person sampler.")
        return None


class PgmPersonaPool:
    """Growable pool of fresh PGM personas with without-replacement matching."""

    def __init__(self, generator, seed: int, initial: int):
        self._gen = generator
        self._rng = np.random.default_rng(seed)
        self._pool: list[dict] = []
        # Vectorized columns (rebuilt on grow) for fast candidate filtering. Persona
        # names are conditioned (in the PGM) on sex + ethnic_background only, so those
        # are the only attributes we match on -- age/DOB/occupation are NOT used.
        self._sex_arr = np.array([], dtype=object)
        self._eth_arr = np.array([], dtype=object)
        self._used = np.zeros(0, dtype=bool)
        self._grow(initial)

    def _grow(self, n: int) -> None:
        df = self._gen.generate_samples(int(max(1, n)))
        new = [_pgm_persona(r) for _, r in df.iterrows()]
        sex, eth = [], []
        for p in new:
            self._pool.append(p)
            sex.append(norm_sex(p.get("sex")) or "")
            eth.append(str(p.get("ethnic_background", "")).strip().lower())
        self._sex_arr = np.concatenate([self._sex_arr, np.array(sex, dtype=object)])
        self._eth_arr = np.concatenate([self._eth_arr, np.array(eth, dtype=object)])
        self._used = np.concatenate([self._used, np.zeros(len(new), dtype=bool)])

    def __len__(self) -> int:
        return len(self._pool)

    def _candidates(self, sex, eth_set, use_sex, use_eth) -> np.ndarray:
        mask = ~self._used
        if use_sex and sex:
            mask &= self._sex_arr == sex
        if use_eth and eth_set:
            mask &= np.isin(self._eth_arr, list(eth_set))
        return np.flatnonzero(mask)

    def match_one(self, sex, sfv) -> dict:
        eth_list = (sfv or {}).get("ethnic_background")
        eth_set = {str(e).strip().lower() for e in eth_list} if eth_list else None
        # Prefer sex+race, then relax to sex-only, race-only, then anything.
        relax = [(True, True), (True, False), (False, True), (False, False)]
        for _ in range(3):
            for flags in relax:
                idxs = self._candidates(sex, eth_set, *flags)
                if idxs.size:
                    pick = int(self._rng.choice(idxs))
                    self._used[pick] = True
                    return self._pool[pick]
            self._grow(1000)
        idxs = np.flatnonzero(~self._used)
        if not idxs.size:
            self._grow(1000)
            idxs = np.flatnonzero(~self._used)
        pick = int(self._rng.choice(idxs))
        self._used[pick] = True
        return self._pool[pick]


class PersonaEngine:
    """Resolves a backend and assigns one synthetic persona per person instance."""

    def __init__(self, cfg: Config, n_instances: int):
        self.cfg = cfg
        self.backend = cfg.persona_backend if n_instances else "none"
        self.pgm_pool: PgmPersonaPool | None = None
        self.person_loader = None
        self.source_counts: Counter = Counter()
        if self.backend == "pgm":
            gen = _load_pgm_generator(cfg)
            if gen is not None:
                pool_n = max(cfg.pool_min_size, cfg.pool_oversample * max(1, n_instances))
                logger.runtime.info(f"[pgm] Generating one fresh persona pool of ~{pool_n:,} people (one-time)...")
                self.pgm_pool = PgmPersonaPool(gen, cfg.random_seed, pool_n)
            else:
                self.backend = "managed"
        if self.backend == "managed":
            self.person_loader = make_person_loader(cfg)

    def _generate_person_column(self, n, kind, sex, sfv, seed):
        from data_designer.config.sampler_params import (
            PersonFromFakerSamplerParams,
            PersonSamplerParams,
            SamplerType,
        )
        from data_designer.engine.sampling_gen.generator import DatasetGenerator
        from data_designer.engine.sampling_gen.schema_builder import SchemaBuilder

        builder = SchemaBuilder()
        # Age is intentionally NOT constrained (names depend on sex + race only); use
        # the full default adult range so the sampler is unbiased by the source age.
        age_range = list(DEFAULT_AGE_RANGE)
        if kind == "person":
            params = PersonSamplerParams(locale=self.cfg.locale, sex=sex, age_range=age_range, select_field_values=sfv)
            builder.add_column(name="person", sampler_type=SamplerType.PERSON, params=params)
            gen = DatasetGenerator(
                None, random_state=seed, person_generator_loader=self.person_loader, schema=builder.build()
            )
        else:
            params = PersonFromFakerSamplerParams(locale=self.cfg.locale, sex=sex, age_range=age_range)
            builder.add_column(name="person", sampler_type=SamplerType.PERSON_FROM_FAKER, params=params)
            gen = DatasetGenerator(None, random_state=seed, schema=builder.build())
        return [dict(p) for p in gen.generate(n)["person"].tolist()]

    def _sample_personas(self, n, sex, sfv, seed) -> tuple[list[dict], str]:
        attempts = []
        if self.person_loader is not None:
            if sfv:
                attempts.append(("person", sfv))
            attempts.append(("person", None))
        attempts.append(("faker", None))
        last_exc: Exception | None = None
        for kind, use_sfv in attempts:
            try:
                return self._generate_person_column(n, kind, sex, use_sfv, seed), kind
            except Exception as exc:
                last_exc = exc
                continue
        raise RuntimeError(f"Failed to sample personas: {last_exc}")

    def sample_one(self, sex: str | None = None, sfv: dict | None = None) -> dict | None:
        """Sample one fresh persona.

        Uses the PGM pool when available (without-replacement, conditioned on sex +
        ethnic_background when given), else the managed/Faker fallback. Returns a
        persona dict (with first_name/last_name) or None if no backend is available.
        """
        if self.pgm_pool is not None:
            p = self.pgm_pool.match_one(sex, sfv)
            self.source_counts["pgm"] += 1
            return p
        self._adhoc_n = getattr(self, "_adhoc_n", 0) + 1
        try:
            personas, source = self._sample_personas(1, sex, sfv, self.cfg.random_seed + 10_000 + self._adhoc_n)
        except Exception:
            return None
        self.source_counts[source] += 1
        return personas[0] if personas else None

    def assign(self, instances: list[dict]) -> None:
        if not instances:
            return
        if self.pgm_pool is not None:
            for inst in instances:
                inst["persona"] = self.pgm_pool.match_one(inst["sex"], inst["select_field_values"])
                inst["persona_source"] = "pgm"
                self.source_counts["pgm"] += 1
            return
        buckets: dict[Any, list[int]] = {}
        for idx, inst in enumerate(instances):
            buckets.setdefault(_constraint_signature(inst), []).append(idx)
        for b_idx, (sig, idxs) in enumerate(buckets.items()):
            sex, sfv_key = sig
            sfv = {k: list(v) for k, v in sfv_key} or None
            personas, source = self._sample_personas(len(idxs), sex, sfv, self.cfg.random_seed + b_idx)
            self.source_counts[source] += len(idxs)
            for inst_idx, persona in zip(idxs, personas):
                instances[inst_idx]["persona"] = persona
                instances[inst_idx]["persona_source"] = source


def _constraint_signature(inst: dict):
    sfv = inst["select_field_values"]
    sfv_key = tuple(sorted((k, tuple(v)) for k, v in (sfv or {}).items()))
    return (inst["sex"], sfv_key)


# ===========================================================================
# Section 8 -- Synthetic value mapping (person) + Faker engine (non-person)
# ===========================================================================
_TITLE_RE = re.compile(r"^(dr|mr|mrs|ms|miss|prof|sir|madam|mx)\.?\s+", re.IGNORECASE)


def split_title(value: str):
    m = _TITLE_RE.match(value.strip())
    if m:
        return value[: m.end()].strip(), value[m.end() :].strip()
    return None, value.strip()


def _stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16)


def _seeded_faker(seed: int, locale: str = "en_US"):
    if Faker is None:
        raise RuntimeError("faker is not installed")
    f = Faker(locale)
    f.seed_instance(seed)
    return f


def pattern_preserving_token(s: str, rng) -> str:
    out = []
    for ch in s:
        if ch.islower():
            out.append(rng.choice(string.ascii_lowercase))
        elif ch.isupper():
            out.append(rng.choice(string.ascii_uppercase))
        elif ch.isdigit():
            out.append(rng.choice(string.digits))
        else:
            out.append(ch)
    return "".join(out)


# --- Value-pattern (Faker-style template) inference -------------------------
# A template keeps constant characters literal and replaces variable positions
# with either an explicit class "[chars]" or a family token. Family tokens
# (ordered smallest alphabet first so we pick the tightest that still covers the
# observed characters):
#   #=digit  ^=A-Z  @=a-z  &=0-9A-Z  %=0-9a-z  *=0-9A-Za-z
# Literal occurrences of a special char are backslash-escaped.
_PATTERN_SPECIALS = set("#^@&%*[]\\")
_FAMILY_TOKENS: list[tuple[str, str]] = [
    ("#", string.digits),
    ("^", string.ascii_uppercase),
    ("@", string.ascii_lowercase),
    ("&", string.digits + string.ascii_uppercase),
    ("%", string.digits + string.ascii_lowercase),
    ("*", string.digits + string.ascii_letters),
]


def _literal_token(ch: str) -> str:
    return "\\" + ch if ch in _PATTERN_SPECIALS else ch


def _position_token(chars: list[str], cfg: Config) -> str:
    """Template token for one column position given the chars observed there."""
    counts = Counter(chars)
    total = sum(counts.values())
    keep = {c for c, n in counts.items() if total and n / total >= cfg.pattern_rare_char_frac}
    if not keep:
        keep = {counts.most_common(1)[0][0]}
    if len(keep) == 1:
        return _literal_token(next(iter(keep)))
    if len(keep) <= cfg.pattern_class_max:
        body = "".join(sorted(keep)).replace("\\", "\\\\").replace("]", "\\]")
        return "[" + body + "]"
    for tok, charset in _FAMILY_TOKENS:
        if keep <= set(charset):
            return tok
    return "*"


def infer_value_pattern(values, cfg: Config) -> str | None:
    """Infer a Faker-style template from sample values (modal length).

    Constant positions stay literal; low-entropy positions become an explicit
    class like "[68]"; high-entropy positions become a family token. Returns None
    when the values are too irregular (mixed lengths) to template safely.
    """
    vals = [str(v) for v in values if v is not None and str(v) != "" and str(v).lower() != "nan"]
    if not vals:
        return None
    seen = list(dict.fromkeys(vals))[: cfg.pattern_sample_cap]
    modal_len = Counter(len(v) for v in seen).most_common(1)[0][0]
    if modal_len == 0:
        return None
    same = [v for v in seen if len(v) == modal_len]
    if len(same) < max(3, int(0.5 * len(seen))):
        return None  # too many distinct lengths -> not a fixed template
    cols = [[v[i] for v in same] for i in range(modal_len)]
    pattern = "".join(_position_token(c, cfg) for c in cols)
    # Reject a degenerate all-literal pattern (nothing would vary).
    if not re.search(r"(?<!\\)[#^@&%*\[]", pattern):
        return None
    return pattern


def generate_from_pattern(pattern: str, rng) -> str:
    """Generate one string from a template produced by `infer_value_pattern`."""
    fam = dict(_FAMILY_TOKENS)
    out: list[str] = []
    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            out.append(pattern[i + 1])
            i += 2
            continue
        if ch == "[":
            j, body = i + 1, []
            while j < n and pattern[j] != "]":
                if pattern[j] == "\\" and j + 1 < n:
                    body.append(pattern[j + 1])
                    j += 2
                    continue
                body.append(pattern[j])
                j += 1
            out.append(rng.choice(body) if body else "")
            i = j + 1
            continue
        if ch in fam:
            out.append(rng.choice(fam[ch]))
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def synth_date_value(original: str, fmt: str, rng) -> str | None:
    """Synthetic date by shifting the original within +/- 1 year, formatted with ``fmt``."""
    try:
        d = datetime.strptime(str(original).strip(), fmt).date()
    except (ValueError, TypeError):
        return None
    offset = rng.randint(-365, 365)
    if offset == 0:
        offset = 1
    return (d + timedelta(days=offset)).strftime(fmt)


def _synth_dob_programmatic(original: str, rng, fmt: str | None = None) -> str | None:
    """Synthetic DOB by perturbing the original birth date up to +/- 1 year.

    ``fmt`` is the dominant strftime format to parse/format with (used when the
    column has 100% coverage of a single pattern). When ``None``, the format is
    detected per value so minority formats are preserved.
    """
    fmt = fmt or detect_date_format(original)
    return synth_date_value(original, fmt, rng)


def synth_value(label: str, original: str, persona: dict, fake=None) -> str | None:
    """Map a sampled persona (+ optional deterministic Faker) onto one field.

    Only persona-sourced fields are handled here. Entity-driven columns
    (unique_identifier, date_of_birth, ...) are replaced via the non-person path
    and never reach this function.
    """
    p = persona or {}
    if label == "first_name":
        return p.get("first_name")
    if label == "last_name":
        return p.get("last_name")
    if label == "middle_name":
        return p.get("middle_name")
    if label == "email":
        return p.get("email_address") or (fake.email() if fake else None)
    if label == "phone_number":
        return p.get("phone_number") or (fake.phone_number() if fake else None)
    if label == "ssn":
        return p.get("ssn") or (fake.ssn() if fake else None)
    if label == "national_id":
        if fake is not None:
            return pattern_preserving_token(str(original), fake.random)
        return None
    if label == "street_address":
        parts = [str(x) for x in (p.get("street_number"), p.get("street_name")) if x not in (None, "")]
        new_street = " ".join(parts)
        if not new_street:
            return None
        # Preserve city/state/zip context: replace only the street line (before first comma).
        if "," in str(original):
            return new_street + "," + str(original).split(",", 1)[1]
        return new_street
    if label == "full_name":
        title, _ = split_title(str(original))
        full = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        if not full:
            return None
        return f"{title} {full}" if title else full
    return None


def _person_key(inst: dict):
    kind, payload = inst["match"]
    return payload if kind == "group" else dict(payload)


def _unit_key(scope: str, group_value, row_indices):
    """Shared unit identity for the replacement map, used to tie a unit's structured
    (person) replacements to its free-text replacements.

    group scope -> the group key's value; record scope -> the row index (a list only in
    the rare case one person signature spans multiple rows). Entries also carry ``scope``,
    so the two sections join on this value.
    """
    if scope == "group":
        return group_value
    idxs = [int(i) if isinstance(i, (int, np.integer)) else i for i in (row_indices or [])]
    return idxs[0] if len(idxs) == 1 else idxs


def compute_instance_synthetics(instances: list[dict], cfg: Config) -> None:
    """Fill inst['synthetic'], inst['syn_by_col'], inst['text_pairs'] in place."""
    for inst in instances:
        persona = inst.get("persona")
        synthetic: dict[str, str] = {}
        syn_by_col: dict[str, str] = {}
        if persona:
            seed = cfg.random_seed ^ _stable_hash(str(_person_key(inst)))
            fake = _seeded_faker(seed, cfg.locale) if Faker is not None else None
            for label, col in inst["field_cols"].items():
                original = inst["originals"].get(label)
                if original is None:
                    continue
                sv = synth_value(label, original, persona, fake)
                if sv is None or str(sv) == str(original):
                    continue
                synthetic[label] = str(sv)
                syn_by_col[col] = str(sv)
        inst["synthetic"] = synthetic
        inst["syn_by_col"] = syn_by_col
        inst["text_pairs"] = instance_text_pairs(inst, cfg)


_TEXT_NAME_LABELS = (
    "first_name",
    "last_name",
    "middle_name",
    "full_name",
    "email",
    "phone_number",
    "ssn",
    "national_id",
)


def instance_text_pairs(inst: dict, cfg: Config | None = None) -> list[tuple[str, str]]:
    syn, orig = inst.get("synthetic", {}), inst["originals"]
    pairs: dict[str, str] = {}
    for label in _TEXT_NAME_LABELS:
        if label in syn and label in orig:
            pairs[orig[label]] = syn[label]
    if {"first_name", "last_name"} <= set(syn) and {"first_name", "last_name"} <= set(orig):
        pairs[f"{orig['first_name']} {orig['last_name']}"] = f"{syn['first_name']} {syn['last_name']}"
    # Name-token aliases: for a person known only by a FULL name (no separate first/last
    # columns), also propagate each name token so honorific/partial mentions stay consistent
    # ("John Smith" -> "Robert Jones" also rewrites a later "Dr. Smith" -> "Dr. Jones"). Done
    # by positionally pairing the original vs synthetic tokens (titles stripped); only tokens
    # >= freetext_alias_min_token_len letters are aliased so short common words aren't hit.
    if cfg is None or cfg.freetext_name_token_aliases:
        min_len = cfg.freetext_alias_min_token_len if cfg is not None else 3
        for label in ("full_name",):
            if label not in syn or label not in orig:
                continue
            _, o_rest = split_title(str(orig[label]))
            _, s_rest = split_title(str(syn[label]))
            o_toks, s_toks = o_rest.split(), s_rest.split()
            if len(o_toks) == len(s_toks) and len(o_toks) > 1:
                for ot, st in zip(o_toks, s_toks):
                    if ot != st and len(ot) >= min_len and ot.isalpha():
                        pairs.setdefault(ot, st)
    # NOTE: no age-drift propagation -- the synthetic DOB is only perturbed within
    # +/- 1 year (see _synth_dob_programmatic), so any age mentioned in free text is at
    # most 1 year off, which is acceptable and not worth rewriting.
    return sorted(pairs.items(), key=lambda kv: len(kv[0]), reverse=True)


def instance_text_pair_labels(inst: dict, cfg: Config | None = None) -> dict[str, str]:
    """{original_value -> taxonomy label} for the same pairs ``instance_text_pairs`` emits.

    Used only to attach an entity label to MVP free-text propagations for the eval's
    precision/recall log (generation itself is label-agnostic). Mirrors the pair-building
    logic above so labels stay in sync.
    """
    syn, orig = inst.get("synthetic", {}), inst["originals"]
    labels: dict[str, str] = {}
    for label in _TEXT_NAME_LABELS:
        if label in syn and label in orig:
            labels[str(orig[label])] = label
    if {"first_name", "last_name"} <= set(syn) and {"first_name", "last_name"} <= set(orig):
        labels[f"{orig['first_name']} {orig['last_name']}"] = "full_name"
    if cfg is None or cfg.freetext_name_token_aliases:
        min_len = cfg.freetext_alias_min_token_len if cfg is not None else 3
        for label in ("full_name",):
            if label not in syn or label not in orig:
                continue
            _, o_rest = split_title(str(orig[label]))
            o_toks = o_rest.split()
            if len(o_toks) > 1:
                for i, ot in enumerate(o_toks):
                    if len(ot) >= min_len and ot.isalpha():
                        labels.setdefault(ot, "first_name" if i == 0 else "last_name")
    return labels


def _instance_demographics(inst: dict) -> dict:
    """Per-person demographics that conditioned name generation, with provenance.

    ``sex`` and ``ethnic_background`` are what the generator actually conditions on;
    raw ``race`` and ``age`` are included when present in the data for reference.
    """
    demo: dict = {}
    sex = inst.get("sex")
    if sex:
        demo["sex"] = {"value": sex, "source": "detected"}
    sfv = inst.get("select_field_values") or {}
    eth = sfv.get("ethnic_background")
    if eth:
        demo["ethnic_background"] = {
            "value": eth,
            "source": "detected",
        }
    if inst.get("race_raw"):
        demo["race"] = {"value": inst["race_raw"], "source": "detected"}
    if inst.get("age") is not None:
        demo["age"] = {"value": inst["age"], "source": "detected"}
    return demo


def build_person_replacement_map(insts: list[dict]) -> dict:
    persons: list[dict] = []
    for inst in insts:
        synthetic = inst.get("synthetic", {})
        if not synthetic:
            continue
        replacements = [
            {"original": inst["originals"][label], "label": label, "synthetic": syn}
            for label, syn in synthetic.items()
            if label in inst["originals"]
        ]
        if replacements:
            gval = inst["match"][1] if inst["match"][0] == "group" else None
            persons.append(
                {
                    "role": inst["role"],
                    "scope": inst["scope"],
                    "key": _person_key(inst),
                    "unit_key": _unit_key(inst["scope"], gval, inst.get("row_indices")),
                    "demographics": _instance_demographics(inst),
                    "replacements": replacements,
                }
            )
    return {"persons": persons, "n_persons": len(persons)}


# --- Non-person Faker engine -------------------------------------------------
def _fake_value(entity: str, original: str, fake) -> str:
    rng = fake.random
    if entity == "credit_debit_card":
        return fake.credit_card_number()
    if entity == "ipv4":
        return fake.ipv4()
    if entity == "ipv6":
        return fake.ipv6()
    if entity == "unique_identifier":
        if _UUID_RE.match(original.strip()):
            return str(fake.uuid4())
        return pattern_preserving_token(original, rng)
    if entity == "api_key":
        for pfx in _API_PREFIXES:
            if original.startswith(pfx):
                return pfx + pattern_preserving_token(original[len(pfx) :], rng)
        return pattern_preserving_token(original, rng)
    return pattern_preserving_token(original, rng)


def build_non_person_maps(original_df: pd.DataFrame, plan: dict, cfg: Config) -> dict[str, dict[str, str]]:
    """Stable original->synthetic map per non-person column (1-1, consistent).

    When the entity carries an inferred `pattern` template, synthetic values are
    drawn from it (keeping constant affixes like a 'pmc-' prefix and constrained
    positions like a first digit in {6,8}); otherwise the entity-specific Faker
    generator is used. Distinct originals always map to distinct synthetics.
    """
    maps: dict[str, dict[str, str]] = {}
    for ent in plan.get("non_person", []):
        col, entity = ent["column"], ent["entity"]
        if col not in original_df.columns:
            continue
        # Date columns are detected but not yet value-replaced (pass-through).
        if entity == "date":
            maps[col] = {}
            continue
        pattern = ent.get("pattern")
        fake = _seeded_faker(cfg.random_seed + _stable_hash(col), cfg.locale)
        rng = fake.random
        originals = [str(v) for v in original_df[col].dropna().unique()]
        used = set(originals)
        mapping: dict[str, str] = {}
        for sv in originals:
            new = None
            if pattern:
                for _ in range(50):
                    cand = generate_from_pattern(pattern, rng)
                    if cand and cand != sv and cand not in used:
                        new = cand
                        break
            if new is None:
                for _ in range(50):
                    cand = _fake_value(entity, sv, fake)
                    if cand and cand != sv and cand not in used:
                        new = cand
                        break
            if new is None:
                new = _fake_value(entity, sv, fake)
            if new and new != sv:
                mapping[sv] = new
                used.add(new)
        maps[col] = mapping
    return maps


# ===========================================================================
# Section 9 -- Free-text substitution helpers
# ===========================================================================
def build_text_substituter(pairs: list[tuple[str, str]]):
    """One alternation regex over all originals (longest-first), dict lookup repl."""
    pairs = [(o, s) for o, s in pairs if o]
    if not pairs:
        return None
    repl = {o: s for o, s in pairs}
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(o) for o, _ in pairs) + r")(?!\w)")

    def _sub(text: Any) -> Any:
        if not isinstance(text, str) or not text:
            return text
        return pattern.sub(lambda m: repl[m.group(1)], text)

    return _sub
