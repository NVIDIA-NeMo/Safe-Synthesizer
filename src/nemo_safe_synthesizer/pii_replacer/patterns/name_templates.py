# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name-part templates for names and emails: read a column's convention, write it back."""

from __future__ import annotations

import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence

import pandas as pd

from ...observability import get_logger
from ..entities import Config
from ..models import DetectedField, SamePersonBundle
from .evidence import PATTERN_SAMPLE_SIZE, dominant_format
from .value_templates import (
    generate_from_pattern,
    matching_template,
    ranked_value_patterns,
    value_matches_template,
)

logger = get_logger(__name__)

DOMAIN_PLACEHOLDER = "{domain}"
EMAIL_DOMAIN_ONLY_PATTERN = f"@{DOMAIN_PLACEHOLDER}"

# Fields whose column may follow a convention worth keeping. Read after
# detection, since an email's convention is only legible next to the row's name.
_NAME_PATTERN_LABELS = ("full_name", "first_name", "last_name", "middle_name", "email")


def name_parts(values: Mapping[str, object]) -> dict[str, str]:
    """Normalize whatever name fields are present into first/middle/last parts.

    Prefer structured fields when they exist; otherwise split a full name.
    """
    parts = {
        label: str(value).strip()
        for label in ("first_name", "middle_name", "last_name")
        if (value := values.get(label)) is not None and str(value).strip()
    }
    full = values.get("full_name")
    if not parts and full is not None and str(full).strip():
        return split_full_name(str(full))
    return parts


def _row_name_parts(row: pd.Series, fields: Mapping[str, str]) -> dict[str, str]:
    """Return ``name_parts`` for one dataframe row using the cluster's column map."""
    return name_parts(
        {
            label: row[col]
            for label in ("first_name", "middle_name", "last_name", "full_name")
            if (col := fields.get(label)) is not None and pd.notna(row.get(col))
        }
    )


def _handle_shape_counts(handles: Sequence[str], cfg: Config) -> Counter:
    """Count character-shape patterns for email locals that are not a person's name."""
    counts: Counter = Counter()
    shapes = ranked_value_patterns(pd.Series(handles, dtype=object), cfg) if handles else []
    for handle in handles:
        shape = next((shape for shape in shapes if value_matches_template(handle, shape)), None)
        counts[f"{shape}@{DOMAIN_PLACEHOLDER}" if shape else EMAIL_DOMAIN_ONLY_PATTERN] += 1
    return counts


def name_column_pattern(
    df: pd.DataFrame, label: str, col: str, fields: Mapping[str, str], cfg: Config
) -> str | None:
    """Dominant name/email convention for one column, or ``None`` if under coverage."""
    rows = df.dropna(subset=[col]).head(PATTERN_SAMPLE_SIZE)
    counts: Counter = Counter()
    handles: list[str] = []
    for _, row in rows.iterrows():
        value = str(row[col])
        if label == "email":
            local, domain = split_email(value)
            if not domain:
                continue
            parts = _row_name_parts(row, fields)
            if own := (infer_email_pattern(value, parts) if parts else None):
                counts[own] += 1
            else:
                handles.append(local)
            continue
        _, rest = split_title(value)
        parts = split_full_name(rest) if label == "full_name" else {label: rest}
        if own := infer_name_pattern(rest, parts):
            counts[own] += 1
    counts.update(_handle_shape_counts(handles, cfg))
    return dominant_format(counts, len(rows), min_coverage=cfg.dominant_pattern_min_coverage)


def _columns_by_label(fields: Mapping[str, DetectedField]) -> dict[str, str]:
    """Map same-person field labels to dataframe column names."""
    return {label: entry.column for label, entry in fields.items()}


def attach_name_patterns(df: pd.DataFrame, bundles: list[SamePersonBundle], cfg: Config) -> None:
    """Write discovered name/email conventions onto each field's ``pattern``."""
    for bundle in bundles:
        fields = bundle.fields
        columns_by_label = _columns_by_label(fields)
        for label in _NAME_PATTERN_LABELS:
            col = columns_by_label.get(label)
            if col is None or col not in df.columns:
                continue
            pattern = name_column_pattern(df, label, col, columns_by_label, cfg)
            if pattern:
                fields[label] = DetectedField(column=col, pattern=pattern)
                logger.runtime.info(
                    f"[PII Replacement] Same-person column {col!r} (entity={label}, pattern={pattern!r})"
                )


# ===========================================================================
# Title / name helpers
# ===========================================================================
_TITLE_RE = re.compile(r"^(dr|mr|mrs|ms|miss|prof|sir|madam|mx)\.?\s+", re.IGNORECASE)


def split_title(value: str):
    """Peel a leading honorific off a name, if present.

    Returns:
        A ``(title, remainder)`` pair; ``title`` is ``None`` when absent.

    Example:
        ``"Dr. Jane Smith"`` -> ``("Dr.", "Jane Smith")``;
        ``"Jane Smith"`` -> ``(None, "Jane Smith")``.
    """
    m = _TITLE_RE.match(value.strip())
    if m:
        return value[: m.end()].strip(), value[m.end() :].strip()
    return None, value.strip()


# --- Name-part templates (names and emails) ----------------------------------
# A name or an email is assembled from name parts rather than masked character by
# character, so its pattern names those parts instead of its characters:
# '{LAST}, {First}' or '{f}.{last}@{domain}'. A placeholder is written the way its
# output is written -- '{first}' is jane, '{First}' is Jane, '{FIRST}' is JANE --
# and a one-letter placeholder is an initial. In an email the parts lose their
# spaces and apostrophes, the way a real address does, and '{domain}' keeps the
# domain the value already had.
_NAME_PARTS = {"first": "first_name", "middle": "middle_name", "last": "last_name"}
_NAME_INITIALS = {"f": "first_name", "m": "middle_name", "l": "last_name"}
# A digit beside a name, as in 'j.smith2@acme.com', tells two people apart rather
# than saying anything about the column, so a name pattern holds the position
# and the replacement draws its own digit there. Spelled as the value templates
# spell it (see _FAMILY_TOKENS).
DIGIT_PLACEHOLDER = "#"
_ALNUM_RUN_RE = re.compile(r"[A-Za-z0-9]+")
# An address whose local part reads as no person is a handle, and a handle names
# nobody: what it has is a shape, so the shape is what the pattern holds. A column
# of them names those shapes the way an identifier column does
# ('usr47[0123]#@{domain}'), and one whose handles line up in no shape at all
# names '@{domain}', which keeps the domain and leaves each handle its own shape.
NAME_PLACEHOLDERS = tuple(
    f"{{{token}}}" for key in _NAME_PARTS for token in (key, key.capitalize(), key.upper())
) + tuple(f"{{{token}}}" for key in _NAME_INITIALS for token in (key, key.upper()))
# A placeholder plus whatever punctuation follows it, so a part the person does
# not have ('{First} {M}. {Last}' for a person with no middle name) takes its own
# separator with it instead of leaving 'Robert . Jones' behind.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z]+)\}([^{}A-Za-z0-9]*)")


def placeholder_tokens(pattern: str) -> list[str]:
    """Return the ``{...}`` names a name-part pattern uses, in order.

    Example:
        ``"{LAST}, {First}"`` -> ``["LAST", "First"]``.
    """
    return [match.group(1) for match in _PLACEHOLDER_RE.finditer(pattern)]


def _cased(text: str, token: str) -> str:
    if token.islower():
        return text.lower()
    if token.isupper():
        return text.upper()
    return text  # preserve original casing (keeps 'de la Cruz')


def _email_token(text: str) -> str:
    """Return a name part as an address writes it.

    Example:
        ``"de la Cruz"`` -> ``"delacruz"``.
    """
    return "".join(c for c in text if c.isalnum())


def _drawn_digits(text: str, rng: random.Random | None) -> str:
    """Replace each ``#`` with a fresh digit.

    Example:
        ``"j.smith#"`` might become ``"j.smith7"`` so the synthetic address does
        not reuse the original's distinguishing number.
    """
    if DIGIT_PLACEHOLDER not in text:
        return text
    draw = (rng or random).randrange
    return "".join(str(draw(10)) if ch == DIGIT_PLACEHOLDER else ch for ch in text)


def render_name_pattern(
    pattern: str,
    parts: Mapping[str, str],
    *,
    for_email: bool = False,
    rng: random.Random | None = None,
) -> str | None:
    """Fill ``pattern`` from normalized name parts.

    Args:
        pattern: Name-part template string.
        parts: Normalized name parts.
        for_email: When ``True``, format parts for email local use.
        rng: Optional random source for ``#`` digit placeholders.

    Returns:
        Rendered string, or ``None`` when ``parts`` has nothing the pattern needs.

    Example:
        ``"{LAST}, {First}"`` with Jane/Smith -> ``"SMITH, Jane"``;
        ``"{f}.{last}#"`` with ``for_email=True`` -> e.g. ``"j.smith4"``.
    """
    filled = dropped = False

    def _replace(match: re.Match[str]) -> str:
        nonlocal filled, dropped
        token, separator = match.group(1), match.group(2)
        key = token.lower()
        field = _NAME_PARTS.get(key) or _NAME_INITIALS.get(key)
        if field is None:
            return match.group(0)
        text = str(parts.get(field) or "")
        if for_email:
            text = _email_token(text)
        if key in _NAME_INITIALS:
            text = text[:1]
        if not text:
            dropped = True
            return ""
        filled = True
        return _cased(text, token) + separator

    rendered = _drawn_digits(_PLACEHOLDER_RE.sub(_replace, pattern).strip(), rng)
    if not filled:
        return None
    # A dropped part can leave the separator that led to it ('Jones, '), but an
    # untouched pattern ends the way it was written ('Jones, R.').
    return rendered.rstrip(" ,.-_") if dropped else rendered


def _part_matchers(parts: Mapping[str, str]) -> list[tuple[re.Pattern[str], str, bool]]:
    """Return every name part as a matcher over a value, paired with its placeholder.

    A part is matched by its runs of letters and digits with any punctuation
    between them, because a value spells a part however it likes: the surname
    ``Galenor-Quill`` is written ``galenor-quill`` in one address and
    ``galenorquill`` in the next, and a first name the column records as
    ``Pella Y.`` keeps that space and period. Matching the punctuation literally
    would leave the part unrecognized, and the scan would then read its first
    letter as an initial and copy the rest of the original into the pattern.

    A part is matched as the column spells it before it is matched run by run, so
    the period of a first name recorded as ``Pella Y.`` belongs to the part in its
    own column, where the value is nothing else, and is a separator in
    ``pella-y.galenor-quill``, where it stands between two parts.

    Longest part first, so a name is read as one rather than as an initial
    followed by letters, and a part of a single letter is only ever an initial
    (``Smith, J.``).

    Args:
        parts: Normalized name parts.

    Returns:
        Matchers sorted longest-first, each as ``(regex, token, is_initial)``.
    """
    ranked: list[tuple[int, re.Pattern[str], str, bool]] = []
    for key, part in _NAME_PARTS.items():
        text = str(parts.get(part) or "")
        runs = _ALNUM_RUN_RE.findall(text)
        if not runs:
            continue
        squashed = "".join(runs)
        if len(squashed) > 1:
            if text != squashed:
                ranked.append((len(text), re.compile(re.escape(text), re.IGNORECASE), key, False))
            spelled = r"[^A-Za-z0-9]*".join(re.escape(run) for run in runs)
            ranked.append((len(squashed), re.compile(spelled, re.IGNORECASE), key, False))
        ranked.append((1, re.compile(re.escape(squashed[0]), re.IGNORECASE), key[0], True))
    ranked.sort(key=lambda item: -item[0])
    return [(matcher, token, is_initial) for _, matcher, token, is_initial in ranked]


def _placeholder_for(token: str, surface: str) -> str:
    """Return the placeholder that writes ``surface``.

    ``{LAST}`` shouts where ``{last}`` does not. A part written in neither case
    throughout (``Galenor-Quill``) is ``{Last}``, which preserves mixed spelling.

    Args:
        token: Base placeholder token (e.g. ``last``).
        surface: Text as it appeared in the source value.

    Returns:
        Cased placeholder name without braces.
    """
    if surface.isupper():
        return token.upper()
    if surface.islower():
        return token
    return token.capitalize()


def infer_name_pattern(value: str, parts: Mapping[str, str]) -> str | None:
    """Read how one value lays out name parts as a reusable pattern.

    Digits become ``#`` so replacements do not reuse the original's number.

    Args:
        value: One name or email-local string.
        parts: Normalized name parts for the same person (row).

    Returns:
        Inferred name-part pattern, or ``None`` when no name parts appear.

    Example:
        ``"SMITH, Jane"`` + Jane/Smith -> ``"{LAST}, {First}"``;
        ``"j.smith2"`` + Jane/Smith -> ``"{f}.{last}#"``;
        ``"usr4701"`` + Jane/Smith -> ``None``.
    """
    matchers = _part_matchers(parts)
    out: list[str] = []
    index, found, after_part = 0, False, False
    while index < len(value):
        for matcher, token, is_initial in matchers:
            match = matcher.match(value, index)
            if match is None:
                continue
            # A single letter is only an initial where a word starts, so the 'a' in
            # 'jane.adams' is not read as Alice's initial.
            if is_initial and index and value[index - 1].isalnum() and not after_part:
                continue
            out.append("{" + _placeholder_for(token, match.group(0)) + "}")
            index = match.end()
            found, after_part = True, True
            break
        else:
            out.append(DIGIT_PLACEHOLDER if value[index].isdigit() else value[index])
            index += 1
            after_part = False
    return "".join(out) if found else None


def split_email(value: str) -> tuple[str, str]:
    """Split an address into local part and domain.

    Returns:
        ``(local, domain)``; domain is empty when no ``@`` is present.

    Example:
        ``"jane.smith@acme.com"`` -> ``("jane.smith", "acme.com")``;
        ``"not-an-email"`` -> ``("not-an-email", "")``.
    """
    local, at, domain = str(value).rpartition("@")
    return (local, domain) if at else (str(value), "")


def handle_email_pattern(original: str, patterns: Sequence[str]) -> str:
    """Pick a shape pattern for an email whose local part is not a person's name.

    Args:
        original: Original email address.
        patterns: Column name/email patterns that may include handle shapes.

    Returns:
        Shape pattern with ``@{domain}`` suffix; falls back to this handle's own
        shape (or ``@{domain}``) when no listed shape fits.

    Example:
        Given column patterns ``["usr47[0123]#@{domain}"]`` and original
        ``"usr4701@acme.com"``, returns ``"usr47[0123]#@{domain}"``.
    """
    local = split_email(original)[0]
    shapes = [shape for pattern in patterns if (shape := split_email(pattern)[0]) and not placeholder_tokens(shape)]
    return f"{matching_template(local, shapes)}@{DOMAIN_PLACEHOLDER}"


def render_email_pattern(
    pattern: str, parts: Mapping[str, str], original: str, rng: random.Random | None = None
) -> str | None:
    """Write a synthetic email from ``pattern``, keeping the original's domain.

    Args:
        pattern: Persona email pattern.
        parts: Normalized name parts.
        original: Source email address (domain donor).
        rng: Optional random source for handle shapes and ``#`` placeholders.

    Returns:
        Synthetic email address, or ``None`` when rendering fails.

    Example:
        ``"{first}.{last}@{domain}"`` with Jane/Smith and ``"...@acme.com"`` ->
        ``"jane.smith@acme.com"``. A handle shape like ``"usr47[0123]#@{domain}"``
        generates a new local of that shape. ``"@{domain}"`` keeps the original local
        shape via ``handle_email_pattern``.
    """
    local_pattern, domain_pattern = split_email(pattern)
    if not local_pattern:
        local_pattern = split_email(handle_email_pattern(original, ()))[0]
    if placeholder_tokens(local_pattern):
        local = render_name_pattern(local_pattern, parts, for_email=True, rng=rng)
    else:
        local = generate_from_pattern(local_pattern, rng or random)
    if not local:
        return None
    domain = domain_pattern.replace(DOMAIN_PLACEHOLDER, split_email(original)[1])
    return f"{local}@{domain}" if domain else None


def infer_email_pattern(value: str, parts: Mapping[str, str]) -> str | None:
    """Like ``infer_name_pattern`` for an address, with the domain as ``{domain}``.

    Args:
        value: One email address.
        parts: Normalized name parts for the same person (row).

    Returns:
        Inferred email name-part pattern, or ``None`` when inference fails.

    Example:
        ``"jane.smith@acme.com"`` + Jane/Smith -> ``"{first}.{last}@{domain}"``.
    """
    local, domain = split_email(value)
    if not domain:
        return None
    local_pattern = infer_name_pattern(local, parts)
    return f"{local_pattern}@{DOMAIN_PLACEHOLDER}" if local_pattern else None


def split_full_name(value: str) -> dict[str, str]:
    """Split a full name into parts using the order the string itself uses.

    Titles are stripped first; trailing periods on initials are not kept.

    Args:
        value: Full name string.

    Returns:
        Name parts keyed by ``first_name``, ``middle_name``, and ``last_name``.

    Example:
        ``"Smith, Jane A."`` ->
        ``{"first_name": "Jane", "middle_name": "A", "last_name": "Smith"}``
        (comma means last-first). ``"Jane A. Smith"`` -> the same parts in
        first-last order.
    """
    _, rest = split_title(str(value))
    if "," in rest:
        last, _, remainder = rest.partition(",")
        tokens = remainder.split()
        first = tokens[0] if tokens else ""
        middle = " ".join(tokens[1:])
    else:
        tokens = rest.split()
        if not tokens:
            return {}
        first, last = tokens[0], (tokens[-1] if len(tokens) > 1 else "")
        middle = " ".join(tokens[1:-1])
    # 'J.' is the initial J, so the punctuation around a part is not part of it.
    parts = {"first_name": first, "last_name": last, "middle_name": middle}
    return {key: stripped for key, text in parts.items() if (stripped := text.strip(" .,;:"))}
