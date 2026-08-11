# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Character templates for identifiers, phones and cards: infer, match, generate, conform."""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from functools import lru_cache

import pandas as pd

from ...observability import get_logger
from ..entities import (
    Config,
    is_identify_only,
    is_missing_value,
    spec,
    sval,
)
from .evidence import pattern_evidence_values, ranked_formats

logger = get_logger(__name__)


def value_patterns(values: pd.Series, cfg: Config) -> list[str]:
    """Return character templates a column writes its values in, most common first.

    Values are first grouped by fine character class (digit/letter case). When that
    fragments samples too much to infer a template — typical of embedded UUIDs /
    hex runs — fall back to coarse structure (alphanumeric vs separators). Within
    each group, ``infer_value_pattern`` still pins literals from the real characters.

    Args:
        values: Column values to inspect.
        cfg: Pattern inference configuration.

    Returns:
        Ranked Faker-style template strings.

    Example:
        ``["pmc-6123", "pmc-8123", "pmc-6124"]`` -> ``["pmc-[68]###"]``.
    """
    sample = pattern_evidence_values(values)
    patterns = _value_patterns_for_shape_fn(sample, cfg, value_shape_template)
    if patterns:
        return patterns
    return _value_patterns_for_shape_fn(sample, cfg, value_structure_template)


def _value_patterns_for_shape_fn(sample: list[str], cfg: Config, shape_fn) -> list[str]:
    by_shape: dict[str, list[str]] = {}
    counts: Counter = Counter()
    for value in sample:
        shape = shape_fn(value)
        counts[shape] += 1
        group = by_shape.setdefault(shape, [])
        if len(group) < cfg.pattern_sample_cap:
            group.append(value)

    patterns: list[str] = []
    for shape in ranked_formats(counts, sum(counts.values())):
        pattern = infer_value_pattern(by_shape[shape], cfg)
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    return patterns


def attach_value_patterns(df: pd.DataFrame, standalone: list[dict], cfg: Config) -> None:
    """Attach character templates each standalone column writes, in place.

    A phone column of ``+1-415-555-####`` values gets
    ``ent["patterns"] = ["+1-###-###-####"]``. Temporal / identify-only entities
    are left alone (they use strftime, not character templates).

    Args:
        df: Source dataframe.
        standalone: Standalone entity dicts to mutate with ``patterns`` keys.
        cfg: Pattern inference configuration.
    """
    for ent in standalone:
        # Temporal columns and birth dates are written in strftime formats, which
        # detection already read, not character-template regeneration.
        entity_label = ent.get("entity")
        if entity_label == "date_of_birth" or (isinstance(entity_label, str) and is_identify_only(entity_label)):
            continue
        # Generators that ignore character templates (IPs, SSN, national ID, …).
        entity_spec = spec(entity_label) if isinstance(entity_label, str) else None
        if entity_spec is not None and entity_spec.pattern_kind != "template":
            ent["patterns"] = []
            continue
        col = ent.get("column")
        if col not in df.columns:
            continue
        ent["patterns"] = value_patterns(df[col].dropna(), cfg)
        logger.runtime.info(f"[PII Replacement] Standalone column {col!r} writes patterns={ent['patterns']}")


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


def _family_token(keep: set[str]) -> str:
    for tok, charset in _FAMILY_TOKENS:
        if keep <= set(charset):
            return tok
    return "*"


def _position_token(chars: list[str], cfg: Config) -> str | None:
    """Return the template token for one column position given observed characters.

    ``None`` means the position is not templatable, which condemns the whole
    column: punctuation sharing a position with anything else says the values are
    not aligned (``9.7.8.128`` under ``217.197.215.20``), so any template built
    over them would scatter separators through the output.

    How narrow a token may be depends on how many values back it. Freezing a
    literal, or narrowing to a class, claims the column can hold nothing else --
    a claim three sample values cannot support -- so thin positions widen to
    their family token and keep only the shape.

    Args:
        chars: Characters observed at this index across sample values.
        cfg: Pattern inference configuration.

    Returns:
        A template token, or ``None`` when positions are misaligned.

    Example:
        ``["6","8","6"]`` with enough evidence -> ``"[68]"``; ``[".","."]`` ->
        ``"."``; ``[".","7"]`` -> ``None`` (misaligned).
    """
    counts = Counter(chars)
    total = sum(counts.values())
    keep = {c for c, n in counts.items() if total and n / total >= cfg.pattern_rare_char_frac}
    if not keep:
        keep = {counts.most_common(1)[0][0]}

    if any(not c.isalnum() for c in keep):
        # Separators are structural rather than sampled, so one is kept literal
        # however few values were seen.
        return _literal_token(next(iter(keep))) if len(keep) == 1 else None

    evidence = cfg.pattern_min_evidence_per_char
    if len(keep) == 1:
        return _literal_token(next(iter(keep))) if total >= evidence else _family_token(keep)
    if len(keep) <= cfg.pattern_class_max and total >= evidence * len(keep):
        body = "".join(sorted(keep)).replace("\\", "\\\\").replace("]", "\\]")
        return "[" + body + "]"
    return _family_token(keep)


_VARIABLE_TEMPLATE_POSITION = re.compile(r"(?<!\\)[#^@&%*\[]")


def value_template_is_constant(pattern: str) -> bool:
    """Return whether a value template is all literals.

    Example:
        ``"pmc-"`` -> ``True``; ``"pmc-#"`` -> ``False``.
    """
    return _VARIABLE_TEMPLATE_POSITION.search(pattern) is None


def infer_value_pattern(values, cfg: Config) -> str | None:
    """Infer a Faker-style template from sample values (modal length).

    Constant positions stay literal; low-entropy positions become an explicit
    class like ``"[68]"``; high-entropy positions become a family token
    (``#`` digit, ``@`` lowercase, …).

    Args:
        values: Sample cell values from one shape group.
        cfg: Pattern inference configuration.

    Returns:
        An inferred template string, or ``None`` when values are too irregular
        (mixed lengths) to template safely.

    Example:
        ``["pmc-612345", "pmc-812901", "pmc-612888"]`` -> ``"pmc-[68]######"``;
        ``["abc", "abcd", "ab"]`` -> ``None``.
    """
    vals = [str(v) for v in values if not is_missing_value(v)]
    if not vals:
        return None
    seen = list(dict.fromkeys(vals))[: cfg.pattern_sample_cap]
    modal_len = Counter(len(v) for v in seen).most_common(1)[0][0]
    if modal_len == 0:
        return None
    same = [v for v in seen if len(v) == modal_len]
    if len(same) < max(3, int(0.5 * len(seen))):
        return None  # too many distinct lengths -> not a fixed template
    tokens = [_position_token([v[i] for v in same], cfg) for i in range(modal_len)]
    if any(token is None for token in tokens):
        return None  # positions do not line up -> not a fixed template
    pattern = "".join(token for token in tokens if token is not None)
    # Reject a degenerate all-literal pattern (nothing would vary).
    if value_template_is_constant(pattern):
        return None
    return pattern


_TEMPLATE_FAMILY_CLASSES = {token: charset for token, charset in _FAMILY_TOKENS}


@lru_cache(maxsize=256)
def _template_positions(pattern: str) -> tuple[tuple[str, bool], ...]:
    """Parse a template into one entry per character it prints.

    Each entry is ``(characters, is_literal)``. A literal entry holds the single
    character to print; a variable entry holds the alphabet to draw from.
    Generating, matching, and re-formatting all walk the same parse, so a template
    means one thing across the three.

    Args:
        pattern: Faker-style value template.

    Returns:
        Parsed position tuples for the template.
    """
    out: list[tuple[str, bool]] = []
    i, n = 0, len(pattern)
    while i < n:
        ch = pattern[i]
        if ch == "\\" and i + 1 < n:
            out.append((pattern[i + 1], True))
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
            if body:
                out.append(("".join(body), False))
            i = j + 1
            continue
        if ch in _TEMPLATE_FAMILY_CLASSES:
            out.append((_TEMPLATE_FAMILY_CLASSES[ch], False))
            i += 1
            continue
        out.append((ch, True))
        i += 1
    return tuple(out)


def generate_from_pattern(pattern: str, rng) -> str:
    """Generate one string from a template produced by ``infer_value_pattern``.

    Args:
        pattern: Faker-style value template.
        rng: Random source with a ``choice`` method.

    Returns:
        One randomly drawn string matching the template.

    Example:
        ``"pmc-[68]####"`` -> e.g. ``"pmc-7123"`` (``#`` draws a digit, ``[68]``
        draws 6 or 8, literals are kept).
    """
    return "".join(chars if literal else rng.choice(chars) for chars, literal in _template_positions(pattern))


@lru_cache(maxsize=256)
def _template_regex(pattern: str) -> re.Pattern[str]:
    """Build a regex accepting exactly the strings ``generate_from_pattern`` can emit.

    Args:
        pattern: Faker-style value template.

    Returns:
        Compiled regex anchored at end of string.
    """
    body = "".join(
        re.escape(chars) if literal else "[" + re.escape(chars) + "]" for chars, literal in _template_positions(pattern)
    )
    return re.compile(body + r"\Z")


def value_matches_template(value: object, pattern: str) -> bool:
    """Return whether one value has the exact shape a value template describes.

    Args:
        value: Cell value to test.
        pattern: Faker-style value template.

    Returns:
        ``True`` when ``value`` matches the template exactly.

    Example:
        ``value_matches_template("pmc-7123", "pmc-[68]####")`` -> ``True``.
    """
    s = sval(value)
    if s is None:
        return False
    return _template_regex(pattern).match(s.strip()) is not None


def matching_template(value: str, patterns: Sequence[str]) -> str:
    """Return the first column template that describes ``value``, else the value's own shape.

    A column names the formats it writes; a value matching none is replaced in its
    own shape rather than rewritten into another format.

    Args:
        value: Original cell text.
        patterns: Column templates, most common first.

    Returns:
        A template string suitable for ``generate_from_pattern``.

    Example:
        ``matching_template("pmc-7123", ["pmc-[68]####"])`` -> ``"pmc-[68]####"``;
        ``matching_template("xyz-99", ["pmc-[68]####"])`` -> ``"@@@-##"``.
    """
    return next((p for p in patterns if value_matches_template(value, p)), value_shape_template(value))


def value_shape_template(value: str) -> str:
    """Return the template one value's own format describes, separators and all.

    Example:
        ``"pmc-6123"`` -> ``"@@@-####"`` (``@`` lowercase, ``#`` digit).
    """
    out: list[str] = []
    for ch in value:
        if ch.isdigit():
            out.append("#")
        elif ch.isupper():
            out.append("^")
        elif ch.islower():
            out.append("@")
        else:
            out.append(_literal_token(ch))
    return "".join(out)


def value_structure_template(value: str) -> str:
    """Return coarse shape: alphanumeric vs separators.

    Used to group values for pattern inference so embedded UUIDs / hex runs that
    differ only in digit-vs-letter positions still share one structural bucket
    (e.g. ``transcription-job-<ts>-call-<uuid>-<epoch>``).

    Args:
        value: One cell string.

    Returns:
        Structural template using ``A`` for alphanumeric runs.

    Example:
        ``"job-abc-123"`` -> ``"A-A-A"``.
    """
    return "".join("A" if ch.isalnum() else _literal_token(ch) for ch in value)


def conform_to_template(value: str, pattern: str, rng) -> str:
    """Re-print ``value``'s characters in the format ``pattern`` describes.

    For values that can only be generated whole: the PGM's phone number tracks its
    persona's address through its area code, so it is produced first and dressed in
    the column's format afterwards.

    The value is laid in from the right, so a number carrying a country code the
    column does not print loses the country code rather than its last digits. A
    literal the template pins (``'+1-###-555-####'`` pins a country code and an
    exchange) stands in for the value's character at that position, which keeps the
    rest aligned; positions the value cannot fill are drawn.

    Args:
        value: Raw alphanumeric content to format.
        pattern: Column value template.
        rng: Random source with a ``choice`` method.

    Returns:
        ``value`` formatted to match ``pattern``.

    Example:
        ``conform_to_template("12065550181", "+1-###-###-####", rng)`` ->
        ``"+1-206-555-0181"``.
    """
    positions = _template_positions(pattern)
    capacity = sum(1 for text, literal in positions if not literal or text.isalnum())
    chars: list[str | None] = [c for c in str(value) if c.isalnum()]
    chars = chars[len(chars) - capacity :] if len(chars) > capacity else [None] * (capacity - len(chars)) + chars
    take = iter(chars)

    out: list[str] = []
    for text, literal in positions:
        if literal:
            if text.isalnum():
                next(take, None)
            out.append(text)
            continue
        char = next(take, None)
        out.append(char if char is not None and char in text else rng.choice(text))
    return "".join(out)


def luhn_valid(digits: str) -> bool:
    """Return whether a digit string satisfies the Luhn checksum.

    Args:
        digits: Digit string to validate.

    Returns:
        ``True`` when ``digits`` passes the Luhn check and is non-empty.

    Example:
        ``luhn_valid("79927398713")`` -> ``True``.
    """
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return bool(digits) and total % 10 == 0


def synth_card_value(pattern: str, rng) -> str:
    """Return a card number in the column's format whose Luhn checksum adds up.

    Filling the template alone gives a well-formed but invalid number, which any
    card validator downstream rejects. The template supplies the format and the
    issuer prefix, both read from the column itself, and one drawn digit is then
    re-drawn until the checksum agrees. The last digit position is tried first,
    since that is the check digit on a real card.

    Args:
        pattern: Card number value template.
        rng: Random source with a ``choice`` method.

    Returns:
        A card number string matching ``pattern`` that passes Luhn validation when
        possible; otherwise the best-effort generated value.

    Example:
        ``"4###############"`` -> a 16-digit string starting with 4 that passes Luhn.
    """
    value = generate_from_pattern(pattern, rng)
    positions = _template_positions(pattern)
    digits = [
        index for index, (chars, literal) in enumerate(positions) if not literal and set(chars) <= set(string.digits)
    ]
    for index in reversed(digits):
        for digit in positions[index][0]:
            candidate = value[:index] + digit + value[index + 1 :]
            if luhn_valid("".join(c for c in candidate if c.isdigit())):
                return candidate
    # A template with no free digit (all-literal digits, or classes too narrow to
    # reach the needed checksum) cannot carry one; the format still holds.
    return value
