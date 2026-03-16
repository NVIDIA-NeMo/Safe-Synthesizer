# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Date string parsing, formatting, and inference utilities.

Supports ISO8601 timezone offsets (via ``strftime_extra`` / ``strptime_extra``),
permutation-based format inference (``parse_date``, ``infer_from_series``),
and date randomization for PII replacement (``randomize``).
"""

import itertools
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from random import randint
from typing import Optional

import pandas as pd

DAYS_TO_MS = 8.64e7
MASK_CH = "#"

date_component_orders = [
    lambda y, m, d, hms, tz: f"{d} {m} {y}",
    lambda y, m, d, hms, tz: f"{m} {d} {y}",
    lambda y, m, d, hms, tz: f"{y} {m} {d}",
    lambda y, m, d, hms, tz: f"{y} {m} {d}",
    lambda y, m, d, hms, tz: f"{y} {m} {d} {hms}",
    lambda y, m, d, hms, tz: f"{y} {m} {d} {hms} {tz}",
]
"""This list contains date orderings by component."""


component_formats = {
    "y": {"%y", "%Y"},
    "m": {"%b", "%B", "%m"},
    "d": {"%a", "%A", "%d"},
    "hms": {"%X", "%X %f"},
    "tz": {"%z", "%Z", "%!z"},
}
"""For every date component, there may exist multiple formats. This dictionary maps
components to any number of format variations. This used in conjunction with
``date_component_orders`` let us build up permutations of valid date string formats.
"""


component_seperators = ["/", ".", "-", " ", ",", "T", "Z", "+"]
"""Characters from this list will be removed from a date string and used to build up
a string containing only date components that hopefully match from ``date_component_orders``.
"""


def strftime_extra(dt: datetime, fmt: str) -> str:
    """Formats a datetime object, supporting ISO8601 timezone offsets.

    strftime_extra(dt, fmt) behaves like dt.strftime(fmt), with the exception that it supports
    the special %!z format directive. %!z is formatted as an ISO8601 UTC offset. For naive datetimes,
    it always expands to the empty string (same as %z); otherwise, it expands to a colon-separated
    ``[+-]hh:mm`` offset or ``Z`` for UTC. For the sake of an easier implementation, %!z is only allowed
    at the end of the format string.

    Args:
        dt: the datetime object to format.
        fmt: the format string to use (which may include %!z).

    Returns:
        the formatted datetime.
    """
    tz_extra_pos = fmt.find("%!z")
    if tz_extra_pos == -1:
        # When %!z isn't used, behave like strftime.
        return dt.strftime(fmt)

    # Ensure that %!z is only used at the end of the string.
    if tz_extra_pos != len(fmt) - 3:
        raise ValueError("%!z format modifier may only occur at the end of the format string")

    # Format date without timezone offset first, and then append the ISO8601-formatted offset
    # separately.
    formatted_base = dt.strftime(fmt[:-3])
    formatted_tzoffset = dt.strftime("%z")
    if not formatted_tzoffset:
        # dt is naive (no timezone info)
        return formatted_base
    if formatted_tzoffset == "+0000":
        formatted_tzoffset = "Z"
    else:
        # Insert a colon for a non-empty %z format.
        formatted_tzoffset = formatted_tzoffset[:3] + ":" + formatted_tzoffset[3:]
    return formatted_base + formatted_tzoffset


tz_suffix_re = re.compile(r"(Z|[+-]\d\d:\d\d)$")


def _transform_tz_suffix(m):
    suffix = m.group(1)
    if suffix == "Z":
        return "+0000"
    return suffix[:3] + suffix[4:]


def strptime_extra(date_string: str, fmt: str) -> datetime:
    """Parses a string as a datetime object, supporting ISO8601 timezone offsets.

    See the documentation on ``strftime_extra`` regarding the semantics of the new ``%!z`` format
    specifier.

    Args:
        date_string: the datetime in string format.
        fmt: the format string to use for parsing (which may include %!z).

    Returns:
        the parsed datetime.
    """
    tz_extra_pos = fmt.find("%!z")
    if tz_extra_pos == -1:
        # If %!z isn't used, behave like strptime.
        return datetime.strptime(date_string, fmt)

    # Ensure that %!z (if at all used) only occurs at the end of the string.
    if tz_extra_pos != len(fmt) - 3:
        raise ValueError("%!z format modifier may only occur at the end of the format string")

    # Replace a ``Z`` or ``[+-]hh:mm`` suffix with [+-]hhmm. Otherwise, just drop the timezone offset.
    # Without this, a timezone offset without a colon (such as +0000) would be parsed successfully with
    # %!z, which we want to avoid.
    date_string, nsubs = tz_suffix_re.subn(_transform_tz_suffix, date_string)
    dt = datetime.strptime(date_string, fmt[:-3] + ("%z" if nsubs else ""))

    return dt


def date_component_permutations() -> list[tuple[str, str, str, str, str]]:
    """Return the Cartesian product of per-component format strings.

    Each tuple is indexed by (year, month, day, hms, tz) and can be
    passed into a formatter from ``date_component_orders``.
    """
    return list(itertools.product(*component_formats.values()))


def gen_date_str_fmt_permutations() -> set[str]:
    """Return the set of all unique date format permutations."""
    return {order(*str_fmt) for str_fmt in date_component_permutations() for order in date_component_orders}


date_str_fmt_permutations = gen_date_str_fmt_permutations()
"""A unique list of date string formats"""


@dataclass
class TokenizedStr:
    """Represents a date string that has been broken up into individual date
    components. This class is useful when trying to rebuild a new string with
    the same format.
    """

    original_str: str
    """The original source string"""

    masked_str: str
    """A masked version of the string. Masked strings only contain the mask characters
    and component seperators.
    """

    components: list[tuple[str, tuple[int, int]]]
    """A list of components and their string index mapped from the source string"""

    seperators: list[str]
    """A list of component seperators. Zipping this list with ``components`` yields
    the original string.
    """

    @property
    def component_str(self) -> str:
        """A string containing only the components of the date. This is
        used to matched a date with a date format.
        """
        return " ".join([s for s, _ in self.components])

    def assemble_str_from_components(self, new_components: list[str]) -> str:
        """Given a new set of components, rebuild the string with formatting preserved.

        Args:
            new_components: The new set of component to reassemble the string with.
        """
        components_seperators = [
            token
            for token in itertools.chain(*itertools.zip_longest(new_components, self.seperators))
            if token is not None or token
        ]
        return "".join(components_seperators)


@dataclass
class ParsedDate:
    """Wrapper for a parsed date and associated model_metadata"""

    component_order: str
    """Matched date string format order form ``date_component_orders``. This can be
    used to reconstruct the original date string format including seperators.
    """

    date: datetime
    """The parsed datetime object"""

    tokenized_date: TokenizedStr
    """A reference to the tokenized date string"""

    @property
    def fmt_str(self) -> str:
        """The date format string used to to build the original date. This can be used
        with function like ``strftime`` or ``strptime``.

        Returns:
            Date format string such as "%m/%d/%Y".
        """
        comps = self.component_order.split(" ")
        return self.tokenized_date.assemble_str_from_components(comps)

    def date_to_fmt_str(self, date: datetime) -> str:
        """Given a new date object, returns that date in the parsed date format"""
        comps = date.strftime(self.component_order).split(" ")
        return self.tokenized_date.assemble_str_from_components(comps)

    def shift(self, days: int | None = None, ms: int | None = None, delta: timedelta | None = None) -> str:
        """Given a date shift in days or milliseconds or a ``timedelta`` object,
        will return a new date using the same original string format.
        Shifting by milliseconds is useful if the date is a timestamp.
        """
        if not isinstance(delta, timedelta):
            if not days or ms:
                raise ValueError("must specify days or ms")
            delta = timedelta(milliseconds=ms) if ms else timedelta(days=days)
        new_date = self.date + delta
        return self.date_to_fmt_str(new_date)


def tokenize_date_str(input: str) -> TokenizedStr:
    """Given a raw input date, will return an instance of ``TokenizedStr``. Any
    business logic, or edge cases for tokenizing a string belong in this method.
    """
    if MASK_CH in input:
        raise ValueError(f"Input date cannot be parsed. Contains mask {MASK_CH}")

    masked = list(input)
    components = []
    contig_sep = []

    cur_start = 0
    last_sep_idx = None
    for idx in range(0, len(input)):
        is_sep = input[idx] in component_seperators
        if is_sep:
            if idx - 1 == last_sep_idx:
                contig_sep[-1] += input[idx]
            else:
                contig_sep += input[idx]
            last_sep_idx = idx

        if is_sep or idx == len(input) - 1:
            # increment the end character by one if we're at the end the input str
            stop_idx = idx + 1 if idx == len(input) - 1 and not is_sep else idx

            comp = "".join(masked[cur_start:stop_idx])
            components.append((comp, (cur_start, stop_idx)))

            masked[cur_start:stop_idx] = MASK_CH * (stop_idx - cur_start)

            if stop_idx == len(input):
                break

            cur_start = idx + 1

    # this block checks to see if the last component might be a timezone. if it is,
    # we want to merge what we originally thought was a separator, into the
    # timezone component.
    if len(components) == 5 and contig_sep[-1] in {"-", "+"}:
        sep = contig_sep.pop(-1)
        val, span = components[-1]
        components[-1] = (f"{sep}{val}", (span[0] - 1, span[1]))
    # this block checks to see if the last separator was a Z, which, in the absence of
    # a timezone component and at the end of the string, we treat as an ISO8601 abbreviated
    # timezone offset.
    elif len(components) == 4 and contig_sep[-1] == "Z" and last_sep_idx is not None:
        sep = contig_sep.pop(-1)
        components.append((sep, (last_sep_idx, last_sep_idx + 1)))

    return TokenizedStr(input, "".join(masked), components, contig_sep)


def maybe_match(date, format) -> Optional[datetime]:
    """Attempt to parse ``date`` with ``format``, returning None on failure."""
    try:
        return strptime_extra(date, format)
    except ValueError:
        return None


def parse_date(
    input_date: str,
    date_str_fmts: list[str] | set[str] = date_str_fmt_permutations,
) -> Optional[ParsedDate]:
    """Parse a date string and return the first matching ``ParsedDate``, or None."""
    return next(parse_date_multiple(input_date, date_str_fmts), None)


def parse_date_multiple(
    input_date: str,
    date_str_fmts: list[str] | set[str] = date_str_fmt_permutations,
) -> Iterator[ParsedDate]:
    """Yield all valid ``ParsedDate`` interpretations of ``input_date`` across known formats."""
    tokenized_date = tokenize_date_str(input_date)

    for str_fmt in date_str_fmts:
        date = maybe_match(tokenized_date.component_str, str_fmt)
        if date:
            yield ParsedDate(str_fmt, date, tokenized_date)


def randomize(date: str, days: int) -> Optional[str]:
    """Given a date string of some unknown format, returns a randomly shifted version
    of that date.

    Args:
        date: The date string to shift
        days: The max number of days to shift the date by. The range of valid days
            include [-days, days].
    """
    parsed_date = parse_date(date)

    if not parsed_date:
        return None

    if days == 0:
        return None

    # days_to_shift must be non-zero
    days_to_shift = randint(-days, days)
    while days_to_shift == 0:
        days_to_shift = randint(-days, days)

    return parsed_date.shift(days_to_shift)


def d_str_to_fmt_multiple(input_date: str) -> Iterator[str]:
    """Yield all plausible ``strftime`` format strings for a date string."""
    for parsed_date in parse_date_multiple(input_date):
        yield parsed_date.fmt_str


def maybe_d_str_to_fmt_multiple(input_date: str) -> Iterator[str]:
    """Like ``d_str_to_fmt_multiple`` but silently yields nothing on ``ValueError``."""
    try:
        yield from d_str_to_fmt_multiple(input_date)
    except ValueError:
        pass


def d_str_to_fmt(input_date: str) -> Optional[str]:
    """Infer the most likely ``strftime`` format string for a date string, or None."""
    return next(d_str_to_fmt_multiple(input_date), None)


def infer_from_series(date_series: Iterable[str]) -> Optional[str]:
    """Infer the best ``strftime`` format for a series of date strings.

    Evaluates each date against all known format permutations and returns
    the most frequently matched format. This is more reliable than
    single-string inference, which can confuse ambiguous components like
    ``%m`` and ``%d``.
    """
    fmt_occurrences = Counter()
    for date in date_series:
        for fmt in maybe_d_str_to_fmt_multiple(date):
            fmt_occurrences[fmt] += 1
    highest_occurrence = fmt_occurrences.most_common(1)
    if highest_occurrence:
        return highest_occurrence[0][0]


def fit_and_transform_dates(
    df: pd.DataFrame,
    inplace: bool = False,
) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    """Detect date columns, convert them to elapsed seconds, and record the transformation.

    For each object-typed column, samples values to infer a date format. If
    successful, converts the column to seconds elapsed since the column minimum
    and records the format and min date for later reversal.

    Args:
        df: Input DataFrame.
        inplace: If True, mutate ``df`` directly instead of copying.

    Returns:
        A tuple of (date_min_dict, result_df). ``date_min_dict`` maps column
        names to ``{"format": ..., "min": ...}`` dicts needed by
        ``transform_dates`` for reversal.
    """
    date_min_dict = {}
    object_cols = [col for col, col_type in df.dtypes.iteritems() if col_type == "object"]
    result_df = df.copy() if not inplace else df
    for object_col in object_cols:
        no_nans = result_df[object_col].dropna(axis=0).reset_index(drop=True)
        if not no_nans.empty:
            inferred_format = infer_from_series((no_nans.sample(100, replace=True)).astype(str))
            if inferred_format:
                try:
                    inferred_format = inferred_format.replace("!", "")
                    dates = pd.to_datetime(result_df.loc[:, object_col], format=inferred_format)
                    min_date = dates.min()
                    result_df[object_col] = (dates - min_date).dt.total_seconds()
                    date_min_dict[object_col] = {
                        "format": inferred_format,
                        "min": str(min_date),
                    }
                except (ValueError, TypeError):
                    pass
    return date_min_dict, result_df


def transform_dates(dates: dict[str, dict[str, str]], df: pd.DataFrame) -> pd.DataFrame:
    """Apply a previously fitted date-to-seconds transformation to a DataFrame.

    Args:
        dates: Mapping from column names to ``{"format": ..., "min": ...}``
            dicts as returned by ``fit_and_transform_dates``.
        df: DataFrame to transform.

    Returns:
        A copy of ``df`` with date columns converted to elapsed seconds.
    """
    result_df = df.copy()
    for col, details in dates.items():
        _dates = pd.to_datetime(result_df[col], format=details["format"], errors="coerce")
        result_df[col] = (_dates - pd.Timestamp(details["min"])).dt.total_seconds()
    return result_df
