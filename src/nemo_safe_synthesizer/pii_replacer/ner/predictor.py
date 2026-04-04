# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from re import Pattern
from typing import Optional

from ...data_processing.records.base import KVPair
from ...data_processing.records.json_record import JSONRecord
from .ner import NERPrediction

DEFAULT_CONTEXT_SPAN_SIZE = 16


def _get_neighbor_strings(data: str, start: int, end: int, span: int) -> tuple[str, str]:
    left_mark = start - span
    if left_mark < 0:  # if we use a negative number we'll data from the end of the string
        left_mark = 0

    right_mark = min(len(data), (end + span))

    # here we want to walk left, and walk right to find the first whitepace and
    # adjust our markers. this is so we don't return partial words in the surrounding
    # text
    if left_mark > 0 and data[left_mark].isalnum():
        pos = left_mark
        while True:
            pos -= 1
            if not pos:
                left_mark = 0
                break
            if not data[pos].isalnum():
                left_mark = pos + 1
                break

    if right_mark < len(data) and data[right_mark].isalnum():
        pos = right_mark
        while True:
            pos += 1
            if pos >= len(data):
                right_mark = len(data)
                break
            if not data[pos].isalnum():
                right_mark = pos
                break

    return data[left_mark:start].casefold(), data[end:right_mark].casefold()


@dataclass
class ContextSpan:
    """This class can be used to search for surrounding context given an
    input string and some start/end offsets within that string.  You create
    this object by providing a list of discrete strings or regex patterns
    to match on, and then how far "left" and "right" of the target string
    to search for these patterns.

    In the below example we'll search for context left and right of
    a phone number::

        tgt = "Please give me a call at 867-5309"

    We can create a ContextSpan to use the "call" string as context::

        c = ContextSpan(pattern_list=["call"])
        assert c.is_match(tgt, 25, 33)

    Args:
        pattern_list: A list of strings or regex Patterns to use for matching
        span: How many characters left of the start index and right of the end index
            to search for any matches from the ``pattern_list`` objects.
    """

    pattern_list: list[str | Pattern]
    span: int = DEFAULT_CONTEXT_SPAN_SIZE

    def is_match(self, data: str, start: int, end: int) -> bool:
        left, right = _get_neighbor_strings(data, start, end, self.span)
        for pattern in self.pattern_list:
            if isinstance(pattern, str):
                if pattern in left:
                    return True
                elif pattern in right:
                    return True
                else:
                    continue
            elif isinstance(pattern, Pattern):
                if pattern.search(left):
                    return True
                elif pattern.search(right):
                    return True
                else:
                    continue
            else:
                continue

        return False


def is_context_matched(data: str, start: int, end: int, spans: list[ContextSpan]) -> bool:
    for span in spans:
        if span.is_match(data, start, end):
            return True
    return False


@dataclass
class PredictorContext(ABC):
    """Base class for an arbitrary context object that can be
    passed into a predictor. Arbitrary contexts can be subclassed
    from here and passed into the ``Predictor`` objects.

    This can be useful when predictors should have the same business
    logic but perhaps some differing settings like contexts, etc
    """


class Predictor(ABC):
    """
    Base class for managing an entity prediction.

    Predictors operate at the record level and might
    be managed via a `PredictionPipeline` parent class.
    For a NLP pipeline this might represent a model. In pattern
    based pipelines a `Predictor` might represent a single
    entity matcher such as an IP address.
    """

    # label source options
    KEY = 1
    VALUE = 2
    BOTH = 3

    default_namespace: str = "safe-synthesizer"
    default_name: str = None
    """Subclasses can set a default name to use that
    can be directly accessed as a class attr if need
    be.
    """

    def __init__(
        self,
        name: str,
        namespace: str = None,
        predictor_context: Optional[PredictorContext] = None,
    ):
        if namespace is None:
            namespace = self.default_namespace

        if not name:
            raise ValueError("name required")

        self.source = f"{namespace.lower()}/{name.lower()}"
        self._context = predictor_context

    @abstractmethod
    def evaluate(self, in_data: JSONRecord) -> list[NERPrediction]:
        """This MUST be implemented by each Predictor"""
        pass

    def header_has_context(
        self,
        field_pair: KVPair,
        header_context_source: int,
        token_patterns: Pattern = None,
        regex_patterns: Pattern = None,
    ) -> bool:
        """Checks to see if the field has a label match."""
        _field = field_pair
        if header_context_source == self.BOTH:
            search_string = (_field.field + " " + _field.value if _field.field else _field.value).casefold()
        elif header_context_source == self.VALUE:
            search_string = _field.value.casefold()
        else:
            if _field.field is None:
                return False
            search_string = _field.field.casefold()

        if regex_patterns is not None and regex_patterns.search(search_string):
            return True

        if token_patterns is not None:
            for token in field_pair.field_tokens:
                if token_patterns.match(token):
                    return True

        return False
