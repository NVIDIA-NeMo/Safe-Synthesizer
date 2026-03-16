# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from re import Pattern as PatternType
from typing import AnyStr, Optional

# We import as RePattern here separately from PatternType
# so we can use it for isinstance checks. But if the first
# part of this import succeeds, we cannot use it for type
# hinting so we import that one separately
try:
    from re import Pattern as RePattern
except ImportError:
    from re import Pattern as RePattern

from ...data_processing.records.base import KVPair
from ...data_processing.records.json_record import JSONRecord
from .entity import Entity, Score
from .ner import NERError, NERPrediction
from .predictor import ContextSpan, Predictor, is_context_matched


def split_header_contexts(
    contexts: list[str | RePattern],
) -> tuple[RePattern | None, RePattern | None]:
    """Split a list of strings and RePatterns into two distcit regexes.

    Returns (regexes, tokens)
    """
    # split the header contexts into two distinct patterns
    _header_patterns = []
    _header_tokens = []
    for context in contexts:
        if isinstance(context, RePattern):
            _header_patterns.append(context)
        elif isinstance(context, str):
            _header_tokens.append(context)
        else:
            continue

    if _header_patterns:
        _header_patterns_regex = re.compile("|".join([pat.pattern for pat in _header_patterns]), re.IGNORECASE)
    else:
        _header_patterns_regex = None

    if _header_tokens:
        _header_tokens_regex = re.compile("|".join(["^" + tok + "$" for tok in _header_tokens]), re.IGNORECASE)
    else:
        _header_tokens_regex = None

    return _header_patterns_regex, _header_tokens_regex


@dataclass
class Pattern:
    """Represents a single regex pattern and its settings on how to be
    applied.

    Raises:
        `NERError` if `pattern` is not a string or regex Pattern
    """

    pattern: str | RePattern

    context_score: Optional[float] = Score.HIGH
    """This is the optimal score that you want to assign when context exists
    either in the header name or the surrounding text. We default this to high.
    """

    raw_score: Optional[float] = Score.LOW
    """This is the score that gets applied if there are no
    matching contexts. We default this to low.
    """

    ignore_raw_score: bool = False
    """If set, do not emit a match if only the raw regex matches without any context
    """

    header_contexts: Optional[list[str | RePattern]] = field(default_factory=list)
    """A list of strings or regexes that should be used to check the
    name of the field / header for a match. If there are any matches here, then
    the ``context_score`` value will be used as the matched score
    """

    header_regexes: Optional[RePattern] = field(init=False, default=None)
    header_tokens: Optional[RePattern] = field(init=False, default=None)

    neg_header_contexts: Optional[list[str | RePattern]] = field(default_factory=list)
    """A list of strings or regexes that can be used to disqualify a field from being analyzed.
    If used, any matches were will short-circuit processing for a given key/value pair."""

    neg_header_regexes: Optional[RePattern] = field(init=False, default=None)
    neg_header_tokens: Optional[RePattern] = field(init=False, default=None)

    header_context_source: int = Predictor.KEY
    """If doing header context searching, this dictates where to search for the context. We default
    to only searching within the field name itself. But we can also search the value or the concatenation
    of the field name and value
    """

    span_contexts: Optional[ContextSpan | list[ContextSpan]] = field(default_factory=list)
    """A list of ``ContextSpan`` instances that will be used, if provided, to
    search surrounding text of a string match for other discrete strings or
    matching regular expressions. See the ``ContextSpan`` usage for more details.
    """

    compiled_regex: PatternType[AnyStr] = field(init=False)

    def __post_init__(self):
        if isinstance(self.pattern, str):
            self.compiled_regex = re.compile(self.pattern)
        elif isinstance(self.pattern, RePattern):
            self.compiled_regex = self.pattern
        else:
            raise NERError(f"Could not initialize regex with {self.pattern}")

        if self.span_contexts and isinstance(self.span_contexts, ContextSpan):
            self.span_contexts = [self.span_contexts]

        self.header_regexes, self.header_tokens = split_header_contexts(self.header_contexts)
        self.neg_header_regexes, self.neg_header_tokens = split_header_contexts(self.neg_header_contexts)


class RegexPredictor(Predictor):
    """Base class that represents a single entity.

    Entities are matched based on a set of patterns
    with varying accuracy scores.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        patterns: list[Pattern] = None,
        entity: Optional[Entity] = None,
        namespace: Optional[str] = None,
    ):
        if patterns is None:
            patterns = []

        self.patterns = patterns
        self.entity = entity

        # NOTE: If a name is not provided, then we will use the
        # name of the module that holds the ``RegexPredictor``
        # subclass to create the name
        if name is None:
            name = self.__class__.__module__.split(".")[-1]

        super().__init__(name, namespace=namespace)

    def validate_match(self, matched_text: str, original_text: str):
        """
        A base method for regex rules to implement.

        The validate function is used to confirm
        an entity match. If the return value is not
        `None` the max score for that entity will
        be used.
        """
        return True

    def filter_by_range_by_score(self, field_matches: set[NERPrediction]) -> list[NERPrediction]:
        """Filter predictions by text range and take max score."""
        by_range = itertools.groupby(
            sorted(field_matches, key=lambda n: n.text),
            key=lambda p: (p.text, p.start, p.end),
        )

        return [max(ps, key=lambda p: p.score) for _, ps in by_range]

    def evaluate(self, in_record: JSONRecord, res_by_field=False) -> list[NERPrediction]:
        """
        Given a single record determine if any
        entities are represented.

        Args:
            in_record: the record to match patterns against

        Returns:
            A list of entity predictions sorted by score. Top score is
            first entry in list.
        """
        record_fields = in_record.kv_pairs
        result_set_by_field = [set() for _ in record_fields]

        record_field: KVPair
        for field_matches, record_field in zip(result_set_by_field, record_fields):
            # if the pattern requires a label, filter on field keys
            # that match the label.

            # WARNING: this updates the original record, which means that all the
            #  predictors running after this one, will see value as a string.

            # TODO(PROD-276): If we want these "value"s to always be strings, we should
            #  add that behavior one level up, so it's not handled by the predictor.
            #  And if we don't, we should make a copy here, so that original is not changed.
            record_field.value = str(record_field.value).strip()

            pattern: Pattern
            for pattern in self.patterns:
                # Check if there are any negative header contexts. If the header / field name
                # match on any of these, we do not need to process any further
                if pattern.neg_header_contexts:
                    if self.header_has_context(
                        record_field,
                        pattern.header_context_source,
                        token_patterns=pattern.neg_header_tokens,
                        regex_patterns=pattern.neg_header_regexes,
                    ):
                        continue

                # Check if there is any context in the header
                # by default, we assume there are no matches in the header
                header_label_match = False
                if pattern.header_contexts:
                    header_label_match = self.header_has_context(
                        record_field,
                        pattern.header_context_source,
                        token_patterns=pattern.header_tokens,
                        regex_patterns=pattern.header_regexes,
                    )

                # If there is no context requirements for the match and we do not
                # want to keep a raw match score, we just bail here since we don't need to bother
                # with running the regex
                if not header_label_match and not pattern.span_contexts and pattern.ignore_raw_score:
                    continue

                for match in re.finditer(pattern.compiled_regex, record_field.value):
                    start_pos, end_pos = match.span()
                    matched_text = match.group(0)
                    if self.validate_match(matched_text, record_field.value):
                        if header_label_match:
                            _score = pattern.context_score
                        else:
                            if is_context_matched(
                                record_field.value,
                                start_pos,
                                end_pos,
                                pattern.span_contexts,
                            ):
                                _score = pattern.context_score
                            elif pattern.ignore_raw_score:
                                continue
                            else:
                                _score = pattern.raw_score

                        field_matches.add(
                            NERPrediction(
                                text=matched_text,
                                start=start_pos,
                                end=end_pos,
                                field=record_field.field,
                                value_path=record_field.value_path,
                                score=_score,
                                label=self.entity.tag if self.entity else self.source,
                                source=self.source,
                            )
                        )

        filtered_results = map(self.filter_by_range_by_score, result_set_by_field)

        if res_by_field:
            return [list(res_set) for res_set in result_set_by_field]

        results_flat = itertools.chain.from_iterable(filtered_results)
        results = sorted(results_flat, key=lambda i: i.score, reverse=True)

        return list(results)

    @classmethod
    def from_pattern(cls, pattern: Pattern, name: str = None, namespace: str = None):
        return cls(patterns=[pattern], name=name, namespace=namespace)

    @classmethod
    def from_regex(cls, regex: str, name: str = None, namespace: str = None):
        pattern = Pattern(pattern=regex, raw_score=Score.MAX)
        return RegexPredictor.from_pattern(pattern, name=name, namespace=namespace)


@dataclass
class PhrasePatterns:
    case: list[str] = field(default_factory=list)
    no_case: list[str] = field(default_factory=list)

    def to_strings(self):
        return "|".join(self.case), "|".join(self.no_case)


class PhraseMatcherBuilder:
    """Build specialized ``RegexPredictor`` objects that are designed to match
    on phrases in a many-to-one relationship between phrases and entities. This
    utilizes ``RegexPredictors`` and a simple way by constructing single regexes
    that logically "OR" together many phrases and simply set the raw score to
    HIGH.

    Once this is init'd, phrases can be added and the regex patterns will be
    mapped per-entity. A list of ``RegexPredictors`` can be exported at any time.
    """

    phrase_patterns: dict[str | Entity, PhrasePatterns]
    """Map all labels to an object that holds a list of phrases to match. An arbitrary
    string or a specified entity can be used. This will determine how the actual ``name``
    param is utilized in the exported ``RegexPredictor`` objects
    """

    def __init__(self, name: str, *, namespace: Optional[str] = "safe-synthesizer"):
        self.phrase_patterns = defaultdict(PhrasePatterns)  # type: Dict[str, PhrasePatterns]
        """Map each unique label to two lists, one for case insensitive and one for
        case sensitive matches
        """
        self.name = name
        self.namespace = namespace

    def add_phrase(self, label: str | Entity, phrase: str, case=False):
        """Take a simple phrase and modify it to become a regex"""
        # escape special chars
        phrase = phrase.replace(".", r"\.")
        phrase = phrase.replace(" ", r"\s")
        phrase = phrase.replace("$", r"\$")
        phrase = phrase.replace("+", r"\+")
        phrase = phrase.replace("|", r"\|")

        # pad the phrase with word boundaries if they start / end with alphanum
        if phrase[0].isalnum():
            phrase = r"\b" + phrase
        if phrase[-1].isalnum():
            phrase += r"\b"

        phrase_pattern = self.phrase_patterns[label]

        if case:
            phrase_pattern.case.append(phrase)
        else:
            phrase_pattern.no_case.append(phrase)

    def get_predictors(self) -> list[RegexPredictor]:
        out_predictors = []
        for label, phrase_pattern in self.phrase_patterns.items():
            if isinstance(label, Entity):
                _entity = label
            else:
                _entity = None

            _re_patterns = []
            case, nocase = phrase_pattern.to_strings()
            if case:
                _re_patterns.append(Pattern(pattern=re.compile(case), raw_score=Score.PRETTY_HIGH))
            if nocase:
                _re_patterns.append(
                    Pattern(
                        pattern=re.compile(nocase, re.IGNORECASE),
                        raw_score=Score.PRETTY_HIGH,
                    )
                )

            out_predictors.append(
                RegexPredictor(
                    patterns=_re_patterns,
                    entity=_entity,
                    name=self.name,
                    namespace=self.namespace,
                )
            )

        return out_predictors


def phrase_predictors_from_entity_ruler(name: str, er_patterns: list[dict], entity_map: dict) -> list[RegexPredictor]:
    """Given a list of Spacy EntityRuler patterns, create
    a phrase matcher predictor.
    """
    er_label_map = entity_map

    builder = PhraseMatcherBuilder(name)

    for er_pattern in er_patterns:
        _label = er_label_map.get(er_pattern["label"], None)
        # _label = er_pattern["label"]
        if not _label:
            continue
        _pattern = er_pattern["pattern"]
        if isinstance(_pattern, str):
            builder.add_phrase(_label, _pattern, case=True)

        elif isinstance(_pattern, list):
            # this is less effecient than doing a join() on the
            # list of tokens, but we need to evaluate each token
            # to determine if it should have a whitespace added
            # before it

            # seed the base string
            _pattern = iter(_pattern)
            _str = next(_pattern)["LOWER"]

            for part in _pattern:
                part = part["LOWER"]
                if not part.isalnum() and len(part) == 1:
                    _str += part
                else:
                    _str += f" {part}"
            builder.add_phrase(_label, _str)

    return builder.get_predictors()


def create_exact_field_matcher(match: str) -> RePattern:
    """Helper function that takes a full string that should be matched
    for exactly in a field name and put it into a regex that supports
    finding that exact string in potentially flattened fields.

    If we are looking for the word "foo" exactly, we want to support
    looking for it in the following header names:

    "foo"
    "foo.bar"
    "bar.foo"
    "bar.foo.baz"
    """
    sep = r"[\.\-\s_]"
    return re.compile(
        r"(^{m}$)|(^{m}{s})|({s}{m}$)|({s}{m}{s})".format(m=match, s=sep),
        flags=re.IGNORECASE,
    )
