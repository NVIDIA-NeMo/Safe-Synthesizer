# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom module for Person Name detection."""

import gzip
import io
import itertools
import re
from dataclasses import dataclass, field
from typing import FrozenSet, Iterable, List

from flashtext import KeywordProcessor

try:
    from re import Pattern
except ImportError:
    from typing import Pattern

from ...data_processing.records.base import KVPair, tokenize_header
from ...data_processing.records.json_record import JSONRecord
from .entity import Entity, Score
from .models import (
    ModelManifest,
    ObjectRef,
    Visibility,
    get_cache_manager,
)
from .ner import NERPrediction
from .predictor import Predictor

DEFAULT_MANIFEST = ModelManifest(
    model="person_name_wordlist",
    version="1",
    sources=[
        ObjectRef(key="word_list", file_name="person_name_word_list.txt.gz"),
        ObjectRef(key="headers", file_name="person_name_headers.txt.gz"),
        ObjectRef(key="headers_neg", file_name="person_name_headers_neg.txt.gz"),
        ObjectRef(key="headers_pairs", file_name="person_name_headers_pairs.txt.gz"),
        ObjectRef(key="parts", file_name="person_name_parts.txt.gz"),
    ],
    visibility=Visibility.INTERNAL,
)

# Match on all words, unless they start with a number
TOKEN_REGEX = re.compile(r"\b(?!\d)\w+", re.IGNORECASE)

MAX_STR_LEN = 64


def build_name_only_headers(others: Iterable[str]) -> List[Pattern]:
    out = []
    for other in others:
        out.append(r"{}.?{}".format("name", other))
        out.append(r"{}.?{}".format(other, "name"))
    return out


@dataclass
class WordList:
    """Where we load the decompressed data from S3 / FS. These attr
    names MUST match the ref names of the files from the manifest
    """

    word_list: KeywordProcessor = field(default_factory=frozenset)
    """The master list of actual names"""

    headers: Pattern = None  # NOTE: init'd as a FrozenSet then converted
    """The list of partial header names that can trigger the prediction flow"""

    headers_neg: KeywordProcessor = None  # NOTE: init'd as a FrozenSet then converted
    """A list of header tokens that should not be present to trigger prediction flow"""

    headers_pairs: FrozenSet[str] = field(default_factory=frozenset)
    """A list of words that can be combined with the word 'name', this should
    be used to build additional header pairs for analysis
    """

    parts: FrozenSet[str] = field(default_factory=frozenset)
    """These are parts of a name that can be used to match, things like
    Mr., Mrs., etc
    """

    def __post_init__(self):
        # we need to take the headers_pairs and construct new positive
        # matching header values from them, and then re-set our master
        # header list
        tmp = build_name_only_headers(self.headers_pairs)
        _header_strings = list(self.headers | frozenset(tmp))
        _header_regex = re.compile("|".join(_header_strings), re.IGNORECASE)
        self.headers = _header_regex

        _neg_headers = KeywordProcessor()
        _neg_headers.add_keywords_from_list(list(self.headers_neg))
        self.headers_neg = _neg_headers

        # self.word_list = re.compile("|".join([w + "$" for w in list(self.word_list)]), re.IGNORECASE)
        _word_list = KeywordProcessor()
        _word_list.add_keywords_from_list(list(self.word_list))
        self.word_list = _word_list

        # _parts = KeywordProcessor()
        # _parts.add_keywords_from_list(list(self.parts))
        # self.parts = _parts

    @classmethod
    def init_from_manifest(cls, manifest: ModelManifest = None):
        if manifest is None:
            manifest = DEFAULT_MANIFEST
        cache_data = get_cache_manager().resolve(manifest, skip_pickle=True)
        if cache_data is None:
            raise RuntimeError("Model cached returned None data for word list")
        kwargs = {}
        # NOTE: this functionality depends on the dataclass attrs
        # being named the same as the keys in the ``ObjectRef``
        # instances since those keys are what is returned
        # by the cache manager resolution
        for key, raw_bytes in cache_data.items():
            reader = gzip.GzipFile(fileobj=io.BytesIO(raw_bytes))
            tmp = [line.decode().strip() for line in reader]
            kwargs[key] = frozenset(tmp)
        return cls(**kwargs)


class PersonNamePredictor(Predictor):
    default_name: str = "person_name"

    def __init__(self):
        super().__init__(self.default_name)
        self.word_list = WordList.init_from_manifest()

    def create_prediction(self, record_field: KVPair):
        return NERPrediction(
            text=record_field.value,
            start=0,
            end=len(record_field.value),
            field=record_field.field,
            value_path=record_field.value_path,
            score=Score.HIGH,
            label=Entity.PERSON_NAME.tag,
            source=self.source,
        )

    def check_exact_name_header_data(self, field_value: str) -> bool:
        """The logic here is that
        1) Every token must exist in one of three lists
        2) At least one of the tokens must exist in the main name list
        """
        tokens = [token.lower() for token in re.findall(TOKEN_REGEX, field_value)]
        _found = False
        for token in tokens:
            if len(token) == 1:
                continue

            in_main_word_list = self.word_list.word_list.extract_keywords(token)

            # if the token is not in any of these lists, fail
            if (
                not in_main_word_list
                and not self.word_list.headers.match(token)
                # and token not in self.word_list.headers
                and token not in self.word_list.parts
            ):
                return False
            # the token is one of these three lists, if we haven't
            # found a token that's also in the main word_list yet,
            # we check that here
            if not _found and in_main_word_list:
                _found = True

        return _found

    def _is_neg_header_in_value(self, value) -> bool:
        value_tokens = tokenize_header(str(value))
        for _token in value_tokens:
            # if _token in self.word_list.headers_neg:
            if self.word_list.headers_neg.match(_token):
                return True
        return False

    def evaluate(self, in_record: JSONRecord) -> List[NERPrediction]:
        record_fields = in_record.kv_pairs
        result_set_by_field = [set() for _ in record_fields]

        record_field: KVPair
        for field_matches, record_field in zip(result_set_by_field, record_fields):
            record_field.value = str(record_field.value)

            # check if the current value's strlen is too big
            if len(record_field.value) > MAX_STR_LEN:
                continue

            # check if any negative header fields exist
            for header_token in record_field.field_tokens:
                if self.word_list.headers_neg.extract_keywords(header_token):
                    continue

            # tokenize the value and see if any of tokens exist
            # in the negative header list
            # if record_field.value and self._is_neg_header_in_value(record_field.value):
            if self.word_list.headers_neg.extract_keywords(record_field.value):
                continue

            if not self.header_has_context(record_field, self.KEY, regex_patterns=self.word_list.headers):
                # specialy handling with the field name is exactly "name", we need
                # to check if every token in the value exists in one of our specific
                # name or modifier lists
                if record_field.field.lower() in ("name", "names"):
                    if self.check_exact_name_header_data(record_field.value):
                        field_matches.add(self.create_prediction(record_field))
                # either way, we will not continue processing beyond this
                # point for missing header context or exact "name" field name
                continue

            # tokenize, and evaluate all tokens against the word list
            # for token in re.finditer(TOKEN_REGEX, record_field.value):
            #    token_str = token.group(0).lower()

            if not self.word_list.word_list.extract_keywords(record_field.value):
                continue

            # if the token is in the word list, we consider
            # the entire field value to be a name
            field_matches.add(self.create_prediction(record_field))

        preds = list(itertools.chain.from_iterable(result_set_by_field))
        return preds
