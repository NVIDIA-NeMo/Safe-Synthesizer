# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from dateparser import parse
from dateparser.date import get_date_from_timestamp
from dateparser.search import search_dates

from ...data_processing.records.json_record import JSONRecord
from .entity import (
    Entity,
    Score,
)
from .ner import NERPrediction
from .predictor import Predictor, PredictorContext
from .regex import (
    create_exact_field_matcher,
    split_header_contexts,
)

LABELS = [
    "timestamp",
    "issued",
    "expired",
    "collected",
    "gathered",
    "retrieved",
    "date",
    "time",
    "when",
    "upload",
    "uploaded",
    "download",
    "downloaded",
    "sent",
    "send",
    "receive",
    "received",
    "utc",
    "zulu",
    "schedule",
    "scheduled",
    "login",
    "logon",
    "logoff",
    "logout",
    "check",  # checkin, checkout, etc
    "open",  # when opened, file open
    "opened",
    "close",  # when closed
    "closed",
    "create",
    "created",
    "delete",
    "deleted",
    "update",
    "updated",
    "amend",
    "amended",
    "launch",
    "launched",
    "embark",
    "embarked",
    "take",
    "took",
    "leave",
    "left",
    "arrive",
    "arrived",
    "init",
    "convene",
    "convened",
    "during",
    "while",
    "day",
    "month",
    "year",
    "hour",
    "minute",
    "visit",
    "visited",
    "from",
    "to",
    create_exact_field_matcher("ts"),
]

BIRTH_LABELS = ["birth", "dob", "yob", "birthday", "birthdate", "birthdates"]

PARSERS = ["timestamp", "absolute-time"]


@dataclass
class BaseContext(PredictorContext):
    header_contexts: list
    header_regexes: list = None
    header_tokens: list = None

    def __post_init__(self):
        self.header_regexes, self.header_tokens = split_header_contexts(self.header_contexts)


@dataclass
class DateTimeContext(BaseContext):
    def get_entity_label(self, match: tuple):
        return Entity.DATETIME.tag


@dataclass
class BirthDateContext(BaseContext):
    def get_entity_label(self, _):
        return Entity.BIRTH_DATE.tag


def _parse_dates(value: str | int | float, scalar_type: Optional[str] = None) -> Optional[list[tuple[str, datetime]]]:
    if scalar_type == "number" and isinstance(value, str):
        # TODO(PROD-276): this is necessary, as our regex predictors change values from kv_pair
        #  to strings, so we need to use "scalar_type" field to figure out if the
        #  original value was a number.
        try:
            value = float(value)
        except Exception:
            pass

    if isinstance(value, float) and math.isnan(value):
        # don't try to match something that is NaN
        # when converted to string, "nan" is name for August
        # in this locale: https://www.localeplanet.com/icu/mgh/index.html
        return None

    if isinstance(value, int) or isinstance(value, float):
        value = str(value)
        # since this is a single number - try to parse as timestamp
        check = get_date_from_timestamp(value, settings=None)
        if check is not None:
            return [(value, check)]

        # int or float can only be a timestamp.
        # if we didn't find one - we don't need to keep looking
        return None

    if not isinstance(value, str):
        value = str(value)

    # first try checking the entire string as potential match
    if len(value) < 32:
        check = parse(value)
        if check is not None:
            # we return it in the same format as the search_dates() function
            # so things can be handled the same way in the calling fn
            return [(value, check)]

    date_time_list = search_dates(
        value,
        languages=["en"],
        settings={"STRICT_PARSING": True, "PARSERS": PARSERS},
    )

    return date_time_list


class DateTime(Predictor):
    """
    Date date/time matcher.
    """

    default_name: str = "datetime"

    def __init__(self, name: str = None):
        if name is None:
            name = self.default_name
        super().__init__(name)
        self._context = DateTimeContext(LABELS)

    def evaluate(self, in_record: JSONRecord) -> list[NERPrediction]:
        """
        Given a single record determine if any
        entities are represented.

        Args:
            in_record: the record to match patterns against

        Returns:
            A list of entity predictions sorted by score. Top score is first entry in list.
        """
        result_set_by_field = [[] for _ in in_record.kv_pairs]
        for field_matches, record_field in zip(result_set_by_field, in_record.kv_pairs):
            # NOTE(jm): Changed to require header context no matter what, too many
            # FPs when looking in unstructured text
            if not self.header_has_context(
                record_field,
                self.KEY,
                token_patterns=self._context.header_tokens,
                regex_patterns=self._context.header_regexes,
            ):
                continue

            try:
                date_time_list = _parse_dates(record_field.value, record_field.scalar_type)
            except Exception:
                # NOTE:(jm) skip over issues like this: https://github.com/scrapinghub/dateparser/issues/679
                continue

            if date_time_list:
                for date_time in date_time_list:
                    label = self._context.get_entity_label(date_time)
                    start = str(record_field.value).find(date_time[0])
                    end = start + len(date_time[0])
                    matched_text = date_time[0]

                    field_matches.append(
                        NERPrediction(
                            text=matched_text,
                            start=start,
                            end=end,
                            field=record_field.field,
                            value_path=record_field.value_path,
                            score=Score.HIGH,
                            label=label,
                            source=self.source,
                        )
                    )

        date_matches = list(itertools.chain.from_iterable(result_set_by_field))
        results = sorted(date_matches, key=lambda i: i.score, reverse=True)
        return results


class BirthDateTime(DateTime):
    default_name: str = "birth_date"

    def __init__(self):
        super().__init__(self.default_name)
        self._context = BirthDateContext(BIRTH_LABELS)
