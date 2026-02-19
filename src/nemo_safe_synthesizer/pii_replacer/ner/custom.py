# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Module that supports reading in custom Regex predictors from a config file.

Support reading in a YAML file of user defined predictor properties and load
them into individual Regex predictors and patterns.

Example YAML:
------
namespace: acme  # required prefix for all entity names
regex:
    user_id:  # the name of the entity that will be created
        patterns:
            - score: high  # one of high, med, low
              regex: "^user_id[\\d]{8}_[A-Z]{3}"  # the actual regex
              header_match:  # regexes for header matches
                - uid
                - user
                - member
phrase:
    names:  # the name of the entity that will be created
        paths:
            - path: /path/to/wordlist.txt
              case: yes # default is "no"
------
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from re import Pattern
from typing import Optional

import yaml

from ...observability import get_logger
from .entity import Score
from .predictor import Predictor
from .regex import Pattern as RegexPattern
from .regex import PhraseMatcherBuilder, RegexPredictor

logger = get_logger(__name__)

score_map = {"low": Score.LOW, "med": Score.MED, "high": Score.HIGH}

PatternListType = list[re.Pattern]


def _str_to_pattern(data: str | list[str]) -> PatternListType:
    if isinstance(data, str):
        return [re.compile(data)]

    return [re.compile(_str) for _str in data]


class CustomPredictorError(Exception):
    pass


@dataclass
class CustomRegexPattern:
    """A simplified version of the internal Pattern object used by
    custom managed Regex Patterns. This can be expanded as needed
    to provide more user-facing features for BYO patterns
    """

    score: str
    regex: str
    header_match: Optional[str | list[str]] = field(default_factory=list)
    header_skip: Optional[str | list[str]] = field(default_factory=list)

    # set after init
    regex_compiled: re.Pattern = None
    header_match_compiled: PatternListType = field(default_factory=list)
    header_skip_compiled: PatternListType = field(default_factory=list)

    def __post_init__(self):
        if self.score not in score_map:
            raise CustomPredictorError("score must be one of low, med, high")

        self.regex_compiled = self.regex
        if not isinstance(self.regex, Pattern):
            self.regex_compiled = re.compile(str(self.regex))

        self._load_header_patterns()

    def _load_header_patterns(self):
        if self.header_match:
            self.header_match_compiled = _str_to_pattern(self.header_match)

        if self.header_skip:
            self.header_skip_compiled = _str_to_pattern(self.header_skip)

    def get_synthesizer_pattern(self):
        _score = score_map[self.score]
        return RegexPattern(
            pattern=self.regex_compiled,
            context_score=_score,
            raw_score=_score,
            header_contexts=self.header_match_compiled,
            neg_header_contexts=self.header_skip_compiled,
            ignore_raw_score=False if not self.header_match_compiled else True,
        )


@dataclass
class CustomRegexPredictor:
    namespace: str
    name: str
    patterns: list[CustomRegexPattern]

    @classmethod
    def from_config(cls, name: str, namespace: str, patterns: list[dict]):
        _patterns = [CustomRegexPattern(**p) for p in patterns]
        return cls(name=name, namespace=namespace, patterns=_patterns)

    def get_safe_synthesizer_predictor(self):
        patterns = [p.get_synthesizer_pattern() for p in self.patterns]
        return RegexPredictor(name=self.name, namespace=self.namespace, patterns=patterns)


namespace_validator = re.compile(r"^[A-Za-z][A-Za-z_]{2,16}")


def _namespace_from_config(config: dict) -> str:
    namespace = config.get("namespace", None)
    if namespace is None:
        raise CustomPredictorError("namespace is required")
    if not namespace_validator.match(namespace):
        raise CustomPredictorError(
            "namespace must start with a letter and only contain letters and underscores, min 3 chars and max 16 chars"
        )  # noqa
    return namespace


#####################
# Config Dict Parsers
#####################


def get_regex_predictors_from_config(config: dict) -> Optional[list[RegexPredictor]]:
    out_predictors = []
    namespace = _namespace_from_config(config)

    predictor_dicts = config.get("regex", None)
    if predictor_dicts is None:
        return

    for name, patterns in predictor_dicts.items():
        predictor_name = name.lower()
        custom_predictor = CustomRegexPredictor.from_config(predictor_name, namespace, patterns["patterns"])
        out_predictors.append(custom_predictor.get_safe_synthesizer_predictor())

    return out_predictors


def _process_phrase_list_file(config: dict, builder: PhraseMatcherBuilder) -> PhraseMatcherBuilder:
    _path = config.get("path", None)
    _case = config.get("case", None)

    if _path is None:
        raise CustomPredictorError("missing path for phrase list config")

    if not Path(_path).is_file():
        raise CustomPredictorError(f"phrase list error: {_path} is not a valid file")

    _case = True if _case == "yes" else False

    logger.info("Processing phrase list from file: %s", _path)

    with open(_path) as fin:
        for line in fin:
            line = line.rstrip("\n")
            builder.add_phrase("__custom__", line, case=_case)

    return builder


def get_phrase_predictors_from_config(config: dict) -> Optional[list[RegexPredictor]]:
    out_predictors = []
    namespace = _namespace_from_config(config)

    phrase_predictors = config.get("phrase", None)
    if phrase_predictors is None:
        return

    for predictor_name, phrase_config in phrase_predictors.items():
        predictor_name = predictor_name.lower()
        builder = PhraseMatcherBuilder(predictor_name, namespace=namespace)

        # Handle any path based configurations
        path_config_list = phrase_config.get("paths", None)
        if path_config_list is None:
            continue

        for path_config_dict in path_config_list:
            builder = _process_phrase_list_file(path_config_dict, builder)

        # TODO: Handle any in-line phrase declarations

        out_predictors.extend(builder.get_predictors())

    return out_predictors


#########################
# YAML Predictor Parsers
#########################


def _yaml_to_dict(file_path: str) -> dict:
    with open(file_path) as f:
        config_dict = yaml.safe_load(f)
    return config_dict


def get_regex_predictors_from_yaml(file_path: str) -> list[RegexPredictor]:
    return get_regex_predictors_from_config(_yaml_to_dict(file_path))


def get_phrase_predictors_from_yaml(file_path: str) -> list[RegexPredictor]:
    return get_phrase_predictors_from_config(_yaml_to_dict(file_path))


#####################
# Primary Entrypoint
#####################


def get_predictors_from_yaml(file_path: str) -> list[Predictor]:
    out = []
    parsing_fns = (get_regex_predictors_from_yaml, get_phrase_predictors_from_yaml)
    for fn in parsing_fns:
        predictors = fn(file_path)
        if predictors:
            out.extend(predictors)
    return out
