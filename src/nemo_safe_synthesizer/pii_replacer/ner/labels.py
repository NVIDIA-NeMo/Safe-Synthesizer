# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from typing import Iterator, Pattern

from ...data_processing.records.base import normalize_label, normalize_labels
from ...observability import get_logger
from .entity import Entity

logger = get_logger(__name__)


def get_built_in_labels() -> list[str]:
    labels = [normalize_label(key) for key in Entity.__members__.keys()]

    # FIXME(pm): This is hacky way to prioritize email_address transform, so that
    #  domain_name transform is later on the list
    labels.remove("email_address")
    return ["email_address"] + labels


class LabelEvaluator:
    """
    Evaluates labels specified by the user in the config and provides a simple
    interface other places in the code that use that label configuration.

    One notable example is expanding wildcards from the label config (e.g. ``acme/*`` or ``*``).
    """

    def __init__(self, explicit_labels: set[str], label_regexes: list[Pattern]):
        self._explicit_labels = normalize_labels(explicit_labels)
        self._label_regexes = label_regexes

    def filter_labels(self, labels: list[str]) -> Iterator[str]:
        """
        Filters provided list of labels against configured labels and label regexes.

        Example::

            evaluator = LabelEvaluator(explicit_labels=["test"], label_regexes=["^acme/.*$"])
            filtered = evaluator.filter_labels(["test", "test_2", "acme/abc", "test/test"])
            assert list(filtered) == "test", "acme/abc"

        Args:
            labels: List of labels to be filtered.

        Returns: Filtered labels as they are calculated.
        """
        # check explicit labels
        for label in labels:
            if normalize_label(label) in self._explicit_labels:
                yield label

        # check wildcard labels
        if self._label_regexes:
            for label in labels:
                if self._matches_any_regex(label):
                    yield label

    def any_label_configured(self, labels: list[str]) -> bool:
        # checks if there is any item in the filtered list
        return next(self.filter_labels(labels), None) is not None

    def _matches_any_regex(self, label: str):
        return any(regex.match(label) for regex in self._label_regexes)

    def explicit_lables(self) -> set[str]:
        return self._explicit_labels

    @classmethod
    def create_from_config(cls, config_labels: list[str]) -> LabelEvaluator:
        """
        Loads labels defined by the user in the config.

        Args:
            config_labels: Labels configured by the users.
        """

        explicit_labels = set([])
        label_regexes: list[Pattern] = []

        for label in config_labels:
            if "*" not in label:
                explicit_labels.add(label)
            else:
                # there is a wildcard
                parts = label.split("/")
                if len(parts) == 2:
                    namespace, entity = parts
                    # match all labels inside a namespace
                    label_regexes.append(re.compile(rf"^{namespace}/.+$", re.IGNORECASE))

                elif len(parts) == 1:
                    # match all labels that don't have namespace
                    label_regexes.append(re.compile(r"^[^/]+$", re.IGNORECASE))

                else:
                    logger.warning(f"Invalid label specification '{label}'. Skipping.")

        return cls(explicit_labels, label_regexes)
