# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...base.analyzer import (
    AnalyzerContext,
    ArtifactAnalyzer,
)
from ...base.data_checks import (
    DataCheckResult,
)
from ...base.metrics import timed


class DataChecksAnalyzer(ArtifactAnalyzer):
    """
    Artifact analyzer that runs list of data checks.

    It always runs all the checks, even if one fails, so the result that is
    returned contains all potential warnings, not only the first one.

    Args:
        checks: List of checks to run. They run in the order on the list.
        add_results_to_manifest: Whether to add check results to artifact manifest.
    """

    def __init__(self, checks: list[DataCheck], *, add_results_to_manifest: bool = False):
        self._checks = checks
        self._add_results_to_manifest = add_results_to_manifest

    @timed("DataChecksTime")
    def analyze(self, context: AnalyzerContext) -> None:
        results = [self._run_check(check, context) for check in self._checks]

        context.data_check_results = results

        if self._add_results_to_manifest:
            context.manifest.data_check_results = [_result_to_manifest(r, context) for r in results]

    def _run_check(self, check: DataCheck, context: AnalyzerContext) -> DataCheckResult:
        try:
            return check.run_check(context)
        except Exception as e:
            return DataCheckResult.failed_to_run(check.check_id, error=str(e))


def _result_to_manifest(result: DataCheckResult, context: AnalyzerContext) -> dict[str, Any]:
    """
    Serializes the result to be added to artifact manifest.
    We want to:
    - make sure the size doesn't get too large
    - anonymize all the field names
    """
    result_dict = _filter_none(result.model_dump())

    for warning in result_dict["warnings"]:
        # `explain` can get quite large, so let's just keep the context
        warning.pop("explain")

        if field_names := warning.get("field_names"):
            warning["field_names"] = [context.field_name_anonymizer.anonymize(f) for f in field_names]

        if warning.get("context") is None:
            del warning["context"]

    return result_dict


def _filter_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


class DataCheck(ABC):
    """
    Base class for data checks.

    Each check is identified by its ``check_id`` and needs to implement a method that
    performs the check.
    """

    @property
    @abstractmethod
    def check_id(self) -> str:
        """
        Returns: Unique ID of the check, to be used for reporting, etc.
        """
        ...

    @abstractmethod
    def run_check(self, context: AnalyzerContext) -> DataCheckResult: ...


def plural_noun(noun: str, count: int) -> str:
    """
    Note: this is a very simple mechanism that doesn't deal with any exceptions.
    It only adds "s" if the count is > 1.

    Args:
        noun: Name of the thing.
        count: Number of things.

    Returns: Plural form of a noun, if the count > 1 and original form otherwise.
    """
    return noun if count == 1 else noun + "s"


def plural_verb(verb: tuple[str, str], count: int) -> str:
    """
    Returns one of 2 forms of verb based on count of something.

    Args:
        verb: Tuple with 2 items: first is a singular form, second is plural
        count: Number of things.

    Returns: Singular form of a verb, if the count == 1 and plural otherwise.
    """
    return verb[0] if count == 1 else verb[1]


def warning_explain_prefix(field_names: list[str], issue_name: str) -> str:
    """
    Returns: Common prefix for fields-based warnings.
    """
    count = len(field_names)
    template = "{count} {fields} with {issue_name}: {field_names}"
    return template.format(
        count=count,
        fields=plural_noun("field", count),
        issue_name=issue_name,
        field_names=", ".join(repr(name) for name in field_names),
    )
