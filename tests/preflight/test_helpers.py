# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``preflight.helpers``."""

from __future__ import annotations

import pytest

from nemo_safe_synthesizer.preflight import IssueCollector, helpers


def _collector() -> IssueCollector:
    return IssueCollector(check_name="test.helper")


@pytest.mark.unit
class TestEmitOnRaise:
    def test_returns_true_when_action_succeeds(self):
        collector = _collector()
        result = helpers.emit_on_raise(
            collector,
            lambda: None,
            expect=ValueError,
            code="bad",
        )
        assert result is True
        assert collector.issues == []

    def test_emits_error_with_message_when_expected_exception_raised(self):
        collector = _collector()

        def boom():
            raise ValueError("nope")

        result = helpers.emit_on_raise(
            collector,
            boom,
            expect=ValueError,
            code="bad_input",
        )
        assert result is False
        assert len(collector.issues) == 1
        issue = collector.issues[0]
        assert issue.severity == "error"
        assert issue.code == "bad_input"
        assert issue.message == "nope"

    def test_emits_warning_when_severity_warning(self):
        collector = _collector()

        def boom():
            raise ValueError("soft")

        helpers.emit_on_raise(
            collector,
            boom,
            expect=ValueError,
            code="soft_issue",
            severity="warning",
        )
        assert collector.issues[0].severity == "warning"

    def test_supports_tuple_of_exception_types(self):
        collector = _collector()

        def boom():
            raise KeyError("missing")

        result = helpers.emit_on_raise(
            collector,
            boom,
            expect=(ValueError, KeyError),
            code="any",
        )
        assert result is False
        assert len(collector.issues) == 1

    def test_unexpected_exception_propagates(self):
        collector = _collector()

        def boom():
            raise RuntimeError("unhandled")

        with pytest.raises(RuntimeError, match="unhandled"):
            helpers.emit_on_raise(
                collector,
                boom,
                expect=ValueError,
                code="x",
            )
        assert collector.issues == []

    def test_empty_expect_tuple_is_rejected(self):
        # ``except ()`` catches nothing, so ``expect=()`` is always a
        # programming error; fail fast rather than degenerate silently.
        collector = _collector()
        with pytest.raises(ValueError, match="catches nothing"):
            helpers.emit_on_raise(
                collector,
                lambda: None,
                expect=(),
                code="x",
            )
        assert collector.issues == []


@pytest.mark.unit
class TestRequireImport:
    def test_returns_module_when_import_succeeds(self):
        collector = _collector()
        mod = helpers.require_import(
            collector,
            "json",
            code="no_json",
            message="json missing",
        )
        assert mod is not None
        assert mod.__name__ == "json"
        assert collector.issues == []

    def test_emits_error_and_returns_none_when_missing(self):
        collector = _collector()
        mod = helpers.require_import(
            collector,
            "nemo_safe_synthesizer__definitely_not_a_module",
            code="missing_dep",
            message="please install xyz",
        )
        assert mod is None
        assert len(collector.issues) == 1
        issue = collector.issues[0]
        assert issue.code == "missing_dep"
        assert issue.severity == "error"
        assert issue.message == "please install xyz"

    def test_emits_warning_when_severity_warning(self):
        collector = _collector()
        helpers.require_import(
            collector,
            "nemo_safe_synthesizer__definitely_not_a_module",
            code="missing_dep",
            message="optional",
            severity="warning",
        )
        assert collector.issues[0].severity == "warning"


def test_resolved_record_count_is_re_exported():
    # Sanity: the helper from checks._helpers is available on the public surface.
    from nemo_safe_synthesizer.preflight.checks._helpers import resolved_record_count as original

    assert helpers.resolved_record_count is original
