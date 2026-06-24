# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``build_registry`` / ``_validate_registry`` shape invariants."""

from __future__ import annotations

import pytest
from typing_extensions import override

from nemo_safe_synthesizer.preflight import (
    AdvisoryCheck,
    ConfigCheck,
    ConfigView,
    DataFrameCheck,
    DataFrameView,
    IssueCollector,
    MetadataCheck,
    MetadataView,
    build_registry,
)
from nemo_safe_synthesizer.preflight.registry import _validate_registry


class _NoopConfig(ConfigCheck):
    name = "plugintest.noop_cfg"
    label = "Noop config"

    @override
    def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
        return


class _NoopConfigB(ConfigCheck):
    name = "plugintest.noop_cfg_b"
    label = "Noop config B"

    @override
    def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
        return


class _NoopDataFrame(DataFrameCheck):
    name = "plugintest.noop_df"
    label = "Noop df"

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        return


class _NoopMetadata(MetadataCheck):
    name = "plugintest.noop_meta"
    label = "Noop meta"

    @override
    def check(self, ctx: MetadataView, collector: IssueCollector) -> None:
        return


class _NoopAdvisory(AdvisoryCheck):
    name = "plugintest.noop_adv"
    label = "Noop advisory"

    @override
    def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
        return


@pytest.mark.unit
def test_build_registry_accepts_valid_sources():
    registry = build_registry((_NoopConfig(), _NoopConfigB()), (_NoopDataFrame(),))
    assert [c.name for c in registry] == [
        "plugintest.noop_cfg",
        "plugintest.noop_cfg_b",
        "plugintest.noop_df",
    ]


@pytest.mark.unit
def test_build_registry_rejects_duplicate_names():
    with pytest.raises(RuntimeError, match="Duplicate"):
        build_registry((_NoopConfig(), _NoopConfig()))


@pytest.mark.unit
def test_build_registry_rejects_unknown_requires():
    class NeedsMissing(ConfigCheck):
        name = "plugintest.needs_missing"
        label = "Needs missing"
        requires = ("plugintest.does_not_exist",)

        @override
        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    with pytest.raises(RuntimeError, match="unknown or not declared earlier"):
        build_registry((NeedsMissing(),))


@pytest.mark.unit
def test_build_registry_rejects_out_of_order_requires():
    class LeaderConfig(ConfigCheck):
        name = "plugintest.leader"
        label = "Leader"

        @override
        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    class NeedsLeader(ConfigCheck):
        name = "plugintest.needs_leader"
        label = "Needs leader"
        requires = ("plugintest.leader",)

        @override
        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    with pytest.raises(RuntimeError, match="unknown or not declared earlier"):
        build_registry((NeedsLeader(), LeaderConfig()))


@pytest.mark.unit
def test_build_registry_sorts_by_stage():
    """Plugins appended after core still slot into the correct stage block."""
    registry = build_registry((_NoopAdvisory(), _NoopConfig()))
    assert [c.name for c in registry] == ["plugintest.noop_cfg", "plugintest.noop_adv"]


@pytest.mark.unit
def test_validate_registry_rejects_backwards_stages():
    """Direct validation still requires stage-monotonic ordering."""
    with pytest.raises(RuntimeError, match="stage-monotonic"):
        _validate_registry((_NoopAdvisory(), _NoopConfig()))
