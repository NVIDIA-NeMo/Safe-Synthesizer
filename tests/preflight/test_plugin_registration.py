# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public preflight plugin surface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import nemo_safe_synthesizer.preflight as pf_mod
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.preflight import PreflightParameters
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.preflight import (
    CRASH_CODE,
    ConfigCheck,
    ConfigView,
    DataFrameCheck,
    DataFrameView,
    IssueCollector,
    PreflightIssue,
    PreflightStage,
    build_registry,
    register_preflight_check,
    reset_preflight_plugins,
    run_preflight,
)
from nemo_safe_synthesizer.preflight.orchestrator import _run_registry

from .conftest import make_ctx


@pytest.fixture(autouse=True)
def _restore_registry():
    """Reset registered plugins to a clean slate between tests."""
    reset_preflight_plugins()
    try:
        yield
    finally:
        reset_preflight_plugins()


# ---------------------------------------------------------------------------
# register_preflight_check
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_preflight_check_appends_to_registry():
    class MyCheck(ConfigCheck):
        name = "myplugin.registered"
        label = "Registered plugin"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            collector.warning("myplugin_fired", "hello")

    instance = MyCheck()
    register_preflight_check(instance)
    assert pf_mod.get_registry()[instance.name] is instance
    # Plugin slots into its stage block (CONFIG here), so it is the last
    # entry of that stage block rather than the end of the registry.
    config_stage_checks = [c for c in pf_mod.get_registry() if c.stage is PreflightStage.CONFIG]
    assert config_stage_checks[-1] is instance


@pytest.mark.unit
def test_plugin_check_fires_via_run_preflight():
    class MyCheck(ConfigCheck):
        name = "myplugin.fires"
        label = "Plugin fires"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            collector.warning("plugin_signal", "plugin ran")

    register_preflight_check(MyCheck())
    config = SafeSynthesizerParameters()
    metadata = MagicMock(spec=ModelMetadata)
    metadata.tokenizer = None

    with patch("torch.cuda.is_available", return_value=True):
        with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "x", "HF_TOKEN": "y"}):
            report = run_preflight(pd.DataFrame({"a": range(300)}), config, metadata)

    assert any(c.name == "myplugin.fires" for c in report.checks)
    assert any(i.code == "plugin_signal" for i in report.issues)


# ---------------------------------------------------------------------------
# Namespace enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plugin_rejects_reserved_core_namespace():
    class BadPlugin(ConfigCheck):
        name = "gpu.rogue"
        label = "Rogue plugin"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    with pytest.raises(ValueError, match="reserved core namespace"):
        register_preflight_check(BadPlugin())


# ---------------------------------------------------------------------------
# Registration rollback on validation failure
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_register_preflight_check_rolls_back_on_duplicate_name():
    """A failed registration must not poison ``_PLUGIN_CHECKS``.

    Register a legitimate plugin, then attempt to register a second
    plugin with the *same* name. The second call must raise, and a
    third registration with a unique name must still succeed --
    which would not be the case if the failed second entry had been
    appended to the internal list.
    """

    class First(ConfigCheck):
        name = "myplugin.first"
        label = "First"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    class Duplicate(ConfigCheck):
        name = "myplugin.first"
        label = "Also first"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    class Third(ConfigCheck):
        name = "myplugin.third"
        label = "Third"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    register_preflight_check(First())
    with pytest.raises(RuntimeError, match="Duplicate"):
        register_preflight_check(Duplicate())
    register_preflight_check(Third())

    assert "myplugin.first" in pf_mod.get_registry().checks
    assert "myplugin.third" in pf_mod.get_registry().checks


# ---------------------------------------------------------------------------
# API version enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plugin_rejects_mismatched_api_version():
    def _noop_check(self: ConfigCheck, ctx: ConfigView, collector: IssueCollector) -> None:
        return

    with pytest.raises(TypeError, match="preflight API"):
        # Use ``type()`` so ``__init_subclass__`` fires (and raises) without
        # introducing an unused local class binding.
        type(
            "FuturePlugin",
            (ConfigCheck,),
            {
                "__preflight_api_version__": 2,
                "name": "myplugin.future",
                "label": "Future",
                "check": _noop_check,
            },
        )


# ---------------------------------------------------------------------------
# Metadata enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_plugin_must_define_required_attrs():
    def _noop_check(self: ConfigCheck, ctx: ConfigView, collector: IssueCollector) -> None:
        return

    with pytest.raises(TypeError, match="must define"):
        # Use ``type()`` so ``__init_subclass__`` fires (and raises) without
        # introducing an unused local class binding. ``name`` is omitted so
        # the "must define" branch triggers.
        type(
            "Incomplete",
            (ConfigCheck,),
            {
                "label": "no name",
                "check": _noop_check,
            },
        )


# ---------------------------------------------------------------------------
# Duplicate-name collision with core
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_registry_rejects_plugin_colliding_with_core():
    class CoreCollision(ConfigCheck):
        # Plugins cannot use core namespace prefixes like "gpu"; pick a
        # non-core namespace but then manually collide a fake "core" check
        # via build_registry directly.
        name = "myplugin.collides"
        label = "Collides"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    a = CoreCollision()
    b = CoreCollision()
    with pytest.raises(RuntimeError, match="Duplicate"):
        build_registry((a, b))


# ---------------------------------------------------------------------------
# config.preflight.disabled_checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_disabled_checks_skips_named_check(sample_df):
    config = SafeSynthesizerParameters(preflight=PreflightParameters(disabled_checks=["gpu.cuda"]))
    metadata = MagicMock(spec=ModelMetadata)
    metadata.tokenizer = None

    with patch("torch.cuda.is_available", return_value=True):
        with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "x", "HF_TOKEN": "y"}):
            report = run_preflight(sample_df, config, metadata)

    assert not any(c.name == "gpu.cuda" for c in report.checks)


@pytest.mark.unit
def test_preflight_parameters_roundtrip():
    params = SafeSynthesizerParameters(preflight=PreflightParameters(disabled_checks=["gpu.vram"]))
    dumped = params.model_dump()
    assert dumped["preflight"]["disabled_checks"] == ["gpu.vram"]
    restored = SafeSynthesizerParameters(**dumped)
    assert restored.preflight.disabled_checks == ["gpu.vram"]


@pytest.mark.unit
def test_disabled_dep_gates_dependents():
    class PrereqPlugin(ConfigCheck):
        name = "myplugin.prereq"
        label = "Prereq"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            collector.warning("prereq_ran", "should not fire when disabled")

    class DependentPlugin(ConfigCheck):
        name = "myplugin.dependent"
        label = "Dependent"
        requires = ("myplugin.prereq",)

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            collector.warning("dependent_ran", "should not fire when prereq disabled")

    register_preflight_check(PrereqPlugin())
    register_preflight_check(DependentPlugin())

    config = SafeSynthesizerParameters(preflight=PreflightParameters(disabled_checks=["myplugin.prereq"]))
    metadata = MagicMock(spec=ModelMetadata)
    metadata.tokenizer = None

    with patch("torch.cuda.is_available", return_value=True):
        with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "x", "HF_TOKEN": "y"}):
            report = run_preflight(pd.DataFrame({"a": range(300)}), config, metadata)

    by_name = {c.name: c for c in report.checks}
    # Prereq is disabled by config -- it is filtered before the orchestrator
    # sees it, so it has no result entry.
    assert "myplugin.prereq" not in by_name
    # Dependent surfaces as a skipped result so the user can see it was gated.
    assert by_name["myplugin.dependent"].status == "skipped"
    assert not by_name["myplugin.dependent"].issues
    assert not any(i.code == "dependent_ran" for i in report.issues)


@pytest.mark.unit
def test_unknown_disabled_checks_name_warns():
    config = SafeSynthesizerParameters(preflight=PreflightParameters(disabled_checks=["myplugin.does_not_exist"]))
    metadata = MagicMock(spec=ModelMetadata)
    metadata.tokenizer = None

    with patch("nemo_safe_synthesizer.preflight.orchestrator.logger.user.warning") as mock_warn:
        with patch("torch.cuda.is_available", return_value=True):
            with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "x", "HF_TOKEN": "y"}):
                run_preflight(pd.DataFrame({"a": range(300)}), config, metadata)

    assert mock_warn.called
    rendered = " ".join(
        str(arg) for call in mock_warn.call_args_list for arg in (call.args + tuple(call.kwargs.values()))
    )
    assert "myplugin.does_not_exist" in rendered
    assert "disabled_checks" in rendered


@pytest.mark.unit
def test_run_preflight_accepts_custom_registry():
    class SoloCheck(ConfigCheck):
        name = "myplugin.solo"
        label = "Solo"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            collector.warning("solo_ran", "solo")

    metadata = MagicMock(spec=ModelMetadata)
    metadata.tokenizer = None

    from nemo_safe_synthesizer.preflight import build_registry

    report = run_preflight(
        pd.DataFrame({"a": [1, 2, 3]}),
        SafeSynthesizerParameters(),
        metadata,
        registry=build_registry((SoloCheck(),)),
    )

    assert [c.name for c in report.checks] == ["myplugin.solo"]
    assert any(i.code == "solo_ran" for i in report.issues)


# ---------------------------------------------------------------------------
# Failure isolation + namespace property
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_check_crash_is_reported_and_does_not_halt_registry():
    class Crasher(ConfigCheck):
        name = "myplugin.crasher"
        label = "Crasher"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            raise RuntimeError("boom")

    class Follower(ConfigCheck):
        name = "myplugin.follower"
        label = "Follower"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            collector.warning("followed", "ran after crash")

    from nemo_safe_synthesizer.preflight import build_registry

    registry = build_registry((Crasher(), Follower()))
    results = _run_registry(make_ctx(), registry)
    by_name = {r.name: r for r in results}
    assert [r.name for r in results] == ["myplugin.crasher", "myplugin.follower"]
    crash_issues = [i for r in results for i in r.issues if i.code == CRASH_CODE]
    assert len(crash_issues) == 1
    assert "RuntimeError: boom" in crash_issues[0].message
    assert by_name["myplugin.crasher"].status == "failed"
    assert by_name["myplugin.follower"].status == "passed"


@pytest.mark.unit
def test_issue_namespace_returns_prefix_or_none():
    dotted = PreflightIssue(code="x", severity="warning", check="myplugin.foo", message="m")
    bare = PreflightIssue(code="x", severity="warning", check="cardinality", message="m")
    assert dotted.namespace == "myplugin"
    assert bare.namespace is None


# ---------------------------------------------------------------------------
# Stage preservation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stage_markers_are_preserved_on_plugin_subclasses():
    class P1(ConfigCheck):
        name = "myplugin.stage_cfg"
        label = "Plugin cfg"

        def check(self, ctx: ConfigView, collector: IssueCollector) -> None:
            return

    class P2(DataFrameCheck):
        name = "myplugin.stage_df"
        label = "Plugin df"

        def check(self, ctx: DataFrameView, collector: IssueCollector) -> None:
            return

    assert P1.stage is PreflightStage.CONFIG
    assert P2.stage is PreflightStage.DATAFRAME
