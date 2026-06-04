# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``vllm_observability``.

Scope: contracts consumers depend on — degraded-mode behavior, stable
dict keys, schema round-trip, flattening semantics. NOT testing
implementation details (field counts, log wording, init constants);
those would break on every refactor without adding value.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from nemo_safe_synthesizer.generation.vllm_observability import (
    ENGINE_CONFIG_CHECKED_FIELDS,
    METRIC_KV_CACHE_USAGE_PERC,
    METRIC_PREFIX_CACHE_HITS,
    METRIC_PREFIX_CACHE_QUERIES,
    METRIC_SPEC_NUM_ACCEPTED_TOKENS,
    METRIC_SPEC_NUM_DRAFT_TOKENS,
    GenerationObservability,
    NvmlPeakSampler,
    flag_engagement_mismatches,
    probe_engine_runtime_config,
    read_loadavg,
    read_vllm_runtime_metrics,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _metric(name: str, value: float | int) -> Any:
    """Build a minimal metric-like object with ``.name`` and ``.value``."""
    m = MagicMock()
    m.name = name
    m.value = value
    return m


@pytest.fixture
def mock_llm_no_metrics() -> Any:
    """LLM mock that returns an empty metric list."""
    llm = MagicMock()
    llm.get_metrics.return_value = []
    return llm


@pytest.fixture
def mock_llm_full_metrics() -> Any:
    """LLM mock returning all known counters/gauges."""
    llm = MagicMock()
    llm.get_metrics.return_value = [
        _metric(METRIC_KV_CACHE_USAGE_PERC, 0.42),
        _metric(METRIC_PREFIX_CACHE_HITS, 80),
        _metric(METRIC_PREFIX_CACHE_QUERIES, 100),
        _metric(METRIC_SPEC_NUM_ACCEPTED_TOKENS, 720),
        _metric(METRIC_SPEC_NUM_DRAFT_TOKENS, 1000),
    ]
    return llm


@pytest.fixture
def populated_event() -> GenerationObservability:
    """A GenerationObservability with one value of each non-trivial field shape."""
    return GenerationObservability(
        peak_vram_gb=64.5,
        kv_cache_usage_perc=0.42,
        prefix_cache_hit_rate=0.99,
        spec_accept_rate=0.72,
        loadavg_pre=(2.5, 3.0, 3.5),
        loadavg_post=(4.0, 3.5, 3.2),
        engine_runtime_config={"enable_prefix_caching": True, "max_num_seqs": 256},
        flag_did_not_engage=True,
    )


# ---------------------------------------------------------------------------
# GenerationObservability schema contracts
# ---------------------------------------------------------------------------


class TestGenerationObservabilitySchema:
    """Schema contracts: defaults, round-trip, extra-fields policy."""

    def test_all_measurement_fields_optional(self) -> None:
        """Producers populate what they have; missing fields default to None/empty.

        Important because the four primitives each have degraded-mode
        paths that return None — the schema must accept that.
        """
        event = GenerationObservability()
        assert event.peak_vram_gb is None
        assert event.kv_cache_usage_perc is None
        assert event.loadavg_pre is None
        assert event.engine_runtime_config == {}
        assert event.flag_did_not_engage is False

    def test_json_round_trip_lossless(self, populated_event: GenerationObservability) -> None:
        """Consumers depend on JSON round-trip — structured logs serialize to JSON, deserializers must rebuild the same event."""
        rt = GenerationObservability.model_validate_json(populated_event.model_dump_json())
        assert rt == populated_event

    def test_extra_fields_forbidden(self) -> None:
        """``extra='forbid'`` is the contract that forces schema updates when fields are added — silent drift would break consumers."""
        with pytest.raises(ValidationError):
            GenerationObservability.model_validate({"peak_vram_gb": 1.0, "unknown_field": "x"})


# ---------------------------------------------------------------------------
# to_wandb_payload flattening contracts
# ---------------------------------------------------------------------------


class TestToWandbPayload:
    """Wandb-side flattening: namespace, None-drop, tuple unpack, dict flatten."""

    def test_keys_are_namespaced_under_prefix(self, populated_event: GenerationObservability) -> None:
        """Custom prefix prevents collision with other wandb metrics in the same run."""
        payload = populated_event.to_wandb_payload(prefix="custom")
        assert payload  # non-empty
        assert all(k.startswith("custom/") for k in payload)

    def test_none_values_dropped_explicitly(self) -> None:
        """Wandb drops None silently anyway; we drop them explicitly so the payload's shape is predictable."""
        event = GenerationObservability(peak_vram_gb=10.0, kv_cache_usage_perc=None)
        payload = event.to_wandb_payload()
        assert "vllm_gen/peak_vram_gb" in payload
        assert "vllm_gen/kv_cache_usage_perc" not in payload

    def test_loadavg_tuples_unpacked_to_per_horizon_scalars(self) -> None:
        """3-tuples become 3 scalars — wandb plots scalars cleanly, tuples become opaque blobs."""
        event = GenerationObservability(loadavg_pre=(1.0, 2.0, 3.0))
        payload = event.to_wandb_payload()
        assert payload["vllm_gen/loadavg_pre_1m"] == 1.0
        assert payload["vllm_gen/loadavg_pre_5m"] == 2.0
        assert payload["vllm_gen/loadavg_pre_15m"] == 3.0
        # Raw tuple key is gone — only the unpacked scalars are exposed.
        assert "vllm_gen/loadavg_pre" not in payload

    def test_engine_config_flattened_under_engine_runtime_namespace(self) -> None:
        """Engine-config dict gets flattened to scalars under ``engine_runtime/``."""
        event = GenerationObservability(engine_runtime_config={"a": 1, "b": True, "c": None})
        payload = event.to_wandb_payload()
        assert payload["vllm_gen/engine_runtime/a"] == 1
        assert payload["vllm_gen/engine_runtime/b"] is True
        # None values are dropped, symmetric with the scalar-field handling.
        assert "vllm_gen/engine_runtime/c" not in payload


# ---------------------------------------------------------------------------
# flag_engagement_mismatches contracts
# ---------------------------------------------------------------------------


class TestFlagEngagementMismatches:
    """The 'only check explicitly-set fields against probed values' contract."""

    @pytest.mark.parametrize(
        ("intended", "actual", "expect_mismatch"),
        [
            # Clean match — no mismatch.
            ({"enable_prefix_caching": True}, {"enable_prefix_caching": True}, False),
            # Disagreement — mismatch.
            ({"enable_prefix_caching": False}, {"enable_prefix_caching": True}, True),
            # Caller didn't explicitly set the field → skip (no reference value).
            ({"enable_prefix_caching": None}, {"enable_prefix_caching": True}, False),
            # Probe didn't return the field → skip (probe is best-effort).
            ({"enable_prefix_caching": True}, {}, False),
            # Both empty — vacuously clean.
            ({}, {}, False),
        ],
    )
    def test_only_checks_when_both_sides_have_a_value(
        self, intended: dict[str, Any], actual: dict[str, Any], expect_mismatch: bool
    ) -> None:
        mismatches = flag_engagement_mismatches(intended, actual)
        assert bool(mismatches) is expect_mismatch
        if expect_mismatch:
            # Description must include the field name so operators can act.
            assert any("enable_prefix_caching" in m for m in mismatches)


# ---------------------------------------------------------------------------
# read_loadavg contracts
# ---------------------------------------------------------------------------


class TestReadLoadavg:
    def test_shape_when_available(self) -> None:
        """Returns a 3-tuple of non-negative floats on Linux."""
        result = read_loadavg()
        if result is None:
            pytest.skip("/proc/loadavg unavailable on this host")
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(x, float) and x >= 0.0 for x in result)

    def test_returns_none_on_read_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OSError on the file open is the canonical 'unavailable' signal — degraded-mode contract."""
        monkeypatch.setattr("builtins.open", MagicMock(side_effect=OSError("not available")))
        assert read_loadavg() is None


# ---------------------------------------------------------------------------
# probe_engine_runtime_config contracts
# ---------------------------------------------------------------------------


class TestProbeEngineRuntimeConfig:
    @pytest.mark.parametrize("bad_input", [None, object(), "not an llm"])
    def test_empty_dict_on_any_failure(self, bad_input: Any) -> None:
        """Best-effort probe: anything not vLLM-shaped returns ``{}`` rather than raising."""
        assert probe_engine_runtime_config(bad_input) == {}

    def test_extracts_known_fields_from_vllm_config(self) -> None:
        """When the engine exposes vllm_config, the documented fields land in the result.

        Uses ``SimpleNamespace`` instead of ``MagicMock`` because the
        probe's "modern-name first, legacy fallback" logic relies on
        ``getattr(obj, name, None)`` returning ``None`` for missing
        attributes — MagicMock auto-creates them, which doesn't match
        real vLLM behavior.
        """
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    scheduler_config=SimpleNamespace(
                        max_num_seqs=256,
                        max_num_batched_tokens=8192,
                        chunked_prefill_enabled=True,
                    ),
                    cache_config=SimpleNamespace(
                        enable_prefix_caching=True,
                        cache_dtype="auto",  # legacy attribute name
                    ),
                    speculative_config=None,
                ),
            ),
        )
        out = probe_engine_runtime_config(llm)
        assert out["max_num_seqs"] == 256
        assert out["max_num_batched_tokens"] == 8192
        assert out["enable_chunked_prefill"] is True
        assert out["enable_prefix_caching"] is True
        assert out["kv_cache_dtype"] == "auto"  # legacy name surfaced under the modern key

    def test_extracts_speculative_method(self) -> None:
        """``speculative_config.method`` surfaces under ``speculative_method``."""
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    scheduler_config=None,
                    cache_config=None,
                    speculative_config=SimpleNamespace(method="eagle"),
                ),
            ),
        )
        out = probe_engine_runtime_config(llm)
        assert out["speculative_method"] == "eagle"

    def test_one_bad_section_does_not_sink_the_whole_probe(self) -> None:
        """A section attribute that raises degrades that field only, not the result.

        Field-granular degradation is the contract that replaced the old
        whole-body try/except.
        """

        class _Exploding:
            @property
            def enable_prefix_caching(self) -> bool:
                raise RuntimeError("boom")

        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    scheduler_config=SimpleNamespace(max_num_seqs=128),
                    cache_config=_Exploding(),
                    speculative_config=None,
                ),
            ),
        )
        out = probe_engine_runtime_config(llm)
        # The healthy field still lands; the exploding one is simply absent.
        assert out["max_num_seqs"] == 128
        assert "enable_prefix_caching" not in out


class TestEngineConfigCheckedFields:
    """``ENGINE_CONFIG_CHECKED_FIELDS`` is derived from the probe table."""

    def test_checked_fields_match_expected_set(self) -> None:
        """The engagement-checked fields are exactly the probe's ``checked=True`` keys."""
        assert set(ENGINE_CONFIG_CHECKED_FIELDS) == {
            "max_num_seqs",
            "max_num_batched_tokens",
            "enable_chunked_prefill",
            "enable_prefix_caching",
            "kv_cache_dtype",
        }

    def test_speculative_method_probed_but_not_checked(self) -> None:
        """``speculative_method`` is observability-only — excluded from the check."""
        assert "speculative_method" not in ENGINE_CONFIG_CHECKED_FIELDS


# ---------------------------------------------------------------------------
# read_vllm_runtime_metrics contracts
# ---------------------------------------------------------------------------


_STABLE_METRIC_KEYS = {"kv_cache_usage_perc", "prefix_cache_hit_rate", "spec_accept_rate"}


class TestReadVllmRuntimeMetrics:
    def test_stable_keys_when_engine_returns_nothing(self, mock_llm_no_metrics: Any) -> None:
        """Callers depend on the dict having these three keys regardless of which metrics were exposed."""
        result = read_vllm_runtime_metrics(mock_llm_no_metrics)
        assert set(result.keys()) == _STABLE_METRIC_KEYS
        assert all(v is None for v in result.values())

    def test_pulls_each_metric_into_its_slot(self, mock_llm_full_metrics: Any) -> None:
        """Every documented metric lands in its named slot, with derivations applied for the rates."""
        result = read_vllm_runtime_metrics(mock_llm_full_metrics)
        assert result["kv_cache_usage_perc"] == 0.42
        assert result["prefix_cache_hit_rate"] == pytest.approx(0.8)  # 80/100
        assert result["spec_accept_rate"] == pytest.approx(0.72)  # 720/1000

    def test_stable_keys_when_get_metrics_raises(self) -> None:
        """Exception in ``get_metrics()`` → degraded-mode dict with Nones, not propagation."""
        llm = MagicMock()
        llm.get_metrics.side_effect = RuntimeError("engine dead")
        result = read_vllm_runtime_metrics(llm)
        assert set(result.keys()) == _STABLE_METRIC_KEYS
        assert all(v is None for v in result.values())

    def test_zero_denominator_yields_none_not_divide_by_zero(self) -> None:
        """``num_draft_tokens == 0`` is "speculation registered but no drafts proposed" — return None, don't divide."""
        llm = MagicMock()
        llm.get_metrics.return_value = [
            _metric(METRIC_SPEC_NUM_ACCEPTED_TOKENS, 0),
            _metric(METRIC_SPEC_NUM_DRAFT_TOKENS, 0),
        ]
        result = read_vllm_runtime_metrics(llm)
        assert result["spec_accept_rate"] is None


# ---------------------------------------------------------------------------
# NvmlPeakSampler contracts
# ---------------------------------------------------------------------------


class TestNvmlPeakSampler:
    def test_context_manager_protocol(self) -> None:
        """``__enter__`` returns self; ``__exit__`` cleans up without raising."""
        sampler = NvmlPeakSampler()
        result = sampler.__enter__()
        try:
            assert result is sampler
        finally:
            sampler.__exit__(None, None, None)

    def test_peak_gb_typed_correctly(self) -> None:
        """``peak_gb`` is ``float | None`` — the only two shapes consumers handle."""
        with NvmlPeakSampler(interval_seconds=0.05) as sampler:
            time.sleep(0.15)
        assert sampler.peak_gb is None or isinstance(sampler.peak_gb, float)

    def test_pynvml_unavailable_yields_none_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ImportError on pynvml → sampler enters degraded mode, peak_gb is None."""
        # Force the import to fail by patching ``__import__`` for the pynvml name only.
        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def _fail_pynvml(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "pynvml":
                raise ImportError("simulated pynvml unavailable")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fail_pynvml)
        with NvmlPeakSampler() as sampler:
            pass
        assert sampler.peak_gb is None


# ---------------------------------------------------------------------------
# log_generation_observability contracts
# ---------------------------------------------------------------------------


class TestLogGenerationObservability:
    """The wandb-side helper's degraded-mode + best-effort contracts."""

    def test_no_op_when_no_active_wandb_run(self, populated_event: GenerationObservability) -> None:
        """``wandb.run is None`` → return without raising or logging."""
        from nemo_safe_synthesizer.cli import wandb_setup

        with patch.object(wandb_setup.wandb, "run", None):
            # Should not raise; nothing to assert beyond "doesn't blow up".
            wandb_setup.log_generation_observability(populated_event)

    def test_calls_wandb_log_when_run_is_active(self, populated_event: GenerationObservability) -> None:
        """When a run is active, the flattened payload is passed to ``wandb.log``."""
        from nemo_safe_synthesizer.cli import wandb_setup

        fake_run = MagicMock()
        with patch.object(wandb_setup.wandb, "run", fake_run), patch.object(wandb_setup.wandb, "log") as mock_log:
            wandb_setup.log_generation_observability(populated_event)
            mock_log.assert_called_once()
            (call_args,) = mock_log.call_args.args
            # The payload contains exactly the keys from to_wandb_payload (namespaced + flattened).
            assert call_args == populated_event.to_wandb_payload()

    def test_wandb_log_exception_swallowed(self, populated_event: GenerationObservability) -> None:
        """A wandb failure must not break generation — best-effort emission."""
        from nemo_safe_synthesizer.cli import wandb_setup

        fake_run = MagicMock()
        with (
            patch.object(wandb_setup.wandb, "run", fake_run),
            patch.object(wandb_setup.wandb, "log", side_effect=RuntimeError("wandb down")),
        ):
            # Should not raise.
            wandb_setup.log_generation_observability(populated_event)
