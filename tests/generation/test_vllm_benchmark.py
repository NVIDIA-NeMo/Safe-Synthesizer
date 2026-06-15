# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for benchmark harness data models, helpers, presets, and runner dispatch.

Scope: consumer-facing contracts. Runner tests patch vLLM imports and
engine construction so they exercise dispatch/metric semantics without
spinning up a real vLLM engine.
"""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

import nemo_safe_synthesizer.generation.processors as processors_mod
import nemo_safe_synthesizer.generation.vllm_benchmark as benchmark_mod
from nemo_safe_synthesizer.generation.vllm_benchmark import (
    BenchmarkCandidate,
    BenchmarkCorpus,
    BenchmarkEngineConfig,
    BenchmarkOutput,
    BenchmarkPrompt,
    CandidateMetrics,
    SubprocessRunResult,
    TraceHeader,
    _build_vllm_kwargs,
    _extract_ttft_ms,
    _merge_sampling_kwargs,
    _parse_error_class,
    _percentile,
    _truncate_stderr,
    run_benchmark,
)
from nemo_safe_synthesizer.generation.vllm_benchmark_presets import (
    DEFAULT_BENCHMARK_SEED,
    DEFAULT_BRACKETED_AB_N,
    attention_backend_sweep,
    baseline,
    bracketed_ab,
    bracketed_ab_spec_ngram,
    default_matrix,
)
from nemo_safe_synthesizer.generation.vllm_observability import GenerationObservability

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def header() -> TraceHeader:
    return TraceHeader(
        run_id="test-run",
        pretrained_model="mistralai/Mistral-7B-Instruct-v0.3",
        dataset_schema={"col": "string"},
        engine_parameters={"max_lora_rank": 32, "structured_generation_backend": "outlines"},
    )


@pytest.fixture
def empty_base() -> BenchmarkEngineConfig:
    return BenchmarkEngineConfig()


class _FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeLoRARequest:
    def __init__(self, *args: Any) -> None:
        self.args = args


class _NoopPeakSampler:
    peak_gb: float | None = None

    def __enter__(self) -> _NoopPeakSampler:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _ReadyEvent:
    def wait(self) -> None:
        return None


class _FakeCompletion:
    def __init__(self, text: str, token_ids: list[int], finish_reason: str = "stop") -> None:
        self.text = text
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class _FakeRequestOutput:
    def __init__(self, outputs: list[_FakeCompletion], ttft_s: float = 0.01) -> None:
        self.outputs = outputs
        self.metrics = SimpleNamespace(first_token_latency=ttft_s)


class _FakeLLM:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(self, *, prompts: list[str], sampling_params: _FakeSamplingParams, lora_request: Any) -> list[Any]:
        self.calls.append(
            {
                "prompts": list(prompts),
                "sampling_params": sampling_params,
                "lora_request": lora_request,
            },
        )
        if len(prompts) == 1 and sampling_params.kwargs.get("n", 1) > 1:
            return [
                _FakeRequestOutput(
                    [
                        _FakeCompletion(text=f'{{"col": "v{i}"}}', token_ids=[i, i + 100])
                        for i in range(sampling_params.kwargs["n"])
                    ],
                ),
            ]
        return [
            _FakeRequestOutput([_FakeCompletion(text=f'{{"col": "v{i}"}}', token_ids=[i])])
            for i, _prompt in enumerate(prompts)
        ]

    def get_metrics(self) -> list[Any]:
        return []


class _FakeProcessor:
    def __init__(self, calls: list[tuple[int, str]]) -> None:
        self._calls = calls

    def __call__(self, prompt_number: int, text: str) -> Any:
        self._calls.append((prompt_number, text))
        return SimpleNamespace(valid_records=[{"text": text}], invalid_records=[])


def _install_fake_vllm_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []
    lora_mod = types.ModuleType("vllm.lora")
    lora_mod.__path__ = []
    request_mod = types.ModuleType("vllm.lora.request")
    setattr(request_mod, "LoRARequest", _FakeLoRARequest)
    sampling_mod = types.ModuleType("vllm.sampling_params")
    setattr(sampling_mod, "SamplingParams", _FakeSamplingParams)
    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora_mod)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request_mod)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_mod)


def _install_runner_fakes(
    monkeypatch: pytest.MonkeyPatch,
    llm: _FakeLLM,
    *,
    init_seconds: float = 0.0,
    processor_calls: list[tuple[int, str]] | None = None,
) -> None:
    _install_fake_vllm_modules(monkeypatch)
    calls = processor_calls if processor_calls is not None else []
    monkeypatch.setattr(processors_mod, "TabularDataProcessor", lambda *args, **kwargs: _FakeProcessor(calls))
    monkeypatch.setattr(benchmark_mod, "NvmlPeakSampler", _NoopPeakSampler)
    monkeypatch.setattr(benchmark_mod, "read_loadavg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(benchmark_mod, "probe_engine_runtime_config", lambda llm: {})
    monkeypatch.setattr(benchmark_mod, "flag_engagement_mismatches", lambda intended, actual: [])
    monkeypatch.setattr(
        benchmark_mod,
        "read_vllm_runtime_metrics",
        lambda llm: {"kv_cache_usage_perc": None, "prefix_cache_hit_rate": None, "spec_accept_rate": None},
    )

    def fake_build_engine_async(header: TraceHeader, engine_config: BenchmarkEngineConfig) -> tuple[Any, Any, Any]:
        result = SimpleNamespace(llm=llm, exception=None, init_seconds=init_seconds)
        return SimpleNamespace(), _ReadyEvent(), result

    monkeypatch.setattr(benchmark_mod, "_build_engine_async", fake_build_engine_async)


def _benchmark_corpus(prompts: list[str]) -> BenchmarkCorpus:
    return BenchmarkCorpus(
        header=TraceHeader(run_id="r", pretrained_model="m", dataset_schema={"col": "string"}),
        prompts=[
            BenchmarkPrompt(
                row_index=i,
                prompt=prompt,
                original_sampling_params={"temperature": 0.0, "max_tokens": 8},
            )
            for i, prompt in enumerate(prompts)
        ],
    )


# ---------------------------------------------------------------------------
# Data model contracts
# ---------------------------------------------------------------------------


class TestSchemaContracts:
    def test_candidate_metrics_composes_observability(self) -> None:
        """``CandidateMetrics`` embeds ``GenerationObservability`` rather than re-declaring its fields."""
        m = CandidateMetrics(
            name="t",
            raw_tok_s=1.0,
            acceptance_rate=0.9,
            effective_tok_s=0.9,
            ttft_p50_ms=0.0,
            ttft_p99_ms=0.0,
            prompts_attempted=10,
            prompts_accepted=9,
            total_output_tokens=100,
            total_wall_seconds=1.0,
            observability=GenerationObservability(peak_vram_gb=64.5),
        )
        assert m.observability.peak_vram_gb == 64.5

    def test_benchmark_output_roundtrips_through_json(self) -> None:
        """Consumers depend on the JSON serialization being lossless."""
        out = BenchmarkOutput(
            corpus_run_id="r1",
            corpus_size=10,
            candidates=[
                CandidateMetrics(
                    name="t",
                    raw_tok_s=1.0,
                    acceptance_rate=0.9,
                    effective_tok_s=0.9,
                    ttft_p50_ms=0.0,
                    ttft_p99_ms=0.0,
                    prompts_attempted=10,
                    prompts_accepted=9,
                    total_output_tokens=100,
                    total_wall_seconds=1.0,
                    observability=GenerationObservability(peak_vram_gb=64.5, loadavg_pre=(1.0, 2.0, 3.0)),
                ),
            ],
        )
        rt = BenchmarkOutput.model_validate_json(out.model_dump_json())
        assert rt == out
        # Specifically: the nested observability survives the round-trip.
        assert rt.candidates[0].observability.peak_vram_gb == 64.5
        assert rt.candidates[0].observability.loadavg_pre == (1.0, 2.0, 3.0)

    def test_engine_config_tolerates_unknown_kwargs(self) -> None:
        """``BenchmarkEngineConfig.extra='ignore'`` so the CLI can validate raw header dicts."""
        cfg = BenchmarkEngineConfig.model_validate({"attention_backend": "FLASHINFER", "unknown_kwarg": 42})
        assert cfg.attention_backend == "FLASHINFER"

    def test_candidate_extra_fields_forbidden(self) -> None:
        """``BenchmarkCandidate.extra='forbid'`` means adding a field must update the schema."""
        with pytest.raises(ValidationError):
            BenchmarkCandidate.model_validate({"name": "t", "unknown_field": 42})


# ---------------------------------------------------------------------------
# Corpus loader
# ---------------------------------------------------------------------------


class TestBenchmarkCorpus:
    def test_loads_jsonl_with_header_and_records(self, tmp_path: Any) -> None:
        path = tmp_path / "trace.jsonl"
        lines = [
            json.dumps({"kind": "header", "run_id": "r", "pretrained_model": "m", "dataset_schema": {}}),
            json.dumps({"kind": "record", "row_index": 0, "prompt": "p0", "sampling_params": {"temperature": 0.7}}),
            json.dumps({"kind": "record", "row_index": 1, "prompt": "p1", "sampling_params": {"temperature": 0.7}}),
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        corpus = BenchmarkCorpus.from_trace_jsonl(path)
        assert corpus.header.run_id == "r"
        assert len(corpus.prompts) == 2
        assert corpus.prompts[0].original_sampling_params["temperature"] == 0.7

    def test_rejects_missing_header(self, tmp_path: Any) -> None:
        path = tmp_path / "no_header.jsonl"
        path.write_text(json.dumps({"kind": "record", "row_index": 0, "prompt": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="record on line 1 before any header"):
            BenchmarkCorpus.from_trace_jsonl(path)


# ---------------------------------------------------------------------------
# Helper contracts
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_percentile_linear_interp(self) -> None:
        assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
        assert _percentile([1.0], 50) == 1.0
        assert _percentile([], 50) == 0.0  # empty returns 0, not raises

    def test_merge_sampling_strips_non_sampling_fields(self) -> None:
        """``structured_outputs`` is capture-time metadata, not a SamplingParams kwarg."""
        merged = _merge_sampling_kwargs(
            {"temperature": 0.7, "top_p": 0.9, "structured_outputs": "json"},
            {"seed": 42},
        )
        assert merged == {"temperature": 0.7, "top_p": 0.9, "seed": 42}

    def test_merge_overrides_win_on_conflict(self) -> None:
        merged = _merge_sampling_kwargs({"temperature": 0.7}, {"temperature": 0.0})
        assert merged["temperature"] == 0.0

    @pytest.mark.parametrize(
        ("metrics_obj", "expected"),
        [
            (None, None),
            (type("M", (), {"first_token_latency": 0.12})(), 120.0),
            (type("M", (), {"first_token_latency": None, "first_token_time": 1.0, "arrival_time": 0.5})(), 500.0),
            (type("M", (), {"first_token_latency": None, "first_token_time": None, "arrival_time": 0.5})(), None),
        ],
    )
    def test_extract_ttft_ms(self, metrics_obj: Any, expected: float | None) -> None:
        """Tries modern ``first_token_latency`` first, falls back to ``first_token_time - arrival_time``."""
        output = type("Out", (), {"metrics": metrics_obj})()
        assert _extract_ttft_ms(output) == expected

    def test_truncate_stderr_keeps_tail(self) -> None:
        long = "x" * 600
        truncated = _truncate_stderr(long, limit=100)
        assert len(truncated) <= 100
        assert truncated.endswith("xxxxx")  # tail-preserving

    def test_parse_error_class_finds_exception_name(self) -> None:
        assert _parse_error_class("Traceback (most recent call last):\n...\nValueError: bad") == "ValueError"
        assert _parse_error_class("") == "Error"


# ---------------------------------------------------------------------------
# Engine kwargs builder
# ---------------------------------------------------------------------------


class TestBuildVllmKwargs:
    def test_overlays_engine_config_on_header(self, header: TraceHeader) -> None:
        """Candidate engine_config overlays the header's engine_parameters."""
        cfg = BenchmarkEngineConfig(attention_backend="FLASHINFER", max_model_len=4096)
        kwargs = _build_vllm_kwargs(header, cfg)
        assert kwargs["model"] == "mistralai/Mistral-7B-Instruct-v0.3"
        assert kwargs["max_lora_rank"] == 32  # from header
        assert kwargs["max_model_len"] == 4096  # from cfg

    def test_translates_attention_backend_to_attention_config(
        self, header: TraceHeader, empty_base: BenchmarkEngineConfig
    ) -> None:
        """VLLM's public API takes an ``attention_config`` dict, not a bare backend string."""
        cfg = empty_base.model_copy(update={"attention_backend": "FLASHINFER"})
        kwargs = _build_vllm_kwargs(header, cfg)
        assert kwargs["attention_config"] == {"backend": "FLASHINFER"}
        assert "attention_backend" not in kwargs  # translated, not forwarded

    def test_drops_none_valued_overrides(self, header: TraceHeader, empty_base: BenchmarkEngineConfig) -> None:
        """Unset candidate fields stay out of kwargs so vLLM uses its defaults."""
        kwargs = _build_vllm_kwargs(header, empty_base)
        assert "enable_prefix_caching" not in kwargs
        assert "max_num_seqs" not in kwargs

    def test_auto_attention_backend_is_treated_as_unset(
        self, header: TraceHeader, empty_base: BenchmarkEngineConfig
    ) -> None:
        """``attention_backend='auto'`` means no attention_config kwarg should appear."""
        cfg = empty_base.model_copy(update={"attention_backend": "auto"})
        kwargs = _build_vllm_kwargs(header, cfg)
        assert "attention_config" not in kwargs


# ---------------------------------------------------------------------------
# Runner contracts
# ---------------------------------------------------------------------------


class TestRunBenchmark:
    def test_n_fanout_dispatches_one_prompt_with_one_completion_per_corpus_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llm = _FakeLLM()
        processor_calls: list[tuple[int, str]] = []
        _install_runner_fakes(monkeypatch, llm, processor_calls=processor_calls)

        metrics = run_benchmark(
            BenchmarkCandidate(name="fanout", batch_dispatch_mode="n_fanout"),
            _benchmark_corpus(["same", "same", "same"]),
        )

        assert llm.calls[0]["prompts"] == ["same"]
        assert llm.calls[0]["sampling_params"].kwargs["n"] == 3
        assert processor_calls == [
            (0, '{"col": "v0"}'),
            (1, '{"col": "v1"}'),
            (2, '{"col": "v2"}'),
        ]
        assert metrics.prompts_attempted == 3
        assert metrics.prompts_accepted == 3
        assert metrics.total_output_tokens == 6
        assert metrics.finish_reason_distribution == {"stop": 3}

    def test_n_fanout_rejects_non_identical_prompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = _FakeLLM()
        _install_runner_fakes(monkeypatch, llm)

        with pytest.raises(ValueError, match="requires every corpus prompt to be identical"):
            run_benchmark(
                BenchmarkCandidate(name="fanout", batch_dispatch_mode="n_fanout"),
                _benchmark_corpus(["first", "second"]),
            )

    def test_overlap_savings_are_capped_by_actual_engine_init_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        llm = _FakeLLM()
        _install_runner_fakes(monkeypatch, llm, init_seconds=1.25)
        monkeypatch.setattr(benchmark_mod.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(benchmark_mod.time, "monotonic", lambda: 100.0)

        metrics = run_benchmark(
            BenchmarkCandidate(name="baseline"),
            _benchmark_corpus(["p0"]),
            simulate_training_overlap_seconds=10.0,
        )

        assert metrics.startup_seconds == 0.0
        assert metrics.startup_overlap_savings_seconds == 1.25


# ---------------------------------------------------------------------------
# Preset contracts
# ---------------------------------------------------------------------------


class TestPresets:
    def test_baseline_pins_seed(self, empty_base: BenchmarkEngineConfig) -> None:
        cand = baseline(empty_base)
        assert cand.sampling_overrides == {"seed": DEFAULT_BENCHMARK_SEED}

    def test_default_matrix_dedupes(self, empty_base: BenchmarkEngineConfig) -> None:
        """Concatenated sweeps that produce identical (engine, sampling) tuples are collapsed."""
        cands = default_matrix(empty_base)
        keys = {(c.engine_config.model_dump_json(), json.dumps(c.sampling_overrides, sort_keys=True)) for c in cands}
        assert len(keys) == len(cands)

    def test_attention_backend_sweep_covers_known_backends(self, empty_base: BenchmarkEngineConfig) -> None:
        cands = attention_backend_sweep(empty_base)
        backends = {c.engine_config.attention_backend for c in cands}
        assert {"FLASHINFER", "FLASH_ATTN", "TRITON_ATTN"}.issubset(backends)


class TestBracketedAb:
    def test_emits_2n_cells_interleaved(self, empty_base: BenchmarkEngineConfig) -> None:
        """N baselines + N candidate runs, interleaved by bracket_position."""
        cells = bracketed_ab(
            empty_base,
            candidate_engine_overrides={"enable_prefix_caching": True},
            condition_label="prefix_on",
            n_samples_per_condition=3,
        )
        assert len(cells) == 6
        # Even positions are baseline; odd positions are the candidate.
        for i, cell in enumerate(cells):
            expected_label = "baseline" if i % 2 == 0 else "prefix_on"
            assert cell.condition_label == expected_label
            assert cell.bracket_position == i

    def test_spec_ngram_wrapper_applies_speculative_config(self, empty_base: BenchmarkEngineConfig) -> None:
        cells = bracketed_ab_spec_ngram(empty_base)
        # Candidate runs have speculative_config; baselines don't.
        candidates = [c for c in cells if c.condition_label == "spec_ngram"]
        baselines = [c for c in cells if c.condition_label == "baseline"]
        assert len(candidates) == DEFAULT_BRACKETED_AB_N
        assert len(baselines) == DEFAULT_BRACKETED_AB_N
        for c in candidates:
            assert c.engine_config.speculative_config is not None
            assert c.engine_config.speculative_config["method"] == "ngram"
        for b in baselines:
            assert b.engine_config.speculative_config is None

    def test_all_cells_seed_pinned(self, empty_base: BenchmarkEngineConfig) -> None:
        cells = bracketed_ab_spec_ngram(empty_base)
        for c in cells:
            assert c.sampling_overrides.get("seed") == DEFAULT_BENCHMARK_SEED


# ---------------------------------------------------------------------------
# SubprocessRunResult
# ---------------------------------------------------------------------------


class TestSubprocessRunResult:
    def test_success_shape(self) -> None:
        m = CandidateMetrics(
            name="t",
            raw_tok_s=1.0,
            acceptance_rate=0.9,
            effective_tok_s=0.9,
            ttft_p50_ms=0.0,
            ttft_p99_ms=0.0,
            prompts_attempted=10,
            prompts_accepted=9,
            total_output_tokens=100,
            total_wall_seconds=1.0,
        )
        r = SubprocessRunResult(metrics=m)
        assert r.metrics is not None and r.error is None

    def test_failure_shape(self) -> None:
        r = SubprocessRunResult(error="exit 1", error_class="RuntimeError")
        assert r.metrics is None and r.error_class == "RuntimeError"
