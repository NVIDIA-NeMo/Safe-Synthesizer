# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from rich.console import Console
from transformers import PretrainedConfig, PreTrainedTokenizerBase

from nemo_safe_synthesizer.config.data import DataParameters
from nemo_safe_synthesizer.config.parameters import SafeSynthesizerParameters
from nemo_safe_synthesizer.config.time_series import TimeSeriesParameters
from nemo_safe_synthesizer.config.training import TrainingHyperparams
from nemo_safe_synthesizer.defaults import DEFAULT_MAX_SEQ_LENGTH, PSEUDO_GROUP_COLUMN
from nemo_safe_synthesizer.llm.metadata import ModelMetadata
from nemo_safe_synthesizer.llm.utils import ModelRef
from nemo_safe_synthesizer.preflight import (
    AdvisoryCheck,
    ConfigCheck,
    ConstantColumnCheck,
    CUDAAvailabilityCheck,
    DatasetSizeCheck,
    GroupbyColumnCheck,
    HFModelAvailabilityCheck,
    InferenceModelCheck,
    OrderbyColumnCheck,
    OversamplingCheck,
    PreflightContext,
    PreflightIssue,
    PreflightRegistry,
    PreflightReport,
    PreflightStage,
    PseudoColumnCheck,
    SmallDatasetCheck,
    TimeSeriesDataShapeCheck,
    TimestampColumnCheck,
    TokenBudgetCheck,
    VRAMHeadroomCheck,
    get_registry,
    run_preflight,
)
from nemo_safe_synthesizer.tooling import PreflightRenderContext, render_preflight_report

from .conftest import make_ctx


def _issue_by_code(issues: list[PreflightIssue], code: str) -> PreflightIssue:
    return next(issue for issue in issues if issue.code == code)


class _PseudoColumnSensitiveTokenizer(PreTrainedTokenizerBase):
    """Tokenizer that makes pseudo-column leakage visible in budget tests."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return []

    def __call__(self, texts: list[str], *, add_special_tokens: bool) -> dict[str, list[list[int]]]:
        assert add_special_tokens is False
        return {"input_ids": [[0] * (100 if PSEUDO_GROUP_COLUMN in text else 1) for text in texts]}


# ---------------------------------------------------------------------------
# Per-check tests
#
# Each production ``PreflightCheck`` gets its own ``TestCheckX`` class. The
# class grouping makes it trivial to run a single check's tests with
# ``pytest -k TestCheckX`` and makes the file scannable by check name.
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCUDAAvailabilityCheck:
    def test_no_gpu_emits_error(self, default_config):
        with patch("torch.cuda.is_available", return_value=False):
            issues = CUDAAvailabilityCheck().run(make_ctx(config=default_config))
        assert any(i.code == "no_gpu" and i.severity == "error" for i in issues)

    def test_with_gpu_is_silent(self, default_config):
        with patch("torch.cuda.is_available", return_value=True):
            issues = CUDAAvailabilityCheck().run(make_ctx(config=default_config))
        assert not any(i.code == "no_gpu" for i in issues)

    def test_missing_torch_emits_torch_missing_not_no_gpu(self, default_config):
        # A system without PyTorch is a different failure mode from one
        # with PyTorch but no CUDA device; the two must have distinct codes
        # so docs/users can disambiguate.
        with patch(
            "nemo_safe_synthesizer.preflight.helpers.importlib.import_module",
            side_effect=ImportError("No module named 'torch'"),
        ):
            issues = CUDAAvailabilityCheck().run(make_ctx(config=default_config))
        codes = {i.code for i in issues}
        assert "torch_missing" in codes
        assert "no_gpu" not in codes


@pytest.mark.unit
class TestVRAMHeadroomCheck:
    @staticmethod
    def _autoconfig(**overrides) -> PretrainedConfig:
        defaults = dict(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            vocab_size=128_256,
            num_attention_heads=32,
            num_key_value_heads=8,
            head_dim=128,
            tie_word_embeddings=False,
        )
        defaults.update(overrides)
        return cast(PretrainedConfig, SimpleNamespace(**defaults))

    @staticmethod
    def _metadata(autoconfig=None):
        md = MagicMock(spec=ModelMetadata, autoconfig=autoconfig)
        md.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
        return md

    @staticmethod
    def _estimated_vram_gib(config: SafeSynthesizerParameters, autoconfig: PretrainedConfig) -> float:
        from nemo_safe_synthesizer.preflight.checks.environment import (
            estimate_base_model_params,
            estimate_training_vram_components,
        )

        result = estimate_base_model_params(autoconfig)
        assert result is not None
        n_params, _method = result
        comp = estimate_training_vram_components(
            n_params=n_params,
            training_cfg=config.training,
            batch_size=config.training.batch_size,
            seq_len=DEFAULT_MAX_SEQ_LENGTH,
            hidden_size=getattr(autoconfig, "hidden_size", None),
            num_hidden_layers=getattr(autoconfig, "num_hidden_layers", None),
        )
        return comp.total_gib

    def test_low_vram_warns(self, default_config):
        """A marginally oversized model emits a warning, not a hard error."""
        metadata = self._metadata(autoconfig=self._autoconfig())
        fake_props = MagicMock(total_memory=16 * 1024**3)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_safe_synthesizer.llm.utils.get_max_vram", return_value={0: 1.0}),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
        ):
            issues = VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))
        assert any(i.code == "low_vram" and i.severity == "warning" for i in issues)

    def test_ample_vram_is_silent(self, default_config):
        """Same config on an 80 GiB GPU must not warn (bf16 ~15 GiB base + overhead).

        The default config has quantize_model=False, so bytes_per_base_weight is
        2.0 (bf16) and the ~8B base weights estimate ~15 GiB; 80 GiB leaves ample
        headroom for both the low_vram warning and the hard-fail threshold.
        """
        metadata = self._metadata(autoconfig=self._autoconfig())
        fake_props = MagicMock(total_memory=80 * 1024**3)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_safe_synthesizer.llm.utils.get_max_vram", return_value={0: 1.0}),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
        ):
            issues = VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))
        assert not any(i.code == "low_vram" for i in issues)
        assert not any(i.code == "vram_exceeds_capacity" for i in issues)

    def test_absurd_batch_errors(self, default_config):
        """Per-device batch_size far too large must fail preflight."""
        default_config.training.batch_size = 100_000
        metadata = self._metadata(autoconfig=self._autoconfig())
        fake_props = MagicMock(total_memory=80 * 1024**3)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_safe_synthesizer.llm.utils.get_max_vram", return_value={0: 1.0}),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
        ):
            issues = VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))
        assert any(i.code == "vram_exceeds_capacity" and i.severity == "error" for i in issues)

    @pytest.mark.parametrize(
        ("ratio", "expected_code", "expected_severity"),
        [
            pytest.param(1.49, "low_vram", "warning", id="below-hard-fail-ratio"),
            pytest.param(1.50, "vram_exceeds_capacity", "error", id="at-hard-fail-ratio"),
        ],
    )
    def test_vram_hard_fail_ratio_boundary(self, default_config, ratio, expected_code, expected_severity):
        """The 1.5x hard-fail threshold is inclusive."""
        autoconfig = self._autoconfig()
        metadata = self._metadata(autoconfig=autoconfig)
        estimated_gib = self._estimated_vram_gib(default_config, autoconfig)
        fake_props = MagicMock(total_memory=100 * 1024**3)
        available_fraction = estimated_gib / (ratio * 100)
        opposite_code = "vram_exceeds_capacity" if expected_code == "low_vram" else "low_vram"
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_safe_synthesizer.llm.utils.get_max_vram", return_value={0: available_fraction}),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
        ):
            issues = VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))
        assert any(i.code == expected_code and i.severity == expected_severity for i in issues)
        assert not any(i.code == opposite_code for i in issues)

    @pytest.mark.parametrize("quantization_bits", (4, 8))
    def test_quantized_load_can_reduce_full_vram_check_below_warning(self, default_config, quantization_bits):
        """Runtime quantization settings affect the complete VRAM preflight path."""
        metadata = self._metadata(autoconfig=self._autoconfig())
        fake_props = MagicMock(total_memory=14 * 1024**3)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_safe_synthesizer.llm.utils.get_max_vram", return_value={0: 1.0}),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
        ):
            default_config.training.quantize_model = True
            default_config.training.quantization_bits = quantization_bits
            quantized_issues = VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))
            default_config.training.quantize_model = False
            unquantized_issues = VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))

        assert not any(i.code in {"low_vram", "vram_exceeds_capacity"} for i in quantized_issues)
        assert any(i.code in {"low_vram", "vram_exceeds_capacity"} for i in unquantized_issues)

    def test_gradient_accumulation_steps_does_not_fake_batch(self, default_config):
        """Larger gradient_accumulation_steps alone should not match absurd batch activation."""
        metadata = self._metadata(autoconfig=self._autoconfig())
        default_config.training.gradient_accumulation_steps = 10_000
        fake_props = MagicMock(total_memory=80 * 1024**3)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_safe_synthesizer.llm.utils.get_max_vram", return_value={0: 1.0}),
            patch("torch.cuda.get_device_properties", return_value=fake_props),
        ):
            issues = VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))
        assert not any(i.code == "low_vram" for i in issues)

    @pytest.mark.parametrize("fraction", (0.80, 0.50))
    def test_get_max_vram_receives_max_vram_fraction(self, default_config, fraction):
        default_config.training.max_vram_fraction = fraction
        metadata = self._metadata(autoconfig=self._autoconfig())
        fake_props = MagicMock(total_memory=80 * 1024**3)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_safe_synthesizer.llm.utils.get_max_vram", return_value={0: 1.0}) as mock_gmv,
            patch("torch.cuda.get_device_properties", return_value=fake_props),
        ):
            VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))
        mock_gmv.assert_called_once_with(max_vram_fraction=fraction)

    def test_stub_metadata_skips_silently(self, default_config):
        """Stubbed metadata (autoconfig=None) must skip without raising."""
        metadata = MagicMock(spec=ModelMetadata, autoconfig=None)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("nemo_safe_synthesizer.llm.utils.get_max_vram", return_value={0: 1.0}),
        ):
            issues = VRAMHeadroomCheck().run(make_ctx(config=default_config, metadata=metadata))
        assert issues == []

    def test_unquantized_load_uses_more_bytes_per_param_than_quantized_load(self, default_config):
        """bf16 base weights must estimate higher than a quantized base load."""
        from nemo_safe_synthesizer.preflight.checks.environment import (
            bytes_per_base_weight,
            estimate_base_model_params,
        )

        # SimpleNamespace can't be materialised by ``AutoModelForCausalLM``,
        # so this exercises the ``estimate_params_from_shape`` fallback.
        result = estimate_base_model_params(self._autoconfig())
        assert result is not None
        n, method = result
        assert method == "approximate"
        assert n > 7e9  # Llama-3-8B shape ~8B params
        default_config.training.quantize_model = False
        bpw_unquantized = bytes_per_base_weight(default_config.training)
        default_config.training.quantize_model = True
        default_config.training.quantization_bits = 4
        bpw_quantized = bytes_per_base_weight(default_config.training)
        assert bpw_unquantized > bpw_quantized

    def test_qlora_peft_label_without_quantize_model_uses_unquantized_base_weights(self, default_config):
        """Only quantize_model controls k-bit base-weight memory."""
        from nemo_safe_synthesizer.preflight.checks.environment import bytes_per_base_weight

        default_config.training.peft_implementation = "QLORA"
        default_config.training.quantize_model = False

        assert bytes_per_base_weight(default_config.training) == 2.0

    def test_vram_component_estimate_falls_back_without_activation_shape(self, default_config):
        from nemo_safe_synthesizer.preflight.checks.environment import estimate_training_vram_components

        comp = estimate_training_vram_components(
            n_params=1_000_000_000,
            training_cfg=default_config.training,
            batch_size=default_config.training.batch_size,
            seq_len=None,
            hidden_size=4096,
            num_hidden_layers=32,
        )

        assert comp.activation_gib is None
        assert comp.overhead_gib == pytest.approx(2.0)
        assert comp.total_gib == pytest.approx(comp.base_weights_gib + 2.0)

    @pytest.mark.parametrize(
        ("model_type", "fields", "expected_params_millions"),
        [
            # Gated-SwiGLU family: Llama-style architecture.
            pytest.param(
                "llama",
                dict(
                    hidden_size=128,
                    num_hidden_layers=2,
                    intermediate_size=256,
                    vocab_size=1024,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=32,
                    tie_word_embeddings=False,
                    max_position_embeddings=128,
                ),
                # embed+lm_head=2*128*1024=262144; per_layer attn 2*128*128+2*128*2*32=49152, mlp 3*128*256=98304
                # total = 262144 + 2*(49152+98304) = 557056 ≈ 0.557M
                0.56,
                id="llama-swiglu",
            ),
            # Non-gated squared-ReLU family: Nemotron. Proves the meta-tensor
            # path counts 2 MLP projections (up/down), not 3.
            pytest.param(
                "nemotron",
                dict(
                    hidden_size=128,
                    num_hidden_layers=2,
                    intermediate_size=256,
                    vocab_size=1024,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=32,
                    tie_word_embeddings=False,
                    max_position_embeddings=128,
                ),
                # Same except mlp=2*128*256=65536 → per_layer=114688; total=262144+2*114688=491520 ≈ 0.49M
                0.49,
                id="nemotron-squared-relu",
            ),
        ],
    )
    def test_meta_tensor_path_is_architecture_exact(self, model_type, fields, expected_params_millions):
        """The meta-tensor path picks the right module graph per architecture.

        Constructs a tiny real config via ``AutoConfig.for_model`` and verifies
        ``estimate_base_model_params`` returns a family-appropriate param
        count -- not the SwiGLU-shape formula result, which would over-count
        Nemotron's MLP by 50%.
        """
        from transformers import AutoConfig

        from nemo_safe_synthesizer.preflight.checks.environment import estimate_base_model_params

        config = AutoConfig.for_model(model_type, **fields)
        result = estimate_base_model_params(config)
        assert result is not None
        n, method = result
        assert method == "exact"
        # 10% tolerance absorbs per-layer biases, LayerNorm params, and RoPE
        # buffers that vary across families but are small relative to projections.
        assert abs(n / 1e6 - expected_params_millions) / expected_params_millions < 0.10, (
            f"{model_type}: expected ~{expected_params_millions:.2f}M params, got {n / 1e6:.3f}M"
        )


@pytest.mark.unit
class TestInferenceModelCheck:
    def test_empty_env_emits_key_warning(self, default_config):
        with patch.dict("os.environ", {}, clear=True):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        assert any(i.code == "inference_key_missing" and i.severity == "warning" for i in issues)

    def test_inference_key_present_is_silent(self, default_config):
        with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "test-key", "HF_TOKEN": "hf_xxx"}):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        assert not any(i.code == "inference_key_missing" for i in issues)

    def test_pii_disabled_skips_all_checks(self):
        config = SafeSynthesizerParameters(replace_pii=None)
        with patch.dict(
            "os.environ",
            {"NSS_INFERENCE_MODEL": "", "NSS_INFERENCE_ENDPOINT": "not-a-url"},
            clear=True,
        ):
            issues = InferenceModelCheck().run(make_ctx(config=config))
        assert issues == []

    def test_blank_model_emits_warning(self, default_config):
        with patch.dict(
            "os.environ",
            {"NSS_INFERENCE_KEY": "test-key", "NSS_INFERENCE_MODEL": "   "},
        ):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        assert any(i.code == "inference_model_blank" and i.severity == "warning" for i in issues)

    def test_unset_model_is_silent(self, default_config):
        with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "test-key"}, clear=True):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        assert not any(i.code == "inference_model_blank" for i in issues)

    def test_valid_model_is_silent(self, default_config):
        with patch.dict(
            "os.environ",
            {"NSS_INFERENCE_KEY": "test-key", "NSS_INFERENCE_MODEL": "qwen/qwen3-next-80b-a3b-instruct"},
        ):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        assert not any(i.code == "inference_model_blank" for i in issues)

    @pytest.mark.parametrize("endpoint", ["not-a-url", "ftp://example.com", "http://"])
    def test_invalid_endpoint_emits_error(self, default_config, endpoint):
        # An invalid endpoint cannot succeed, so it must fail preflight (error),
        # not merely warn -- otherwise --validate passes a config that fails on
        # the first classification request.
        with patch.dict(
            "os.environ",
            {"NSS_INFERENCE_KEY": "test-key", "NSS_INFERENCE_ENDPOINT": endpoint},
        ):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        assert any(i.code == "inference_endpoint_invalid" and i.severity == "error" for i in issues)

    def test_valid_endpoint_is_silent(self, default_config):
        with patch.dict(
            "os.environ",
            {"NSS_INFERENCE_KEY": "test-key", "NSS_INFERENCE_ENDPOINT": "https://integrate.api.nvidia.com/v1"},
        ):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        assert not any(i.code == "inference_endpoint_invalid" for i in issues)

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_endpoint_is_silent(self, default_config, blank):
        # A blank endpoint is treated as unset (falls back to the default base
        # URL), not as an invalid endpoint.
        with patch.dict(
            "os.environ",
            {"NSS_INFERENCE_KEY": "test-key", "NSS_INFERENCE_ENDPOINT": blank},
        ):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        assert not any(i.code == "inference_endpoint_invalid" for i in issues)

    def test_invalid_endpoint_takes_priority_over_warnings(self, default_config):
        # Single-dispatch match: the invalid-endpoint error is checked first, so
        # it wins over the missing-key and blank-model warnings.
        with patch.dict(
            "os.environ",
            {"NSS_INFERENCE_MODEL": "", "NSS_INFERENCE_ENDPOINT": "not-a-url"},
            clear=True,
        ):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        codes = {i.code for i in issues}
        assert codes == {"inference_endpoint_invalid"}
        assert all(i.severity == "error" for i in issues if i.code == "inference_endpoint_invalid")

    def test_missing_key_takes_priority_over_blank_model(self, default_config):
        # With a valid endpoint, the missing-key warning outranks the blank-model
        # warning.
        with patch.dict(
            "os.environ",
            {"NSS_INFERENCE_MODEL": "   ", "NSS_INFERENCE_ENDPOINT": "https://integrate.api.nvidia.com/v1"},
            clear=True,
        ):
            issues = InferenceModelCheck().run(make_ctx(config=default_config))
        codes = {i.code for i in issues}
        assert codes == {"inference_key_missing"}


@pytest.mark.unit
class TestHFModelAvailabilityCheck:
    """Tests intentionally tied to HF cache and Transformers model directory design."""

    @staticmethod
    def _allow_online_lookup(monkeypatch) -> None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    @staticmethod
    def _clear_hf_tokens(monkeypatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    @staticmethod
    def _use_cache_root(monkeypatch, cache_root: Path) -> None:
        monkeypatch.setattr(ModelRef, "_default_hf_cache_root", staticmethod(lambda: cache_root))

    def test_empty_model_ref_errors(self, pretrained_config, ctx_factory, monkeypatch):
        self._allow_online_lookup(monkeypatch)

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config("")))

        assert any(i.code == "model_ref_empty" and i.severity == "error" for i in issues)

    def test_cached_snapshot_with_required_files_is_silent(
        self, hf_cached_snapshot_factory, pretrained_config, ctx_factory, monkeypatch
    ):
        """Complete HF snapshot handling intentionally tracks Hub cache layout."""
        cache_root, _ = hf_cached_snapshot_factory()
        self._allow_online_lookup(monkeypatch)
        self._use_cache_root(monkeypatch, cache_root)

        issues = HFModelAvailabilityCheck().run(
            ctx_factory(config=pretrained_config("nvidia/Nemotron-Mini-4B-Instruct"))
        )

        assert issues == []

    def test_missing_cache_warns_when_online_lookup_allowed(
        self, tmp_path, pretrained_config, ctx_factory, monkeypatch
    ):
        """Missing-cache behavior intentionally follows HF online fallback rules."""
        self._allow_online_lookup(monkeypatch)
        self._clear_hf_tokens(monkeypatch)
        self._use_cache_root(monkeypatch, tmp_path / "empty")

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config("gpt2")))

        assert any(i.code == "hf_model_not_cached" and i.severity == "warning" for i in issues)
        assert any(i.code == "hf_token_missing" and i.severity == "warning" for i in issues)

    @pytest.mark.parametrize("env_var", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
    def test_missing_cache_with_hf_token_does_not_warn_for_token(
        self, tmp_path, env_var, pretrained_config, ctx_factory, monkeypatch
    ):
        """Token warnings are conditional on online HF access and missing token env."""
        self._allow_online_lookup(monkeypatch)
        self._use_cache_root(monkeypatch, tmp_path / "empty")
        monkeypatch.setenv(env_var, "hf_xxx")

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config("gpt2")))

        assert any(i.code == "hf_model_not_cached" and i.severity == "warning" for i in issues)
        assert not any(i.code == "hf_token_missing" for i in issues)

    @pytest.mark.parametrize("offline_var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_missing_cache_errors_when_offline(
        self, tmp_path, offline_var, value, pretrained_config, ctx_factory, monkeypatch
    ):
        """Offline errors intentionally track HF offline env-var semantics."""
        self._use_cache_root(monkeypatch, tmp_path / "empty")
        monkeypatch.setenv(offline_var, value)

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config("gpt2")))

        assert any(i.code == "hf_model_not_cached" and i.severity == "error" for i in issues)
        assert not any(i.code == "hf_token_missing" for i in issues)

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
    def test_missing_cache_warns_when_offline_env_is_falsey(
        self, tmp_path, value, pretrained_config, ctx_factory, monkeypatch
    ):
        """Falsey offline env handling intentionally tracks HF flag semantics."""
        self._clear_hf_tokens(monkeypatch)
        self._use_cache_root(monkeypatch, tmp_path / "empty")
        monkeypatch.setenv("HF_HUB_OFFLINE", value)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config("gpt2")))

        assert any(i.code == "hf_model_not_cached" and i.severity == "warning" for i in issues)
        assert any(i.code == "hf_token_missing" and i.severity == "warning" for i in issues)

    def test_partial_cached_snapshot_warns(
        self, hf_cached_snapshot_factory, pretrained_config, ctx_factory, monkeypatch
    ):
        """Partial HF snapshot validation intentionally tracks Transformers load needs."""
        cache_root, _ = hf_cached_snapshot_factory(files=("config.json", "tokenizer.json"))
        self._allow_online_lookup(monkeypatch)
        self._use_cache_root(monkeypatch, cache_root)

        issues = HFModelAvailabilityCheck().run(
            ctx_factory(config=pretrained_config("nvidia/Nemotron-Mini-4B-Instruct"))
        )

        assert any(i.code == "hf_model_cache_incomplete" and i.severity == "warning" for i in issues)
        assert "model weights" in _issue_by_code(issues, "hf_model_cache_incomplete").message

    def test_incomplete_sharded_cached_snapshot_warns(
        self, hf_cached_snapshot_factory, hf_weight_index_factory, pretrained_config, ctx_factory, monkeypatch
    ):
        """Sharded snapshot validation intentionally tracks HF weight-index design."""
        cache_root, snapshot = hf_cached_snapshot_factory(
            files=("config.json", "tokenizer.json", "model-00001-of-00002.safetensors"),
        )
        hf_weight_index_factory(
            snapshot,
            shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
        )
        self._allow_online_lookup(monkeypatch)
        self._use_cache_root(monkeypatch, cache_root)

        issues = HFModelAvailabilityCheck().run(
            ctx_factory(config=pretrained_config("nvidia/Nemotron-Mini-4B-Instruct"))
        )

        assert any(i.code == "hf_model_cache_incomplete" and i.severity == "warning" for i in issues)

    @pytest.mark.parametrize("offline_var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_partial_cached_snapshot_errors_when_offline(
        self, hf_cached_snapshot_factory, offline_var, pretrained_config, ctx_factory, monkeypatch
    ):
        """Offline incomplete-cache severity intentionally tracks HF offline env-var semantics."""
        cache_root, _ = hf_cached_snapshot_factory(files=("config.json", "tokenizer.json"))
        self._use_cache_root(monkeypatch, cache_root)
        monkeypatch.setenv(offline_var, "1")

        issues = HFModelAvailabilityCheck().run(
            ctx_factory(config=pretrained_config("nvidia/Nemotron-Mini-4B-Instruct"))
        )

        assert any(i.code == "hf_model_cache_incomplete" and i.severity == "error" for i in issues)
        assert not any(i.code == "hf_token_missing" for i in issues)

    @pytest.mark.parametrize("offline_var", ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"])
    def test_missing_trusted_remote_code_errors_when_offline(
        self, hf_cached_snapshot_factory, offline_var, pretrained_config, ctx_factory, monkeypatch
    ):
        """Missing remote-code files intentionally track Transformers auto-map design."""
        cache_root, snapshot = hf_cached_snapshot_factory()
        (snapshot / "config.json").write_text(
            json.dumps(
                {
                    "auto_map": {
                        "AutoConfig": "configuration_nemotron.NemotronConfig",
                        "AutoModelForCausalLM": "modeling_nemotron.NemotronForCausalLM",
                    }
                }
            )
        )
        self._use_cache_root(monkeypatch, cache_root)
        monkeypatch.setenv(offline_var, "1")

        issues = HFModelAvailabilityCheck().run(
            ctx_factory(config=pretrained_config("nvidia/Nemotron-Mini-4B-Instruct"))
        )

        issue = _issue_by_code(issues, "hf_remote_code_not_cached")
        assert issue.severity == "error"
        assert "configuration_nemotron.py" in issue.message
        assert "modeling_nemotron.py" in issue.message
        assert not any(i.code == "hf_token_missing" for i in issues)

    def test_missing_trusted_remote_code_warns_when_online_allowed(
        self, hf_cached_snapshot_factory, pretrained_config, ctx_factory, monkeypatch
    ):
        """Missing remote-code files warn when HF downloads are still allowed."""
        cache_root, snapshot = hf_cached_snapshot_factory()
        (snapshot / "config.json").write_text(
            json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_nemotron.NemotronForCausalLM"}})
        )
        self._allow_online_lookup(monkeypatch)
        self._clear_hf_tokens(monkeypatch)
        self._use_cache_root(monkeypatch, cache_root)

        issues = HFModelAvailabilityCheck().run(
            ctx_factory(config=pretrained_config("nvidia/Nemotron-Mini-4B-Instruct"))
        )

        issue = _issue_by_code(issues, "hf_remote_code_not_cached")
        assert issue.severity == "warning"
        assert "modeling_nemotron.py" in issue.message
        assert any(i.code == "hf_token_missing" and i.severity == "warning" for i in issues)

    def test_cached_snapshot_with_trusted_remote_code_files_is_silent(
        self, hf_cached_snapshot_factory, pretrained_config, ctx_factory, monkeypatch
    ):
        """Complete trusted remote-code snapshots do not require HF access."""
        cache_root, snapshot = hf_cached_snapshot_factory()
        (snapshot / "config.json").write_text(
            json.dumps({"auto_map": {"AutoModelForCausalLM": "modeling_nemotron.NemotronForCausalLM"}})
        )
        (snapshot / "modeling_nemotron.py").write_text("class NemotronForCausalLM: pass\n")
        self._allow_online_lookup(monkeypatch)
        self._use_cache_root(monkeypatch, cache_root)

        issues = HFModelAvailabilityCheck().run(
            ctx_factory(config=pretrained_config("nvidia/Nemotron-Mini-4B-Instruct"))
        )

        assert issues == []

    def test_existing_local_model_directory_requires_model_files(
        self, tmp_path, model_files_factory, pretrained_config, ctx_factory, monkeypatch
    ):
        """Local directory validation intentionally mirrors Transformers model layout."""
        model_dir = model_files_factory(tmp_path / "local-model", files=("config.json",))
        self._allow_online_lookup(monkeypatch)

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config(str(model_dir))))

        assert any(i.code == "local_model_incomplete" and i.severity == "error" for i in issues)
        issue = _issue_by_code(issues, "local_model_incomplete")
        assert "tokenizer" in issue.message
        assert "model weights" in issue.message

    def test_missing_path_like_model_errors(self, tmp_path, pretrained_config, ctx_factory, monkeypatch):
        model_dir = tmp_path / "missing-model"
        self._allow_online_lookup(monkeypatch)

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config(str(model_dir))))

        assert any(i.code == "local_model_missing" and i.severity == "error" for i in issues)

    def test_existing_local_model_file_errors(self, tmp_path, pretrained_config, ctx_factory, monkeypatch):
        model_file = tmp_path / "model.safetensors"
        model_file.write_text("cached")
        self._allow_online_lookup(monkeypatch)

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config(str(model_file))))

        assert any(i.code == "local_model_not_directory" and i.severity == "error" for i in issues)

    def test_invalid_model_ref_errors(self, pretrained_config, ctx_factory, monkeypatch):
        self._allow_online_lookup(monkeypatch)

        issues = HFModelAvailabilityCheck().run(ctx_factory(config=pretrained_config("not-a-valid/repo##id")))

        assert any(i.code == "model_ref_invalid" and i.severity == "error" for i in issues)


@pytest.mark.unit
class TestGroupbyColumnCheck:
    def test_happy_path(self, sample_df, default_config):
        issues = GroupbyColumnCheck().run(make_ctx(config=default_config, data=sample_df))
        assert not any(i.severity == "error" for i in issues)

    def test_missing_group_by_column(self, sample_df):
        config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by="nonexistent_col"))
        issues = GroupbyColumnCheck().run(make_ctx(config=config, data=sample_df))
        assert any(i.code == "column_not_found" for i in issues)

    def test_nulls_in_group_by_column(self):
        df = pd.DataFrame({"grp": [1, None, 3], "val": [10, 20, 30]})
        config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by="grp"))
        issues = GroupbyColumnCheck().run(make_ctx(config=config, data=df))
        assert any(i.code == "column_nulls" for i in issues)


@pytest.mark.unit
class TestOrderbyColumnCheck:
    def test_missing_order_by_column(self, sample_df):
        config = SafeSynthesizerParameters(
            data=DataParameters(group_training_examples_by="category", order_training_examples_by="nonexistent_col")
        )
        issues = OrderbyColumnCheck().run(make_ctx(config=config, data=sample_df))
        assert any(i.code == "column_not_found" for i in issues)

    def test_timeseries_with_generated_timestamp_bypasses_missing_order_by(self, sample_df):
        config = SafeSynthesizerParameters(
            data=DataParameters(
                group_training_examples_by="category",
                order_training_examples_by="generated_ts",
            ),
            time_series=TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=60),
        )
        issues = OrderbyColumnCheck().run(make_ctx(config=config, data=sample_df))
        assert not any(i.code == "column_not_found" and "generated_ts" in i.message for i in issues)


@pytest.mark.unit
class TestTimestampColumnCheck:
    def test_disabled_when_not_timeseries(self, default_config):
        """``enabled()`` gates the whole check when ``is_timeseries`` is False."""
        assert TimestampColumnCheck().enabled(make_ctx(config=default_config)) is False

    def test_enabled_for_timeseries(self):
        config = SafeSynthesizerParameters(
            data=DataParameters(group_training_examples_by="grp"),
            time_series=TimeSeriesParameters(
                is_timeseries=True,
                timestamp_column="ts",
                timestamp_interval_seconds=60,
            ),
        )
        assert TimestampColumnCheck().enabled(make_ctx(config=config)) is True

    def test_missing_column_reports_error(self):
        df = pd.DataFrame({"val": [1, 2, 3]})
        config = SafeSynthesizerParameters(
            data=DataParameters(group_training_examples_by="grp"),
            time_series=TimeSeriesParameters(
                is_timeseries=True,
                timestamp_column="missing_ts",
                timestamp_interval_seconds=60,
            ),
        )
        issues = TimestampColumnCheck().run(make_ctx(config=config, data=df))
        assert any(i.code == "timestamp_not_found" and i.severity == "error" for i in issues)


@pytest.mark.unit
class TestTimeSeriesDataShapeCheck:
    @staticmethod
    def _make_config(**time_series_overrides):
        return SafeSynthesizerParameters(
            data=DataParameters(group_training_examples_by="grp"),
            time_series=TimeSeriesParameters(
                is_timeseries=True,
                timestamp_column="ts",
                **time_series_overrides,
            ),
        )

    def test_disabled_when_not_timeseries(self, default_config):
        assert TimeSeriesDataShapeCheck().enabled(make_ctx(config=default_config)) is False

    def test_missing_timestamp_prerequisite_is_not_duplicated(self):
        df = pd.DataFrame({"grp": ["A", "A"], "value": [1, 2]})
        config = self._make_config()

        assert TimeSeriesDataShapeCheck().enabled(make_ctx(config=config, data=df)) is False
        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert issues == []

    def test_missing_timestamp_prerequisite_omits_shape_from_full_preflight(self):
        df = pd.DataFrame({"grp": ["A", "A"], "value": [1, 2]})
        config = self._make_config()

        report = run_preflight(df, config, MagicMock(spec=ModelMetadata), stages=frozenset({PreflightStage.DATAFRAME}))
        by_name = {c.name: c for c in report.checks}

        assert by_name["timeseries.timestamp"].status == "failed"
        assert "timeseries.shape" not in by_name

    def test_empty_timeseries_reports_structured_error(self):
        df = pd.DataFrame({"value": []})
        config = SafeSynthesizerParameters(
            time_series=TimeSeriesParameters(is_timeseries=True, timestamp_interval_seconds=60),
        )

        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert any(i.code == "timeseries_empty" and i.severity == "error" for i in issues)
        assert not any(i.code == "preflight.check_crash" for i in issues)

    def test_mixed_timestamp_formats_report_parse_failure(self):
        df = pd.DataFrame(
            {
                "grp": ["A", "A", "B", "B"],
                "ts": ["01/01/2024", "01/02/2024", "01/01/2024", "01/2024"],
                "value": [1, 2, 3, 4],
            }
        )
        config = self._make_config()

        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert any(i.code == "timestamp_parse_failed" and i.severity == "error" for i in issues)

    def test_explicit_timestamp_format_mismatch_reports_error(self):
        df = pd.DataFrame({"grp": ["A", "A"], "ts": ["2024-01-01", "2024-01-02"], "value": [1, 2]})
        config = self._make_config(timestamp_format="%m/%d/%Y")

        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert any(i.code == "timestamp_format_mismatch" and i.severity == "error" for i in issues)

    def test_elapsed_seconds_requires_numeric_timestamp(self):
        df = pd.DataFrame({"grp": ["A", "A"], "ts": ["0", "60"], "value": [1, 2]})
        config = self._make_config(timestamp_format="elapsed_seconds")

        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert any(i.code == "timestamp_elapsed_non_numeric" and i.severity == "error" for i in issues)

    @pytest.mark.parametrize(
        "values,expected_check,expected_code",
        [
            pytest.param([True, False, True], "timeseries.shape", "timestamp_elapsed_invalid", id="boolean"),
            pytest.param([0.0, float("nan"), 60.0], "timeseries.timestamp", "timestamp_nulls", id="nan"),
            pytest.param([0.0, float("inf"), 60.0], "timeseries.shape", "timestamp_elapsed_invalid", id="pos_inf"),
            pytest.param([0.0, float("-inf"), 60.0], "timeseries.shape", "timestamp_elapsed_invalid", id="neg_inf"),
        ],
    )
    def test_invalid_elapsed_second_values_report_stable_preflight_codes(self, values, expected_check, expected_code):
        df = pd.DataFrame({"grp": ["A", "A", "A"], "ts": values, "value": [1, 2, 3]})
        config = self._make_config(timestamp_format="elapsed_seconds")

        report = run_preflight(df, config, MagicMock(spec=ModelMetadata), stages=frozenset({PreflightStage.DATAFRAME}))

        assert any(
            issue.check == expected_check and issue.code == expected_code and issue.severity == "error"
            for issue in report.issues
        )
        assert not any(issue.code == "preflight.check_crash" for issue in report.issues)

    def test_interval_mismatch_reports_error(self):
        df = pd.DataFrame(
            {
                "grp": ["A", "A", "A", "B", "B", "B"],
                "ts": [
                    "2024-01-01 00:00:00",
                    "2024-01-01 01:00:00",
                    "2024-01-01 02:00:00",
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:30:00",
                    "2024-01-01 02:00:00",
                ],
                "value": [1, 2, 3, 4, 5, 6],
            }
        )
        config = self._make_config(timestamp_format="%Y-%m-%d %H:%M:%S")

        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert any(i.code == "timestamp_interval_mismatch" and i.severity == "error" for i in issues)

    def test_group_length_mismatch_reports_error(self):
        df = pd.DataFrame(
            {
                "grp": ["A", "A", "A", "B", "B"],
                "ts": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"],
                "value": [1, 2, 3, 4, 5],
            }
        )
        config = self._make_config(timestamp_format="%Y-%m-%d")

        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert any(i.code == "timeseries_group_length_mismatch" and i.severity == "error" for i in issues)

    def test_start_mismatch_reports_error(self):
        df = pd.DataFrame(
            {
                "grp": ["A", "A", "B", "B"],
                "ts": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"],
                "value": [1, 2, 3, 4],
            }
        )
        config = self._make_config(timestamp_format="%Y-%m-%d")

        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert any(i.code == "timeseries_start_mismatch" and i.severity == "error" for i in issues)

    def test_stop_mismatch_reports_error(self):
        df = pd.DataFrame(
            {
                "grp": ["A", "A", "B", "B"],
                "ts": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-03"],
                "value": [1, 2, 3, 4],
            }
        )
        config = self._make_config(timestamp_format="%Y-%m-%d")

        issues = TimeSeriesDataShapeCheck().run(make_ctx(config=config, data=df))

        assert any(i.code == "timeseries_stop_mismatch" and i.severity == "error" for i in issues)


@pytest.mark.unit
class TestConstantColumnCheck:
    def test_constant_column_is_flagged(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        issues = ConstantColumnCheck().run(make_ctx(config=SafeSynthesizerParameters(), data=df))
        assert any(i.code == "constant_column" for i in issues)


@pytest.mark.unit
class TestPseudoColumnCheck:
    def test_dataset_using_reserved_pseudo_column_is_flagged(self):
        df = pd.DataFrame({PSEUDO_GROUP_COLUMN: [1, 2, 3], "val": [10, 20, 30]})
        issues = PseudoColumnCheck().run(make_ctx(config=SafeSynthesizerParameters(), data=df))
        assert any(i.code == "pseudo_column_collision" for i in issues)

    def test_multiindex_columns_are_flagged_as_unsupported_schema(self):
        df = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples([("a", "x"), ("b", "y")]))
        issues = PseudoColumnCheck().run(make_ctx(config=SafeSynthesizerParameters(), data=df))
        assert any(
            i.code == "pseudo_column_collision" and "MultiIndex columns are not supported" in i.message for i in issues
        )


@pytest.mark.unit
class TestTokenBudgetCheck:
    @staticmethod
    def _metadata(tokenizer, *, max_seq_length: int) -> MagicMock:
        metadata = MagicMock(spec=ModelMetadata)
        metadata.tokenizer = tokenizer
        metadata.max_seq_length = max_seq_length
        metadata.instruction = "Generate: "
        metadata.prompt_config = SimpleNamespace(template="{instruction}{schema}{prefill}")
        return metadata

    def test_tokenizer_unavailable(self, sample_df, default_config):
        metadata = MagicMock(spec=ModelMetadata)
        metadata.tokenizer = None
        issues = TokenBudgetCheck().run(make_ctx(config=default_config, data=sample_df, metadata=metadata))
        assert any(i.code == "tokenizer_unavailable" for i in issues)

    def test_happy_path(self, sample_df, default_config):
        # Pydantic v2 Field attributes are absent from the class __dict__ so
        # MagicMock's spec blocks auto-attribute chains on ``tokenizer``;
        # attach the nested mock explicitly before configuring it.
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(50))
        metadata = self._metadata(tokenizer, max_seq_length=2048)
        issues = TokenBudgetCheck().run(make_ctx(config=default_config, data=sample_df, metadata=metadata))
        assert not any(i.severity == "error" for i in issues)

    def test_does_not_require_columns_groupby(self):
        # Regression: ``requires = ("columns.groupby",)`` used to cause
        # schema/record checks to be skipped when the groupby column was
        # missing, even though only the per-group branch actually depends
        # on it. The per-group branch self-gates on ``group_col in data``.
        assert "columns.groupby" not in TokenBudgetCheck.requires

    def test_batch_tokenizer_flags_oversized_record(self, default_config):
        """Batch-tokenizer path is exercised: an oversized record trips record_exceeds_context.

        Budget = max_seq_length - len(schema_prompt) - 2*NUM_SPECIAL_TOKENS
               = 60 - 20 - 4 = 36 tokens per record.

        The batch ``tokenizer(...)`` call returns a single 100-token record, so
        the budget must flag it. If the batch path were skipped, fallback
        ``encode()`` would return 20 tokens (under budget) and no error would
        be raised — so a positive ``record_exceeds_context`` assertion proves
        both that the batch path was used and that the budget math is correct.
        """
        df = pd.DataFrame({"a": [1]})
        tokenizer = MagicMock()
        tokenizer.encode.return_value = list(range(20))
        tokenizer.return_value = {"input_ids": [list(range(100))]}
        metadata = self._metadata(tokenizer, max_seq_length=60)

        issues = TokenBudgetCheck().run(make_ctx(config=default_config, data=df, metadata=metadata))

        assert any(i.code == "record_exceeds_context" and i.severity == "error" for i in issues)

    def test_sampled_record_budget_excludes_pseudo_group_column(self, default_config):
        df = pd.DataFrame({PSEUDO_GROUP_COLUMN: ["synthetic-group"], "value": ["visible"]})
        metadata = self._metadata(_PseudoColumnSensitiveTokenizer(), max_seq_length=10)

        issues = TokenBudgetCheck().run(make_ctx(config=default_config, data=df, metadata=metadata))

        assert not any(i.code == "record_exceeds_context" for i in issues)

    def test_group_budget_excludes_pseudo_group_column_when_grouping_by_it(self):
        config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by=PSEUDO_GROUP_COLUMN))
        df = pd.DataFrame({PSEUDO_GROUP_COLUMN: ["group-1", "group-1"], "value": ["visible", "also-visible"]})
        metadata = self._metadata(_PseudoColumnSensitiveTokenizer(), max_seq_length=10)
        check = TokenBudgetCheck()
        check.token_sample_size = 0
        check.top_groups_to_check = 1

        issues = check.run(make_ctx(config=config, data=df, metadata=metadata))

        assert not any(i.code == "group_exceeds_context" for i in issues)


@pytest.mark.unit
class TestDatasetSizeCheck:
    def test_happy_path(self, sample_df, default_config):
        issues = DatasetSizeCheck().run(make_ctx(config=default_config, data=sample_df))
        assert not any(i.severity == "error" for i in issues)

    @pytest.mark.parametrize("df_fixture", ["tiny_df", "small_df"])
    def test_below_threshold_emits_error(self, df_fixture, default_config, request):
        df = request.getfixturevalue(df_fixture)
        issues = DatasetSizeCheck().run(make_ctx(config=default_config, data=df))
        assert any(i.code == "dataset_too_small" and i.severity == "error" for i in issues)


@pytest.mark.unit
class TestDatasetRowCountCheck:
    def test_happy_path(self, default_config):
        df = pd.DataFrame({"val": range(2000)})
        issues = SmallDatasetCheck().run(make_ctx(config=default_config, data=df))
        assert not any(i.code == "dataset_small" for i in issues)

    def test_warns_between_error_floor_and_comfort_threshold(self, default_config):
        df = pd.DataFrame({"val": range(500)})
        issues = SmallDatasetCheck().run(make_ctx(config=default_config, data=df))
        assert any(i.code == "dataset_small" and i.severity == "warning" for i in issues)


@pytest.mark.unit
class TestOversamplingCheck:
    def test_extreme_oversampling_is_flagged(self, sample_df):
        config = SafeSynthesizerParameters(training=TrainingHyperparams(num_input_records_to_sample=50000))
        issues = OversamplingCheck().run(make_ctx(config=config, data=sample_df))
        assert any(i.code == "extreme_oversampling" for i in issues)


# ---------------------------------------------------------------------------
# End-to-end: run_preflight
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunPreflight:
    @pytest.fixture(autouse=True)
    def _isolate_hf_offline_env(self, monkeypatch):
        # run_preflight invokes HFModelAvailabilityCheck, which escalates
        # hf_model_not_cached to an error when HF offline mode is enabled. Clear
        # the ambient offline vars so these tests do not fail when the developer
        # (or CI) has HF_HUB_OFFLINE set and the model is not cached.
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    def test_clean_dataset_has_no_errors(self, sample_df, default_config):
        resolved_config = default_config.model_copy(
            update={
                "training": default_config.training.model_copy(update={"num_input_records_to_sample": len(sample_df)})
            }
        )
        metadata = MagicMock(spec=ModelMetadata)
        metadata.tokenizer = MagicMock()
        metadata.tokenizer.encode.return_value = list(range(50))
        metadata.max_seq_length = 2048
        metadata.instruction = "Generate: "
        metadata.prompt_config = SimpleNamespace(template="{instruction}{schema}{prefill}")
        with patch("torch.cuda.is_available", return_value=True):
            with patch.dict("os.environ", {"NSS_INFERENCE_KEY": "test", "HF_TOKEN": "hf_xxx"}):
                report = run_preflight(sample_df, resolved_config, metadata)
        assert len(report.errors) == 0
        by_name = {c.name: c for c in report.checks}
        assert "timeseries.timestamp" not in by_name
        assert "timeseries.shape" not in by_name

    def test_result_status_reflects_outcome(self, sample_df, default_config):
        """``PreflightCheckResult.status`` is populated for all three outcomes."""
        # Use a dataset below the hard size floor so ``dataset.size`` errors
        # and ``dataset.row_count`` (which declares ``requires=("dataset.size",)``)
        # is marked skipped. Exercises ``failed``, ``skipped``, and ``passed``
        # in one registry run.
        df = pd.DataFrame({"val": list(range(50))})
        metadata = MagicMock(spec=ModelMetadata)
        with patch("torch.cuda.is_available", return_value=True):
            report = run_preflight(df, default_config, metadata)
        by_name = {c.name: c for c in report.checks}
        # Failed: dataset_too_small fires below the floor.
        assert by_name["dataset.size"].status == "failed"
        # Skipped: dataset.row_count depends on dataset.size.
        assert by_name["dataset.row_count"].status == "skipped"
        # Passed: pseudo-column check runs cleanly on this data.
        assert by_name["columns.pseudo"].status == "passed"

    def test_token_budget_runs_when_columns_groupby_errors(self):
        # Regression for H2: ``TokenBudgetCheck`` used to declare
        # ``requires=("columns.groupby",)`` and so was skipped whenever
        # the group column was missing -- even though the schema-prompt
        # and sampled-record sub-checks have nothing to do with grouping.
        # Now the per-group branch self-gates, and the schema/record
        # branches always run.
        df = pd.DataFrame({"val": [1, 2, 3]})
        config = SafeSynthesizerParameters(data=DataParameters(group_training_examples_by="missing_col"))
        metadata = MagicMock(spec=ModelMetadata)
        metadata.tokenizer = None  # forces the tokenizer_unavailable branch
        with patch("torch.cuda.is_available", return_value=True):
            report = run_preflight(df, config, metadata)
        by_name = {c.name: c for c in report.checks}
        assert any(i.code == "column_not_found" for i in report.issues)
        assert by_name["columns.groupby"].status == "failed"
        # token_budget ran (status is "passed" — the tokenizer_unavailable
        # warning is the only issue and it's a warning, not an error).
        assert by_name["token_budget"].status == "passed"
        assert any(i.code == "tokenizer_unavailable" for i in by_name["token_budget"].issues)

    def test_dataset_row_count_is_skipped_when_dataset_size_errors(self):
        """``DatasetRowCountCheck`` declares ``requires=("dataset.size",)``.

        When a dataset is under the hard floor, ``DatasetSizeCheck`` errors and
        the advisory row-count check must be marked skipped with no issues --
        otherwise both would fire for the same underlying condition.
        """
        df = pd.DataFrame({"val": list(range(50))})  # well below the 200-row floor
        config = SafeSynthesizerParameters()
        metadata = MagicMock(spec=ModelMetadata)
        with patch("torch.cuda.is_available", return_value=True):
            report = run_preflight(df, config, metadata)
        by_name = {c.name: c for c in report.checks}
        assert by_name["dataset.size"].status == "failed"
        assert any(i.code == "dataset_too_small" and i.check == "dataset.size" for i in by_name["dataset.size"].issues)
        assert by_name["dataset.row_count"].status == "skipped"
        assert not by_name["dataset.row_count"].issues


# ---------------------------------------------------------------------------
# render_preflight_report
# ---------------------------------------------------------------------------


def _make_report(*check_tuples: tuple) -> tuple[PreflightReport, PreflightRegistry]:
    """Build a ``(report, registry)`` pair from ``(name, label, issues[, status])`` tuples.

    ``status`` defaults to ``"failed"`` when issues contain errors,
    ``"passed"`` otherwise; pass a fourth tuple element to force it
    (e.g. ``"skipped"``). The returned registry is a ``PreflightRegistry``
    backed by ``SimpleNamespace`` stand-ins with ``name``, ``label``, and
    ``category`` attributes -- exactly what the renderer consumes.
    """
    from types import MappingProxyType, SimpleNamespace

    from nemo_safe_synthesizer.preflight.types import PreflightCheckResult

    results: list[PreflightCheckResult] = []
    checks_map: dict = {}
    for tup in check_tuples:
        if len(tup) == 4:
            name, label, issues, status = tup
        else:
            name, label, issues = tup
            status = "failed" if any(i.severity == "error" for i in issues) else "passed"
        results.append(PreflightCheckResult(name=name, status=status, issues=list(issues)))
        checks_map[name] = SimpleNamespace(name=name, label=label, category="data quality")
    return PreflightReport(checks=results), PreflightRegistry(checks=MappingProxyType(checks_map))


def _make_render_context(**kwargs) -> PreflightRenderContext:
    """Build a `PreflightRenderContext` for rendering tests."""
    return PreflightRenderContext(**kwargs)


def _capture_render(
    report_and_registry: tuple[PreflightReport, PreflightRegistry],
    context: PreflightRenderContext | None = None,
) -> str:
    """Render ``render_preflight_report`` to a plain-text string for assertion."""
    report, registry = report_and_registry
    buf = StringIO()
    render_preflight_report(
        report,
        registry=registry,
        context=context,
        console=Console(file=buf, force_terminal=False, no_color=True),
    )
    return buf.getvalue()


_DEFAULT_RENDER_CONTEXT = dict(
    config_path=Path("/tmp/config.yaml"),
    data_source="/data.csv",
    artifact_dir=Path("/tmp/artifacts"),
)


@pytest.mark.unit
class TestRenderPreflightReport:
    def test_no_issues_renders_success_banner_and_follow_up_command(self):
        r = _make_report(("gpu", "GPU resources", []), ("env", "Environment variables", []))
        output = _capture_render(r, context=_make_render_context(**_DEFAULT_RENDER_CONTEXT))
        assert "passed" in output
        assert "GPU resources" in output
        assert "/tmp/artifacts" in output
        assert "resolved config" in output
        assert "safe-synthesizer run" in output
        assert "/data.csv" in output

    def test_errors_render_message_and_suppress_follow_up_command(self):
        issues = [PreflightIssue("no_gpu", "error", "gpu", "No GPU")]
        r = _make_report(("gpu", "GPU resources", issues))
        output = _capture_render(r, context=_make_render_context(**_DEFAULT_RENDER_CONTEXT))
        assert "No GPU" in output
        assert "GPU resources" in output
        assert "/tmp/artifacts" in output
        # Errors block the follow-up command — don't invite the user to proceed.
        assert "safe-synthesizer run" not in output

    def test_warnings_render_message_and_still_offer_follow_up_command(self):
        issues = [PreflightIssue("dataset_small", "warning", "size", "Small dataset")]
        r = _make_report(("size", "Dataset size", issues), ("gpu", "GPU resources", []))
        output = _capture_render(r, context=_make_render_context(**_DEFAULT_RENDER_CONTEXT))
        assert "Small dataset" in output
        assert "GPU resources" in output
        assert "/tmp/artifacts" in output
        # Warnings don't block the follow-up command.
        assert "safe-synthesizer run" in output

    def test_shows_every_check_label(self):
        r = _make_report(
            ("gpu", "GPU resources", []),
            ("env", "Environment variables", [PreflightIssue("hf", "warning", "env", "no token")]),
            ("config", "Configuration", []),
        )
        output = _capture_render(r)
        assert "GPU resources" in output
        assert "Environment variables" in output
        assert "Configuration" in output

    def test_issue_code_appears_alongside_message(self):
        """Renderer must emit the machine-readable ``issue.code`` so users can
        cross-reference `docs/user-guide/troubleshooting.md` from the CLI.
        """
        issues = [PreflightIssue("no_gpu", "error", "gpu", "No GPU")]
        r = _make_report(("gpu", "GPU resources", issues))
        output = _capture_render(r, context=_make_render_context(**_DEFAULT_RENDER_CONTEXT))
        assert "no_gpu" in output, "Issue code must appear in rendered output"
        assert "No GPU" in output

    def test_skipped_status_renders_dim_line_and_no_issues(self):
        """Skipped-due-to-failed-dep checks render distinctly from passing checks."""
        r = _make_report(
            ("base", "Base", [PreflightIssue("boom", "error", "base", "forced")]),
            ("dep", "Dependent", [], "skipped"),
        )
        output = _capture_render(r, context=_make_render_context(**_DEFAULT_RENDER_CONTEXT))
        assert "⊘ skipped" in output, "Skipped checks must render the ⊘ glyph"
        assert "Dependent" in output

    def test_follow_up_command_quotes_unsafe_paths(self):
        """``shlex.quote`` must protect against spaces / shell metacharacters."""
        r = _make_report(
            ("gpu", "GPU resources", [PreflightIssue("w", "warning", "gpu", "ok")]),
        )
        output = _capture_render(
            r,
            context=_make_render_context(
                config_path=Path("/tmp/my configs/c.yaml"),
                data_source="/data/Customer Data.csv",
                artifact_dir=Path("/tmp/art"),
            ),
        )
        assert "'/data/Customer Data.csv'" in output
        assert "'/tmp/my configs/c.yaml'" in output

    def test_long_paths_are_not_truncated(self):
        long_path = Path(
            "/root/ss-wt-preflight/safe-synthesizer-artifacts/default---financial_transactions/2026-04-14T20:00:30"
        )
        config = long_path / "safe-synthesizer-config.yaml"
        r = _make_report()
        output = _capture_render(
            r,
            context=_make_render_context(
                config_path=config,
                data_source="/data.csv",
                artifact_dir=long_path,
            ),
        )
        assert str(long_path) in output
        assert "safe-synthesizer-config.yaml" in output
        assert str(config) in output  # full path still appears in the follow-up command


# ---------------------------------------------------------------------------
# Registry shape and determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRegistryShape:
    def test_stage_ordering_is_monotonic(self):
        """Registry entries appear in ``PreflightStage`` declaration order.

        This is the contract plugin authors rely on: a plugin subclassing
        ``DataFrameCheck`` slots after all ``ConfigCheck`` entries, etc.
        Checked on the real registry so core registrations
        can't accidentally break the invariant.
        """
        registry = get_registry()
        stage_order = list(PreflightStage)
        indices = [stage_order.index(c.stage) for c in registry]
        assert indices == sorted(indices), (
            f"Registry is not stage-monotonic: {[(c.name, c.stage.value) for c in registry]}"
        )

    def test_names_are_unique(self):
        names = [c.name for c in get_registry()]
        assert len(names) == len(set(names))

    def test_requires_reference_earlier_names(self):
        seen: set[str] = set()
        for check in get_registry():
            for dep in check.requires:
                assert dep in seen, f"{check.name} requires '{dep}' but it appears later or doesn't exist"
            seen.add(check.name)

    def test_all_stages_represented(self):
        stages_used = {c.stage for c in get_registry()}
        for stage in PreflightStage:
            assert stage in stages_used, f"Stage {stage} has no registered checks"


# ---------------------------------------------------------------------------
# Dependency gating
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDependencyGating:
    """``_run_registry`` skips dependent checks when a required check errors.

    Each test defines tiny in-file plugin classes rather than relying on the
    core registry so the gating contract is isolated from whatever checks
    happen to ship today.
    """

    @staticmethod
    def _run_results(registry):
        """Run the orchestrator against ``registry`` with validation bypassed.

        Tests in this class intentionally feed sequences that violate the
        stage-monotonic / forward-only-requires invariants enforced by
        ``build_registry``, so we wrap the tuple directly in a
        ``PreflightRegistry`` to exercise ``_run_registry`` in isolation.
        """
        from types import MappingProxyType

        from nemo_safe_synthesizer.preflight import PreflightRegistry
        from nemo_safe_synthesizer.preflight.orchestrator import _run_registry

        ctx = PreflightContext(data=pd.DataFrame(), config=MagicMock(), metadata=MagicMock())
        preg = PreflightRegistry(checks=MappingProxyType({c.name: c for c in registry}))
        return _run_registry(ctx, preg)

    def test_error_in_required_check_skips_dependents_but_not_independents(self):
        class AlwaysError(ConfigCheck):
            name = "plugintest.base"
            label = "Base"

            def check(self, ctx, collector):
                collector.error("test_err", "forced error")

        class AlwaysPass(ConfigCheck):
            name = "plugintest.dep"
            label = "Dependent"
            requires = ("plugintest.base",)

            def check(self, ctx, collector):
                return

        class Independent(ConfigCheck):
            name = "plugintest.indep"
            label = "Independent"

            def check(self, ctx, collector):
                return

        results = self._run_results((AlwaysError(), AlwaysPass(), Independent()))
        by_name = {r.name: r for r in results}
        assert by_name["plugintest.base"].status == "failed"
        assert by_name["plugintest.dep"].status == "skipped", (
            "Dependent check should be marked skipped when base has errors"
        )
        assert not by_name["plugintest.dep"].issues
        assert by_name["plugintest.indep"].status == "passed", "Independent check should still run"

    def test_warnings_only_do_not_gate_dependents(self):
        class WarnOnly(ConfigCheck):
            name = "plugintest.warn_base"
            label = "Base"

            def check(self, ctx, collector):
                collector.warning("test_warn", "just a warning")

        class AlwaysPass(ConfigCheck):
            name = "plugintest.warn_dep"
            label = "Dependent"
            requires = ("plugintest.warn_base",)

            def check(self, ctx, collector):
                return

        by_name = {r.name: r for r in self._run_results((WarnOnly(), AlwaysPass()))}
        assert by_name["plugintest.warn_base"].status == "passed", (
            "Warnings-only must leave status=passed (failed is reserved for errors)"
        )
        assert by_name["plugintest.warn_dep"].status == "passed", (
            "Dependent check must run and pass when base has only warnings"
        )

    def test_advisory_errors_do_not_gate_dependents(self):
        class AdvisoryError(AdvisoryCheck):
            name = "plugintest.advisory_base"
            label = "Advisory base"

            def check(self, ctx, collector):
                collector.error("advisory_err", "advisory error")

        class Dependent(ConfigCheck):
            name = "plugintest.advisory_dep"
            label = "Dependent"
            requires = ("plugintest.advisory_base",)

            def check(self, ctx, collector):
                return

        by_name = {r.name: r for r in self._run_results((AdvisoryError(), Dependent()))}
        assert by_name["plugintest.advisory_base"].status == "failed", (
            "Advisory check that emits an error is still failed"
        )
        assert by_name["plugintest.advisory_dep"].status == "passed", (
            "Dependent check must run despite an advisory base erroring (advisory errors do not gate)"
        )
