# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import EvalPrediction, IntervalStrategy

from nemo_safe_synthesizer.cli.artifact_structure import Workdir
from nemo_safe_synthesizer.config import (
    DataParameters,
    DifferentialPrivacyHyperparams,
    GenerateParameters,
    SafeSynthesizerParameters,
    TrainingHyperparams,
)
from nemo_safe_synthesizer.errors import ParameterError
from nemo_safe_synthesizer.training.callbacks import ProgressBarCallback, SafeSynthesizerWorkerCallback
from nemo_safe_synthesizer.training.huggingface_backend import (
    HuggingFaceBackend,
    compute_metrics,
    preprocess_logits_for_metrics,
)


@pytest.fixture
def mock_workdir(fixture_session_cache_dir: Path):
    """Create a mock workdir object."""
    workdir = Workdir(
        base_path=fixture_session_cache_dir,
        config_name="test-config",
        dataset_name="test-dataset",
        run_name="2026-01-15T12:00:00",
    )
    workdir.ensure_directories()
    return workdir


@pytest.fixture
def mock_model_metadata(mock_workdir):
    """Create a mock model metadata object."""
    metadata = MagicMock()
    metadata.workdir = mock_workdir
    metadata.adapter_path = mock_workdir.train.adapter.path
    metadata.save_path = mock_workdir.train.adapter.path
    metadata.metadata_path = mock_workdir.train.adapter.metadata
    metadata.rope_parameters_location = "autoconfig"
    return metadata


@pytest.fixture
def mock_model_metadata_with_rope_automodel(mock_workdir):
    """Create a mock model metadata object."""
    metadata = MagicMock()
    metadata.workdir = mock_workdir
    metadata.adapter_path = mock_workdir.train.adapter.path
    metadata.save_path = mock_workdir.train.adapter.path
    metadata.metadata_path = mock_workdir.train.adapter.metadata
    metadata.rope_parameters_location = "automodel"
    return metadata


@pytest.fixture
def base_params():
    """Create basic SafeSynthesizerParameters for testing."""
    return SafeSynthesizerParameters(
        data=DataParameters(
            group_training_examples_by=None,
            order_training_examples_by=None,
        ),
        training=TrainingHyperparams(
            num_input_records_to_sample=100,
            batch_size=2,
            gradient_accumulation_steps=4,
            validation_ratio=0.0,
            pretrained_model="test-model",
            quantize_model=False,
            lora_r=16,
            lora_alpha_over_r=1.0,
            lora_target_modules=["q_proj", "v_proj"],
        ),
        generation=GenerateParameters(num_records=100),
        privacy=DifferentialPrivacyHyperparams(dp_enabled=False),
    )


@pytest.fixture
def params_with_validation(base_params):
    """Create params with validation enabled."""
    base_params.training.validation_ratio = 0.1
    base_params.training.validation_steps = 10
    return base_params


@pytest.fixture
def params_with_dp(base_params):
    """Create params with differential privacy enabled."""
    base_params.privacy = DifferentialPrivacyHyperparams(
        dp_enabled=True,
        epsilon=1.0,
        delta=1e-5,
        per_sample_max_grad_norm=1.0,
    )
    return base_params


@pytest.fixture
def params_with_quantization(base_params):
    """Create params with quantization enabled."""
    base_params.training.quantize_model = True
    base_params.training.quantization_bits = 4
    return base_params


@pytest.fixture
def params_with_groupby(base_params):
    """Create params with groupby column specified."""
    base_params.data.group_training_examples_by = "group_col"
    return base_params


@pytest.fixture
def params_with_orderby(base_params):
    """Create params with orderby column specified."""
    base_params.data.order_training_examples_by = "order_col"
    return base_params


@pytest.fixture
def backend(base_params, mock_model_metadata, mock_workdir):
    """Create a HuggingFaceBackend instance for testing."""
    return HuggingFaceBackend(
        params=base_params,
        model_metadata=mock_model_metadata,
        workdir=mock_workdir,
    )


@pytest.fixture
def backend_with_validation(params_with_validation, mock_model_metadata, mock_workdir):
    """Create a HuggingFaceBackend instance with validation enabled."""
    return HuggingFaceBackend(
        params=params_with_validation,
        model_metadata=mock_model_metadata,
        workdir=mock_workdir,
    )


@pytest.fixture
def backend_with_dp(params_with_dp, mock_model_metadata, mock_workdir):
    """Create a HuggingFaceBackend instance with DP enabled."""
    backend = HuggingFaceBackend(
        params=params_with_dp,
        model_metadata=mock_model_metadata,
        workdir=mock_workdir,
    )
    backend.tokenizer = MagicMock()
    backend.true_dataset_size = 100
    backend.data_fraction = 0.5
    return backend


@pytest.fixture
def backend_with_quantization(params_with_quantization, mock_model_metadata, mock_workdir):
    """Create a HuggingFaceBackend instance with quantization enabled."""
    return HuggingFaceBackend(
        params=params_with_quantization,
        model_metadata=mock_model_metadata,
        workdir=mock_workdir,
    )


@pytest.fixture(autouse=True)
def mock_autoconfig_from_pretrained():
    """Mock AutoConfig.from_pretrained to prevent HuggingFace API calls during init."""
    mock_config = MagicMock()
    mock_config.max_position_embeddings = 2048
    mock_config.rope_parameters = {"rope_type": "linear", "factor": 1.0, "theta": 10000}
    with patch(
        "nemo_safe_synthesizer.training.huggingface_backend.AutoConfig.from_pretrained",
        return_value=mock_config,
    ):
        yield mock_config


@pytest.fixture
def mock_autoconfig():
    """Create a mock AutoConfig object for explicit use in tests."""
    config = MagicMock()
    config.max_position_embeddings = 2048
    config.rope_parameters = {"rope_type": "linear", "factor": 1.0, "theta": 10000}
    return config


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "col1": [1, 2, 3, 4, 5],
            "col2": ["a", "b", "c", "d", "e"],
            "group_col": ["g1", "g1", "g2", "g2", "g3"],
            "order_col": [1, 2, 3, 4, 5],
        }
    )


class TestFilterModelKwargs:
    def test_filters_trainer_specific_keys(self, backend):
        """Test that trainer-specific keys are filtered out."""
        kwargs = {
            "params": "should_be_removed",
            "model_metadata": "should_be_removed",
            "training_dataset": "should_be_removed",
            "device_map": "auto",
            "dtype": torch.float16,
        }
        result = backend._filter_model_kwargs(kwargs)

        assert "params" not in result
        assert "model_metadata" not in result
        assert "training_dataset" not in result
        assert result["device_map"] == "auto"
        assert result["dtype"] == torch.float16

    def test_preserves_model_kwargs(self, backend):
        """Test that non-trainer-specific keys are preserved."""
        kwargs = {
            "device_map": "cuda:0",
            "attn_implementation": "sdpa",
            "custom_param": "custom_value",
        }
        result = backend._filter_model_kwargs(kwargs)

        assert result["device_map"] == "cuda:0"
        assert result["attn_implementation"] == "sdpa"
        assert result["custom_param"] == "custom_value"

    def test_empty_kwargs(self, backend):
        """Test with empty kwargs."""
        result = backend._filter_model_kwargs({})
        assert result == {}

    def test_all_trainer_specific_keys_filtered(self, backend):
        """Test that all trainer-specific keys are filtered."""
        kwargs = {key: f"value_{key}" for key in HuggingFaceBackend._TRAINER_SPECIFIC_KEYS}
        result = backend._filter_model_kwargs(kwargs)
        assert result == {}

    def test_rope_scaling_factor_is_filtered(self, backend):
        """Test that rope_scaling_factor is filtered out.

        rope_scaling_factor should be in _TRAINER_SPECIFIC_KEYS because it's
        processed separately by _resolve_rope_scaling_factor and converted to
        the properly formatted rope_scaling dict by _apply_rope_scaling.
        """
        kwargs = {
            "rope_scaling_factor": 2.0,
            "device_map": "auto",
        }
        result = backend._filter_model_kwargs(kwargs)

        assert "rope_scaling_factor" not in result
        assert result["device_map"] == "auto"


@patch("nemo_safe_synthesizer.training.huggingface_backend.get_device_map")
class TestBuildBaseFrameworkParams:
    def test_uses_cached_model_ref_target_and_trust(
        self,
        mock_get_device_map,
        base_params,
        mock_model_metadata,
        mock_workdir,
    ):
        """Framework params use resolved model refs before HuggingFace model loading."""
        cached_target = "/tmp/hf-cache/models--nvidia--TinyLlama/snapshots/abc123"
        model_ref = MagicMock()
        model_ref.target.return_value = cached_target
        model_ref.trust_remote_code = True
        mock_get_device_map.return_value = "auto"

        with patch(
            "nemo_safe_synthesizer.training.huggingface_backend.ModelRef.parse", return_value=model_ref
        ) as parse:
            backend = HuggingFaceBackend(
                params=base_params,
                model_metadata=mock_model_metadata,
                workdir=mock_workdir,
            )

        result = backend._build_base_framework_params({})

        parse.assert_called_once_with(base_params.training.pretrained_model)
        assert result["pretrained_model_name_or_path"] == cached_target
        assert result["trust_remote_code"] is True
        mock_get_device_map.assert_called_once_with(
            cached_target,
            autoconfig=backend.autoconfig,
            trust_remote_code=True,
        )

    def test_builds_correct_params_with_kernels(self, mock_get_device_map, backend):
        """Test that base framework params use the configured default attention implementation."""
        mock_get_device_map.return_value = "auto"
        model_kwargs = {"custom_key": "custom_value"}
        with patch.dict("sys.modules", {"kernels": MagicMock()}):
            result = backend._build_base_framework_params(model_kwargs)

        assert result["pretrained_model_name_or_path"] == "test-model"
        assert result["device_map"] == "auto"
        assert result["attn_implementation"] == "sdpa"
        assert result["dtype"] == torch.bfloat16
        assert result["custom_key"] == "custom_value"

    def test_builds_correct_params_without_kernels(self, mock_get_device_map, backend):
        """Test that base framework params use the default when kernels is not available."""
        mock_get_device_map.return_value = "auto"
        model_kwargs = {"custom_key": "custom_value"}
        with patch.dict("sys.modules", {"kernels": None}):
            result = backend._build_base_framework_params(model_kwargs)

        assert result["pretrained_model_name_or_path"] == "test-model"
        assert result["device_map"] == "auto"
        assert result["attn_implementation"] == "sdpa"
        assert result["dtype"] == torch.bfloat16
        assert result["custom_key"] == "custom_value"

    def test_uses_custom_device_map(self, mock_get_device_map, backend):
        """Test that custom device_map is used when provided."""
        mock_get_device_map.return_value = "auto"
        model_kwargs = {"device_map": "cuda:1"}
        result = backend._build_base_framework_params(model_kwargs)
        assert result["device_map"] == "cuda:1"

    def test_uses_custom_attn_implementation(self, mock_get_device_map, backend):
        """Test that custom attn_implementation from kwargs overrides config."""
        mock_get_device_map.return_value = "auto"
        model_kwargs = {"device_map": "auto", "attn_implementation": "sdpa"}
        result = backend._build_base_framework_params(model_kwargs)
        assert result["attn_implementation"] == "sdpa"

    def test_uses_config_attn_implementation(self, mock_get_device_map, backend):
        """Test that attn_implementation from config is used when not in kwargs."""
        mock_get_device_map.return_value = "auto"
        backend.params.training.attn_implementation = "eager"
        model_kwargs = {"device_map": "auto"}
        result = backend._build_base_framework_params(model_kwargs)
        assert result["attn_implementation"] == "eager"
        # Reset to default
        backend.params.training.attn_implementation = "sdpa"

    def test_uses_custom_dtype(self, mock_get_device_map, backend):
        """Test that custom dtype is used when provided."""
        mock_get_device_map.return_value = "auto"
        model_kwargs = {"device_map": "auto", "dtype": torch.float32}
        result = backend._build_base_framework_params(model_kwargs)
        assert result["dtype"] == torch.float32


class TestResolvedModelLoading:
    def test_init_uses_cached_model_ref_target_for_autoconfig(self, base_params, mock_model_metadata, mock_workdir):
        """Backend initialization resolves cached model refs before loading AutoConfig."""
        cached_target = "/tmp/hf-cache/models--nvidia--TinyLlama/snapshots/abc123"
        model_ref = MagicMock()
        model_ref.target.return_value = cached_target
        model_ref.trust_remote_code = True
        autoconfig = MagicMock()

        with (
            patch("nemo_safe_synthesizer.training.huggingface_backend.ModelRef.parse", return_value=model_ref) as parse,
            patch(
                "nemo_safe_synthesizer.training.huggingface_backend.AutoConfig.from_pretrained",
                return_value=autoconfig,
            ) as from_pretrained,
        ):
            backend = HuggingFaceBackend(
                params=base_params,
                model_metadata=mock_model_metadata,
                workdir=mock_workdir,
            )

        assert backend.model_ref is model_ref
        parse.assert_called_once_with(base_params.training.pretrained_model)
        from_pretrained.assert_called_once_with(cached_target, trust_remote_code=True)

    def test_load_pretrained_model_uses_cached_model_ref_target_for_tokenizer(self, backend):
        """Tokenizer loading uses the same resolved model ref as model loading."""
        cached_target = "/tmp/hf-cache/models--nvidia--TinyLlama/snapshots/abc123"
        model_ref = MagicMock()
        model_ref.target.return_value = cached_target
        model_ref.trust_remote_code = True
        tokenizer = MagicMock()
        backend.model_ref = model_ref
        backend.framework_load_params = {
            "pretrained_model_name_or_path": cached_target,
            "trust_remote_code": True,
        }
        backend.model_loader_type = MagicMock()
        backend.model_loader_type.from_pretrained.return_value = MagicMock()
        backend.model_metadata.max_seq_length = 2048

        with (
            patch(
                "nemo_safe_synthesizer.training.huggingface_backend.load_fast_tokenizer",
                return_value=tokenizer,
            ) as load_tok,
            patch(
                "nemo_safe_synthesizer.training.huggingface_backend.add_bos_eos_tokens_to_tokenizer",
                return_value=tokenizer,
            ),
        ):
            backend._load_pretrained_model()

        load_tok.assert_called_once_with(
            cached_target,
            trust_remote_code=True,
            model_max_length=None,
        )


class TestResolveAttnImplementation:
    def test_kernels_available(self, backend):
        """Test that kernels-community path is returned when kernels is importable."""
        with patch.dict("sys.modules", {"kernels": MagicMock()}):
            result = backend._resolve_attn_implementation("sdpa")
        assert result == "sdpa"

    def test_kernels_not_available(self, backend):
        """Test that sdpa fallback is returned when kernels is not importable."""
        with patch.dict("sys.modules", {"kernels": None}):
            result = backend._resolve_attn_implementation("sdpa")
        assert result == "sdpa"

    def test_kernels_community_other_kernel(self, backend):
        """Test fallback for other kernels-community paths."""
        with patch.dict("sys.modules", {"kernels": None}):
            result = backend._resolve_attn_implementation("kernels-community/flash-attn2")
        assert result == "sdpa"

    def test_non_kernels_value_passthrough(self, backend):
        """Test that non-kernels values are passed through as-is."""
        assert backend._resolve_attn_implementation("eager") == "eager"
        assert backend._resolve_attn_implementation("sdpa") == "sdpa"
        assert backend._resolve_attn_implementation("flash_attention_2") == "flash_attention_2"


class TestGetQuantizationConfigIfEnabled:
    def test_returns_none_when_disabled(self, backend):
        """Test that None is returned when quantization is disabled."""
        result = backend._get_quantization_config_if_enabled()
        assert result is None

    @patch("nemo_safe_synthesizer.training.huggingface_backend.get_quantization_config")
    def test_returns_config_when_enabled(self, mock_get_config, backend_with_quantization):
        """Legacy bits-based path: quantization_bits=4 -> BNB_4BIT scheme."""
        from nemo_safe_synthesizer.config.training import QuantizationScheme

        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        result = backend_with_quantization._get_quantization_config_if_enabled()

        mock_get_config.assert_called_once_with(QuantizationScheme.BNB_4BIT)
        assert result == mock_config

    @patch("nemo_safe_synthesizer.training.huggingface_backend.get_quantization_config")
    def test_defaults_to_8_bits(self, mock_get_config, backend_with_quantization):
        """Legacy bits-based path: quantization_bits=8 -> BNB_8BIT scheme."""
        from nemo_safe_synthesizer.config.training import QuantizationScheme

        backend_with_quantization.params.training.quantization_bits = 8
        mock_config = MagicMock()
        mock_get_config.return_value = mock_config

        result = backend_with_quantization._get_quantization_config_if_enabled()

        mock_get_config.assert_called_once_with(QuantizationScheme.BNB_8BIT)
        assert result == mock_config

    @patch("nemo_safe_synthesizer.training.huggingface_backend.get_quantization_config")
    def test_explicit_scheme_overrides_bits(self, mock_get_config, backend_with_quantization):
        """When quantization_scheme is set, it overrides the bits-based fallback."""
        from nemo_safe_synthesizer.config.training import QuantizationScheme

        backend_with_quantization.params.training.quantization_scheme = QuantizationScheme.NVFP4
        mock_get_config.return_value = MagicMock()

        backend_with_quantization._get_quantization_config_if_enabled()

        mock_get_config.assert_called_once_with(QuantizationScheme.NVFP4)


class TestPrepareQuantizeBase:
    @pytest.mark.parametrize("scheme_name", ["FP8", "NVFP4", "MXFP4"])
    def test_loftq_rejects_non_bitsandbytes_schemes(self, backend_with_quantization, scheme_name):
        """LoftQ depends on bitsandbytes quantization schemes."""
        from nemo_safe_synthesizer.config.training import QuantizationScheme

        scheme = getattr(QuantizationScheme, scheme_name)
        backend_with_quantization.params.training.quantization_scheme = scheme
        backend_with_quantization.params.training.peft_implementation = "loftq"

        with pytest.raises(ParameterError, match="requires a bitsandbytes scheme"):
            backend_with_quantization._prepare_quantize_base()


class TestApplyRopeScaling:
    def test_applies_rope_scaling_from_metadata(self, backend):
        """Test that rope_scaling from metadata is applied."""
        from nemo_safe_synthesizer.llm.metadata import RopeScaling

        backend.model_metadata.rope_scaling = RopeScaling(rope_type="linear", factor=2.0, theta=10000.0)
        framework_params = {}
        backend._apply_rope_scaling(framework_params=framework_params)
        assert backend.autoconfig.rope_scaling == {
            "rope_type": "linear",
            "factor": 2.0,
            "theta": 10000.0,
        }

    def test_applies_rope_scaling_with_custom_theta(self, backend):
        """Test that custom theta from metadata is used in logging but HF dict uses factor/type."""
        from nemo_safe_synthesizer.llm.metadata import RopeScaling

        backend.model_metadata.rope_scaling = RopeScaling(rope_type="yarn", factor=4.0, theta=1500000.0)
        framework_params = {}
        backend._apply_rope_scaling(framework_params=framework_params)
        # HF dict format doesn't include theta, just rope_type and factor
        assert backend.autoconfig.rope_scaling == {
            "rope_type": "yarn",
            "factor": 4.0,
            "theta": 1500000.0,
        }

    def test_applies_rope_scaling_to_framework_params(
        self, base_params, mock_model_metadata_with_rope_automodel, mock_workdir
    ):
        """Test that rope_scaling is applied to framework_params when location is automodel."""
        from nemo_safe_synthesizer.llm.metadata import RopeScaling

        mock_model_metadata_with_rope_automodel.rope_scaling = RopeScaling(
            rope_type="linear", factor=2.0, theta=10000.0
        )
        backend = HuggingFaceBackend(
            params=base_params,
            model_metadata=mock_model_metadata_with_rope_automodel,
            workdir=mock_workdir,
        )
        framework_params = {}
        backend._apply_rope_scaling(framework_params=framework_params)
        assert framework_params["rope_scaling"] == {
            "rope_type": "linear",
            "factor": 2.0,
            "theta": 10000.0,
        }

    def test_does_nothing_when_none(self, backend):
        """Test that nothing is done when rope_scaling is None."""
        backend.model_metadata.rope_scaling = None
        framework_params = {}
        backend.autoconfig.rope_scaling = {"rope_type": "linear", "factor": 5.0}
        backend._apply_rope_scaling(framework_params=framework_params)
        # Should preserve existing autoconfig.rope_scaling
        assert backend.autoconfig.rope_scaling == {
            "rope_type": "linear",
            "factor": 5.0,
        }


class TestNormalizeRopeParameters:
    def test_normalizes_rope_parameters_theta_key(self, backend):
        """Test that legacy theta is copied to Transformers v5 rope_theta."""
        backend.model_metadata.rope_scaling = None
        backend.autoconfig.rope_scaling = None
        backend.autoconfig.rope_parameters = {"rope_type": "default", "theta": 50000.0}

        backend._normalize_rope_parameters()

        assert backend.autoconfig.rope_parameters == {
            "rope_type": "default",
            "theta": 50000.0,
            "rope_theta": 50000.0,
        }

    def test_normalizes_rope_parameters_from_metadata(self, backend):
        """Test that model metadata supplies theta when rope_parameters lacks it."""
        from nemo_safe_synthesizer.llm.metadata import RopeScaling

        backend.model_metadata.rope_scaling = RopeScaling(rope_type="yarn", factor=4.0, theta=1500000.0)
        backend.autoconfig.rope_scaling = {"rope_type": "yarn", "factor": 4.0}
        backend.autoconfig.rope_parameters = {"rope_type": "yarn", "factor": 4.0}

        backend._normalize_rope_parameters()

        assert backend.autoconfig.rope_parameters == {
            "rope_type": "yarn",
            "factor": 4.0,
            "rope_theta": 1500000.0,
        }


class TestBuildBaseTrainingArgs:
    def test_builds_correct_args_no_validation(self, backend):
        """Test that training args are built correctly without validation."""
        result = backend._build_base_training_args()

        assert result["output_dir"] == backend.workdir.train.cache.path
        assert result["per_device_train_batch_size"] == 2
        assert result["gradient_accumulation_steps"] == 4
        assert result["eval_strategy"] == IntervalStrategy.NO
        assert result["do_eval"] is False
        assert result["disable_tqdm"] is True

    def test_builds_correct_args_with_validation(self, backend_with_validation):
        """Test that training args are built correctly with validation."""
        result = backend_with_validation._build_base_training_args()

        assert result["eval_strategy"] == IntervalStrategy.STEPS
        assert result["do_eval"] is True


# =============================================================================
# Tests for _apply_eval_dataset_overrides
# =============================================================================


class TestApplyEvalDatasetOverrides:
    def test_applies_overrides_when_eval_dataset_present(self, backend_with_validation):
        """Test that overrides are applied when eval_dataset is present."""
        backend_with_validation.eval_dataset = MagicMock()
        training_args = {}

        backend_with_validation._apply_eval_dataset_overrides(training_args)

        assert training_args["eval_steps"] == 10
        assert training_args["eval_strategy"] == "steps"
        assert training_args["do_eval"] is True
        assert training_args["include_for_metrics"] == ["loss"]
        assert training_args["eval_accumulation_steps"] == 1

    def test_does_nothing_when_no_eval_dataset(self, backend):
        """Test that nothing is done when eval_dataset is None."""
        training_args = {}
        backend._apply_eval_dataset_overrides(training_args)
        assert training_args == {}


# =============================================================================
# Tests for _configure_dp_training
# =============================================================================


class TestConfigureDpTraining:
    def test_configures_dp_training(self, backend_with_dp):
        """Test that DP training is configured correctly."""
        training_args = {"gradient_checkpointing": True}

        data_collator = backend_with_dp._configure_dp_training(training_args)

        assert data_collator is not None
        assert training_args["remove_unused_columns"] is False
        assert training_args["max_grad_norm"] == 0.0
        assert "gradient_checkpointing" not in training_args

    def test_passes_grad_sample_mode_to_dp_trainer(self, backend_with_dp):
        """Test that DP grad_sample_mode is passed to the Opacus trainer."""
        backend_with_dp.params.privacy.grad_sample_mode = "ghost"
        training_args = {}

        backend_with_dp._configure_dp_training(training_args)

        assert backend_with_dp.trainer_type.keywords["grad_sample_mode"] == "ghost"
        assert "bf16" not in training_args

    def test_disables_model_gradient_checkpointing(self, backend_with_dp):
        """Test that DP training disables model-level gradient checkpointing."""
        model = MagicMock()
        model.config.use_cache = True
        backend_with_dp.model = model
        training_args = {"gradient_checkpointing": True}

        backend_with_dp._configure_dp_training(training_args)

        model.gradient_checkpointing_disable.assert_called_once_with()
        assert model.config.use_cache is False

    def test_raises_when_missing_data_fraction(self, backend_with_dp):
        """Test that ParameterError is raised when data_fraction is missing."""
        backend_with_dp.data_fraction = None
        training_args = {}

        with pytest.raises(ParameterError, match="data_fraction and true_dataset_size"):
            backend_with_dp._configure_dp_training(training_args)

    def test_raises_when_missing_true_dataset_size(self, backend_with_dp):
        """Test that ParameterError is raised when true_dataset_size is missing."""
        backend_with_dp.true_dataset_size = None
        training_args = {}

        with pytest.raises(ParameterError, match="data_fraction and true_dataset_size"):
            backend_with_dp._configure_dp_training(training_args)


class TestConfigureStandardTraining:
    def test_configures_standard_training(self, backend):
        """Test that standard training is configured correctly."""
        backend.tokenizer = MagicMock()
        training_args = {}

        data_collator = backend._configure_standard_training(training_args)

        assert data_collator is not None
        assert training_args["gradient_checkpointing"] is True

    def test_uses_provided_data_collator(self, backend):
        """Test that provided data_collator is used."""
        backend.tokenizer = MagicMock()
        custom_collator = MagicMock()
        training_args = {"data_collator": custom_collator}

        data_collator = backend._configure_standard_training(training_args)

        assert data_collator == custom_collator
        assert "data_collator" not in training_args


class TestConfigureTrainerCallbacks:
    @pytest.mark.parametrize(
        ("is_info_enabled", "expected_callback_type"),
        [
            (True, SafeSynthesizerWorkerCallback),
            (False, ProgressBarCallback),
        ],
    )
    def test_uses_central_logger_level_for_progress_callback(self, backend, is_info_enabled, expected_callback_type):
        """Trainer progress callback follows observability-controlled logger level."""
        trainer = MagicMock()

        with patch("nemo_safe_synthesizer.training.huggingface_backend.logger") as mock_logger:
            mock_logger.isEnabledFor.return_value = is_info_enabled
            backend._configure_trainer_callbacks(trainer, {})

        trainer.remove_callback.assert_called_once()
        callback = trainer.add_callback.call_args.args[0]
        assert isinstance(callback, expected_callback_type)


class TestApplyPreprocessing:
    def test_returns_df_when_no_executor(self, backend, sample_dataframe):
        """Test that the same DataFrame is returned when no action_executor."""
        result = backend._apply_preprocessing(sample_dataframe)
        pd.testing.assert_frame_equal(result, sample_dataframe)

    def test_applies_preprocessing(self, backend, sample_dataframe):
        """Test that preprocessing is applied when action_executor is present."""
        mock_executor = MagicMock()
        processed_df = sample_dataframe.copy()
        processed_df["new_col"] = "processed"
        mock_executor.preprocess.return_value = processed_df

        backend.action_executor = mock_executor
        result = backend._apply_preprocessing(sample_dataframe)

        mock_executor.preprocess.assert_called_once_with(sample_dataframe)
        assert "new_col" in result.columns


class TestPreprocessLogitsForMetrics:
    def test_returns_argmax_predictions(self):
        """Test that argmax of logits is returned."""
        # Create mock logits with shape (batch, seq_len, vocab_size)
        logits = torch.tensor(
            [
                [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
                [[0.0, 0.1, 0.9], [0.3, 0.3, 0.4]],
            ]
        )
        labels = torch.tensor([[1, 0], [2, 2]])

        pred_ids, returned_labels = preprocess_logits_for_metrics((logits,), labels)

        expected_pred_ids = torch.tensor([[1, 0], [2, 2]])
        assert torch.equal(pred_ids, expected_pred_ids)
        assert torch.equal(returned_labels, labels)


class TestComputeMetrics:
    def test_computes_mean_loss(self):
        """Test that mean loss is computed correctly."""
        eval_preds = MagicMock(spec=EvalPrediction)
        eval_preds.losses = np.array([0.5, 0.3, 0.4, 0.2])

        result = compute_metrics(eval_preds)

        assert "eval_loss" in result
        assert result["eval_loss"] == pytest.approx(0.35)

    def test_handles_single_loss(self):
        """Test with a single loss value."""
        eval_preds = MagicMock(spec=EvalPrediction)
        eval_preds.losses = np.array([0.42])

        result = compute_metrics(eval_preds)

        assert result["eval_loss"] == pytest.approx(0.42)


@patch("nemo_safe_synthesizer.training.huggingface_backend.get_device_map")
@patch("nemo_safe_synthesizer.training.huggingface_backend.get_max_memory_map")
@patch("nemo_safe_synthesizer.training.huggingface_backend.AutoConfig")
class TestPrepareConfigIntegration:
    def test_early_return_when_already_prepared(
        self, mock_autoconfig, mock_get_max_memory_map, mock_get_device_map, backend
    ):
        """Test that prepare_config returns early when already prepared."""
        backend.framework_load_params = {"already": "prepared"}
        backend.prepare_config()

        mock_autoconfig.from_pretrained.assert_not_called()

    def test_prepares_config_correctly(self, mock_autoconfig, mock_get_max_memory_map, mock_get_device_map, backend):
        """Test that prepare_config sets framework_load_params correctly."""
        mock_get_max_memory_map.return_value = {0: 16 * 1024**3}
        mock_get_device_map.return_value = "auto"
        mock_config = MagicMock()
        mock_autoconfig.from_pretrained.return_value = mock_config
        # Rope scaling is now handled by model_metadata, so just ensure it's None
        backend.model_metadata.rope_scaling = None

        backend.prepare_config()

        mock_get_max_memory_map.assert_called_once_with(max_vram_fraction=backend.params.training.max_vram_fraction)
        assert backend.framework_load_params is not None
        assert backend.framework_load_params["pretrained_model_name_or_path"] == "test-model"
        assert backend.framework_load_params["max_memory"] == {0: 16 * 1024**3}

    def test_prepare_config_max_vram_fraction_kwarg_overrides_config(
        self, mock_autoconfig, mock_get_max_memory_map, mock_get_device_map, backend
    ):
        """Explicit model-load VRAM fraction overrides the configured default."""
        mock_get_max_memory_map.return_value = {0: 8 * 1024**3}
        mock_get_device_map.return_value = "auto"
        mock_autoconfig.from_pretrained.return_value = MagicMock()
        backend.model_metadata.rope_scaling = None

        backend.prepare_config(max_vram_fraction=0.5)

        mock_get_max_memory_map.assert_called_once_with(max_vram_fraction=0.5)
        assert backend.framework_load_params is not None
        assert backend.framework_load_params["max_memory"] == {0: 8 * 1024**3}
        assert "max_vram_fraction" not in backend.framework_load_params


class TestPropagateMaxTokensPerExample:
    """Tests for HuggingFaceBackend._propagate_max_tokens_per_example."""

    @staticmethod
    def _make_stat(count: int, max_value: float) -> MagicMock:
        """Create a stub mirroring ``RunningStatistics`` shape."""
        stat = MagicMock()
        stat.count = count
        stat.max = max_value
        return stat

    def _set_training_examples_stats(self, backend: HuggingFaceBackend, stats: dict) -> None:
        """Attach a minimal ``training_examples`` stub exposing ``.stats``."""
        training_examples = MagicMock()
        training_examples.stats = stats
        backend.training_examples = training_examples

    def test_copies_max_onto_metadata(self, backend):
        """Populated stat is written as ``math.ceil`` onto metadata."""
        self._set_training_examples_stats(
            backend,
            {"tokens_per_example": self._make_stat(count=7, max_value=2251.5)},
        )

        backend._propagate_max_tokens_per_example()

        # math.ceil(2251.5) = 2252
        assert backend.model_metadata.max_tokens_per_example == 2252

    def test_noop_when_stat_missing(self, backend):
        """Missing ``tokens_per_example`` key leaves the metadata untouched."""
        backend.model_metadata.max_tokens_per_example = None
        self._set_training_examples_stats(backend, {"records_per_example": self._make_stat(1, 10)})

        backend._propagate_max_tokens_per_example()

        assert backend.model_metadata.max_tokens_per_example is None

    def test_noop_when_stat_unpopulated(self, backend):
        """Stat with ``count == 0`` is treated as absent."""
        backend.model_metadata.max_tokens_per_example = None
        self._set_training_examples_stats(
            backend,
            {"tokens_per_example": self._make_stat(count=0, max_value=float("-inf"))},
        )

        backend._propagate_max_tokens_per_example()

        assert backend.model_metadata.max_tokens_per_example is None

    def test_rounds_up_fractional_max(self, backend):
        """Fractional ``max`` (from running mean interactions) rounds up."""
        self._set_training_examples_stats(
            backend,
            {"tokens_per_example": self._make_stat(count=3, max_value=1000.01)},
        )

        backend._propagate_max_tokens_per_example()

        assert backend.model_metadata.max_tokens_per_example == 1001
